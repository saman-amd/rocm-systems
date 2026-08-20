// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/models/leaky/model.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rocjitsu::timing::leaky {
namespace {

/// @brief Whether a class does one unit of work per lane rather than per wave.
///
/// @details Model policy, not a property of the ISA, which is why it lives here
/// rather than beside InstClass: a model that issues a wave64 in one pass over
/// a 64-wide unit would answer differently.
constexpr bool is_lane_parallel(InstClass cls) {
  switch (unit_for_class(cls)) {
  case FunctionalUnit::VectorAlu:
  case FunctionalUnit::Transcendental:
  case FunctionalUnit::MatrixMultiply:
  case FunctionalUnit::LocalDataShare:
  case FunctionalUnit::VectorMemory:
  case FunctionalUnit::Export:
    return true;
  case FunctionalUnit::None:
  case FunctionalUnit::ScalarAlu:
  case FunctionalUnit::ScalarMemory:
  case FunctionalUnit::Branch:
  case FunctionalUnit::Count:
    return false;
  }
  return true;
}

/// @brief Largest cycle count that survives a double-to-integer conversion,
///        with room to spare below 2^63.
constexpr double kMaxRepresentableCycles = 9.0e18;

std::uint64_t div_up(std::uint64_t n, std::uint64_t d) { return d ? (n + d - 1) / d : n; }

/// @brief Bytes divided by a rate, in cycles.
///
/// @details The result is bounded before the cast, not after. A rate is a
/// config value, and a mistyped one (`1e-9`) turns a realistic byte count into
/// a double past what a uint64_t can represent, where the conversion is
/// undefined rather than merely large. Saturating is the right answer for a
/// fail-slow model anyway: an absurd rate should read as an absurdly slow
/// machine, which is noticed, not as a wrapped-around fast one, which is not.
std::uint64_t div_up(std::uint64_t n, double rate) {
  if (!(rate > 0.0) || !std::isfinite(rate))
    return n;
  const double cycles = std::ceil(static_cast<double>(n) / rate);
  if (!(cycles < kMaxRepresentableCycles))
    return std::numeric_limits<std::uint64_t>::max();
  return static_cast<std::uint64_t>(cycles);
}

} // namespace

void Buckets::add(const Buckets &other) {
  for (std::size_t i = 0; i < unit_cycles.size(); ++i)
    unit_cycles[i] += other.unit_cycles[i];
  global_bytes += other.global_bytes;
  lds_bytes += other.lds_bytes;
  instructions += other.instructions;
  workgroups = std::max(workgroups, other.workgroups);
}

Tuning resolve_tuning(const TimingHost &host) {
  Tuning t;

  // Pessimistic fallbacks throughout. These are not plausible values for the
  // parameter — they are the slowest value that is still reasonable, so that a
  // config missing the key produces a run that reads slow and is questioned,
  // rather than one that reads fast and is believed. Every one of them is
  // recorded by the host and printed at shutdown.
  for (std::size_t i = 0; i < kNumInstClasses; ++i)
    t.issue_cycles[i] = host.class_issue_cycles(static_cast<InstClass>(i));

  for (std::size_t i = 0; i < kNumFunctionalUnits; ++i) {
    std::string key = std::string(functional_unit_name(static_cast<FunctionalUnit>(i))) + ".ports";
    // One port per unit per compute unit: the fewest a unit can have and still
    // exist, so an unnamed unit drains as slowly as possible.
    t.ports[i] = std::max<std::uint64_t>(1, host.tune(key, 1));
  }

  // One compute unit, i.e. the whole grid runs on a single CU. Absurd, and
  // deliberately so — a config that does not describe the part cannot be used
  // to claim a number about it.
  t.compute_units =
      std::max<std::uint64_t>(1, host.tune("compute_units", kPessimisticComputeUnits));
  t.simd_lanes = std::max<std::uint64_t>(1, host.tune("simd_lanes", 1));
  t.global_bytes_per_cycle = std::max(1e-9, host.tune_real("global.bytes_per_cycle", 1.0));
  t.lds_bytes_per_cycle = std::max(1e-9, host.tune_real("lds.bytes_per_cycle", 1.0));
  // One microsecond at a gigahertz. Not a measurement — a floor deliberately
  // larger than any real pipeline traversal, because this parameter IS the
  // model's entire latency story and zero here reports a short kernel as free.
  // A config that names it replaces this with something defensible; a config
  // that forgets it produces a run that reads slow and says so.
  t.dispatch_latency_cycles = host.tune("dispatch_latency_cycles", 1000);
  // A whole LDS allocation, the largest a single transfer plausibly moves. Only
  // reached when the extent could not be recovered at all.
  t.tensor_unknown_bytes = host.tune("tensor_unknown_bytes", 65536);
  t.clock_ghz = host.clock_ghz();
  return t;
}

LeakyBucketModel::LeakyBucketModel(const TimingHost &host)
    : host_(host), tuning_(resolve_tuning(host)) {
  // The structural gaps, declared once because they are true of every
  // instruction this model will ever see. Each is charged the slow way where
  // there is a slow way to charge it, and each is in the run's ledger either
  // way, so a report from this model can never be read as if it covered them.
  host_.note_unmodeled("cache hierarchy (every access charged at the global rate)");
  host_.note_unmodeled("memory coalescing (every lane charged its own bytes)");
  host_.note_unmodeled("latency and its exposure (throughput bound only)");
  host_.note_unmodeled("wavefront scheduling, occupancy, and s_waitcnt stalls");
  host_.note_unmodeled("concurrent dispatches (each costed as if alone)");
}

void LeakyBucketModel::on_dispatch_begin(const DispatchInfo &info) {
  std::lock_guard lock(mutex_);
  Open &open = open_[info.key];
  open.kernel_name = info.kernel_name;
  open.buckets.workgroups = info.workgroup_count;
}

void LeakyBucketModel::on_instruction(const WaveRef &wave, const InstructionEvent &event) {
  const InstClass cls = event.effective_class;

  // A wave wider than the unit takes one pass per width. Charging by wave width
  // rather than by active lanes is the pessimistic reading of a divergent wave:
  // hardware issues the whole wave regardless, and a model that billed only the
  // active lanes would make divergence look free.
  std::uint64_t passes = 1;
  if (is_lane_parallel(cls))
    passes = div_up(std::max<std::uint64_t>(1, wave.wave_lanes), tuning_.simd_lanes);

  Buckets delta;
  delta.instructions = 1;
  delta.unit_cycles[static_cast<std::size_t>(unit_for_class(cls))] =
      tuning_.issue_cycles[static_cast<std::size_t>(cls)] * passes;

  const MemoryAccess &mem = event.memory;
  if (mem.valid()) {
    // Bytes touched, not bytes transferred: without a cache or a coalescer this
    // model cannot tell that two lanes shared a line, so it charges both. That
    // overcounts, which is the direction to overcount in.
    const std::uint64_t lanes = event.active_lanes ? event.active_lanes : event.wave_lanes;
    switch (mem.space) {
    case MemorySpace::Global:
      delta.global_bytes = lanes * mem.bytes_per_lane;
      break;
    case MemorySpace::Scalar:
      delta.global_bytes = mem.scalar_bytes;
      break;
    case MemorySpace::LocalDataShare:
      delta.lds_bytes = lanes * mem.bytes_per_lane;
      break;
    case MemorySpace::Tensor:
      for (const AddressRange &range : mem.ranges)
        delta.global_bytes += range.bytes;
      delta.lds_bytes = mem.lds_bytes;
      if (delta.global_bytes == 0 && delta.lds_bytes == 0) {
        // The observer could not recover the extent. Falling through to the
        // per-lane fallback below would bill this a wavefront's worth of bytes,
        // and a tensor instruction moves more than a wavefront has registers to
        // receive — the fallback is a cheap guess exactly where the real number
        // is large. Charge a full LDS allocation's worth on both sides instead.
        delta.global_bytes = tuning_.tensor_unknown_bytes;
        delta.lds_bytes = tuning_.tensor_unknown_bytes;
        host_.note_unmodeled("tensor transfer with unrecoverable extent");
      }
      break;
    case MemorySpace::None:
      break;
    }

    if (!mem.addresses_known) {
      // The observer issued the access but could not say where it went. Bill it
      // as a full-width global transfer — the most expensive thing it could
      // have been — because an access with no addresses is exactly where
      // guessing cheap costs nothing and silently deletes real traffic.
      //
      // This model never asks for lane addresses, so it does not reach here for
      // the ordinary case of having declined them; only for an access the
      // observer genuinely lost track of, which is rare enough that counting
      // every one is worth the map lookup.
      delta.global_bytes =
          std::max(delta.global_bytes, std::uint64_t{event.wave_lanes} * mem.bytes_per_lane);
      host_.note_unmodeled("memory access with unrecoverable addresses");
    }
  }

  if (cls == InstClass::Unknown)
    host_.note_unmodeled("unclassified instruction");

  std::lock_guard lock(mutex_);
  auto it = open_.find(wave.dispatch);
  if (it == open_.end())
    it = open_.emplace(wave.dispatch, Open{}).first;
  it->second.buckets.add(delta);
}

void LeakyBucketModel::on_barrier(std::span<const WaveRef> waves) {
  // A barrier costs the spread between the wavefronts that reach it, and this
  // model has no per-wavefront timeline to take a spread of. Nothing is charged;
  // the ledger is the only honest thing to do about it.
  if (!waves.empty())
    host_.note_unmodeled("barrier (no per-wavefront timeline to spread over)");
}

std::uint64_t LeakyBucketModel::drain_cycles(const Buckets &buckets) const {
  std::uint64_t cycles = tuning_.dispatch_latency_cycles;

  // A dispatch spreads over at most as many compute units as it has workgroups:
  // one workgroup does not run on two. Dividing by the whole part regardless
  // would let a four-workgroup grid finish in the time the machine needs to
  // retire four workgroups' work spread over hundreds of compute units, which
  // is faster than a single workgroup can possibly go.
  const std::uint64_t units = buckets.workgroups > 0
                                  ? std::min(tuning_.compute_units, buckets.workgroups)
                                  : tuning_.compute_units;

  for (std::size_t i = 0; i < kNumFunctionalUnits; ++i) {
    // FunctionalUnit::None is included rather than skipped. It stands for the
    // front-end issue slot every instruction occupies whatever unit it then
    // goes to, so waits, barriers and no-ops still cost something. Skipping it
    // would make an s_nop free, and a kernel padded with them infinitely fast.
    cycles = std::max(cycles, div_up(buckets.unit_cycles[i], units * tuning_.ports[i]));
  }

  cycles = std::max(cycles, div_up(buckets.global_bytes, tuning_.global_bytes_per_cycle));
  cycles = std::max(cycles, div_up(buckets.lds_bytes, tuning_.lds_bytes_per_cycle));
  return cycles;
}

void LeakyBucketModel::on_dispatch_end(const DispatchKey &key) {
  std::lock_guard lock(mutex_);
  auto it = open_.find(key);
  if (it == open_.end())
    return;

  const std::uint64_t cycles = drain_cycles(it->second.buckets);
  completed_.push_back({it->second.kernel_name, cycles, it->second.buckets.instructions});
  open_.erase(it);

  // The clock jumps here and stands still in between. This model has an opinion
  // about how long a kernel takes and none about anything else, so charging the
  // gaps between dispatches would be inventing a number. The guest brackets a
  // dispatch with a timestamp on each side, and both land on the right side of
  // this jump.
  device_cycles_.store(device_cycles_.load(std::memory_order_relaxed) + cycles,
                       std::memory_order_relaxed);
}

void LeakyBucketModel::on_finalize() {
  // Close out anything still open, so a run that ends without a completion
  // notification still reports the work it saw rather than dropping it.
  std::vector<DispatchKey> stragglers;
  {
    std::lock_guard lock(mutex_);
    stragglers.reserve(open_.size());
    for (const auto &[key, unused] : open_)
      stragglers.push_back(key);
  }
  for (const DispatchKey &key : stragglers)
    on_dispatch_end(key);
}

void LeakyBucketModel::write_report(std::string &out) const {
  std::lock_guard lock(mutex_);
  const double ns_per_cycle = tuning_.clock_ghz > 0.0 ? 1.0 / tuning_.clock_ghz : 0.0;
  out += "timing model 'leaky': " + std::to_string(completed_.size()) + " dispatches\n";
  for (const Completed &d : completed_) {
    out += "  " + (d.kernel_name.empty() ? std::string("?") : d.kernel_name) + ": " +
           std::to_string(d.cycles) + " cycles (" +
           std::to_string(static_cast<double>(d.cycles) * ns_per_cycle) + " ns), " +
           std::to_string(d.instructions) + " instructions\n";
  }
}

} // namespace rocjitsu::timing::leaky
