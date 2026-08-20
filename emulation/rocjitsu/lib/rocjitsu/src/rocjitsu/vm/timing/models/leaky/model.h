// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file model.h
/// @brief The simplest timing model that is still honest: a leaky bucket.
///
/// @details It ignores ordering entirely. Every instruction pours work into one
/// of a handful of buckets — one per functional unit, one for global memory
/// bytes, one for LDS bytes — and a dispatch takes as long as the fullest
/// bucket needs to drain. There is no timeline, no scoreboard, no cache and no
/// notion of a wavefront waiting for anything.
///
/// That makes it a throughput bound rather than a simulation: it is right when
/// a kernel is limited by a resource it saturates and optimistic when a kernel
/// is limited by latency it cannot hide. It is here as the reference for what
/// the smallest useful model looks like, and as the thing an accurate model has
/// to beat.
///
/// Everything it does not model, it says so — see the note_unmodeled() calls.
/// Its numbers all come from the architecture config file; it contains none.

#pragma once

#include "rocjitsu/vm/timing/timing_host.h"
#include "rocjitsu/vm/timing/timing_model.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu::timing::leaky {

/// @brief Everything the model reads from the config, resolved once.
///
/// @details Resolved at construction rather than per lookup so that the cost of
/// a fallback — and the log line that goes with it — is paid once, and so the
/// set of parameters this model depends on is one readable block.
struct Tuning {
  /// @brief Issue occupancy of one instruction of each class, in cycles.
  std::array<std::uint64_t, kNumInstClasses> issue_cycles{};
  /// @brief How many of each unit one compute unit has, so a bucket drains at
  ///        `compute_units * ports` per cycle's worth of issue. The entry for
  ///        FunctionalUnit::None is the front-end issue slot, which every
  ///        instruction occupies regardless of where it goes next.
  std::array<std::uint64_t, kNumFunctionalUnits> ports{};
  std::uint64_t compute_units = 1;
  /// @brief SIMD width. A wave wider than this costs a pass per extra width.
  std::uint64_t simd_lanes = 16;
  /// @brief Device-wide drain rates.
  double global_bytes_per_cycle = 1.0;
  double lds_bytes_per_cycle = 1.0;
  /// @brief Floor on a dispatch, covering everything between the packet and the
  ///        first wave that this model does not model.
  std::uint64_t dispatch_latency_cycles = 0;
  /// @brief What a tensor transfer costs when its extent could not be
  ///        recovered; see the Tensor case in on_instruction().
  std::uint64_t tensor_unknown_bytes = 0;
  double clock_ghz = 1.0;
};

Tuning resolve_tuning(const TimingHost &host);

/// @brief Work accumulated for one dispatch.
struct Buckets {
  /// @brief Issue cycles owed to each unit, summed over every wavefront.
  std::array<std::uint64_t, kNumFunctionalUnits> unit_cycles{};
  std::uint64_t global_bytes = 0;
  std::uint64_t lds_bytes = 0;
  std::uint64_t instructions = 0;
  /// @brief Workgroups the dispatch was launched with, which bounds how much of
  ///        the machine its work can actually spread over.
  ///
  /// @details Without it a grid of four workgroups would drain at the width of
  /// the whole part and come out faster than one workgroup's worth of work can
  /// possibly finish. Zero means the shape was never announced, which the model
  /// treats as no bound rather than as a bound of zero.
  std::uint64_t workgroups = 0;

  void add(const Buckets &other);
};

class LeakyBucketModel final : public TimingModel {
public:
  explicit LeakyBucketModel(const TimingHost &host);

  std::string_view name() const override { return "leaky"; }

  void on_dispatch_begin(const DispatchInfo &info) override;
  void on_instruction(const WaveRef &wave, const InstructionEvent &event) override;
  void on_barrier(std::span<const WaveRef> waves) override;
  void on_dispatch_end(const DispatchKey &key) override;
  void on_finalize() override;

  std::uint64_t device_cycles() const override {
    return device_cycles_.load(std::memory_order_relaxed);
  }
  double clock_ghz() const override { return tuning_.clock_ghz; }

  void write_report(std::string &out) const override;

  /// @brief How long the dispatch's fullest bucket takes to drain, in cycles.
  ///
  /// @details Exposed so a test can check the arithmetic against hand-built
  /// buckets without running a dispatch through the simulator.
  std::uint64_t drain_cycles(const Buckets &buckets) const;

private:
  /// @brief Per-dispatch accumulation, keyed so that dispatches on different
  ///        queues stay separate even when their ids collide.
  struct Open {
    std::string kernel_name;
    Buckets buckets;
  };

  struct KeyHash {
    std::size_t operator()(const DispatchKey &k) const {
      return (static_cast<std::size_t>(k.queue_id) << 32) ^ k.dispatch_id;
    }
  };

  struct Completed {
    std::string kernel_name;
    std::uint64_t cycles = 0;
    std::uint64_t instructions = 0;
  };

  const TimingHost &host_;
  Tuning tuning_;

  /// @brief Guards open_ and completed_.
  ///
  /// @details One device-wide lock, not a per-compute-unit shard. The host only
  /// serializes calls that name the same compute unit, and this model's state
  /// is per dispatch rather than per compute unit, so waves on different
  /// compute units of one dispatch land here concurrently. A finer scheme is
  /// possible; it is not worth it for a model whose per-instruction work is an
  /// array increment.
  mutable std::mutex mutex_;
  std::unordered_map<DispatchKey, Open, KeyHash> open_;
  std::vector<Completed> completed_;

  /// @brief The clock the guest reads. Advanced once per dispatch, by that
  ///        dispatch's drain time; it stands still in between, because this
  ///        model has an opinion about kernels and none about anything else.
  ///
  /// @details Relaxed: readers want a value that does not go backwards, which a
  /// single writer under mutex_ already guarantees, and they must not take the
  /// lock — device_cycles() is called from guest threads inside an ioctl.
  std::atomic<std::uint64_t> device_cycles_{0};
};

} // namespace rocjitsu::timing::leaky
