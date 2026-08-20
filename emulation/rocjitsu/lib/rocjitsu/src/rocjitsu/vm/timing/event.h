// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file event.h
/// @brief The boundary between observing rocjitsu and modelling time.
///
/// @details Nothing in this header names a rocjitsu type. rocjitsu's job is to
/// translate execution into these events; a model's job is to turn events into
/// cycles. Keeping the seam this narrow buys two things:
///
///   - A model can be driven directly from a test, with no simulator, no
///     compiled kernel and no GPU. Most of what is worth testing about a timing
///     model is how stalls compose, and building the exact event sequence by
///     hand tests that far more precisely than hoping a kernel reaches the case.
///   - A second model consumes the same events, so adding one does not mean
///     re-deriving the observation layer — which is the part that is fiddly and
///     easy to get subtly wrong.

#pragma once

#include "rocjitsu/vm/timing/inst_class.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::timing {

/// @brief Which register file a dependency names.
enum class RegisterFile : std::uint8_t { Scalar, Vector, Accumulator };

/// @brief A contiguous run of registers an instruction reads or writes.
struct RegisterRange {
  RegisterFile file = RegisterFile::Vector;
  std::uint32_t index = 0;
  std::uint32_t count = 1;
};

/// @brief The hardware counters a wavefront can wait on.
///
/// @details Which counter an operation posts to is a per-target ISA decision,
/// not something derivable from the instruction class — see
/// MemoryAccess::wait_counter.
enum class WaitCounter : std::uint8_t {
  VectorLoad,   ///< vmcnt / loadcnt. Also stores, on targets with no store counter.
  VectorStore,  ///< vscnt / storecnt, where the target separates them.
  LgkmCombined, ///< lgkmcnt, the pre-GFX11 counter shared by LDS, scalar and messages.
  LdsAndGds,    ///< dscnt, where the target splits LDS out of lgkmcnt.
  ScalarMemory, ///< kmcnt, where the target splits scalar memory out of lgkmcnt.
  Export,       ///< expcnt.
  Tensor,       ///< tensorcnt.
  Async,        ///< asynccnt.
  Count,
};

inline constexpr std::size_t kNumWaitCounters = static_cast<std::size_t>(WaitCounter::Count);

/// @brief Where a memory access went, which decides the path it is charged.
enum class MemorySpace : std::uint8_t {
  None,           ///< Not a memory access.
  Global,         ///< Global, buffer, scratch, or flat resolved to memory.
  LocalDataShare, ///< Local data share.
  Scalar,         ///< Scalar (constant) path.
  Tensor,         ///< Tensor data mover, between global memory and the LDS.
};

/// @brief A contiguous run of bytes.
///
/// @details The tensor data mover's unit of description. Per-lane addresses
/// cannot describe it: it has no lanes, and one instruction of it moves more
/// bytes than a wavefront has registers to receive.
struct AddressRange {
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;
};

/// @brief The addresses one memory instruction actually touched.
///
/// @details Real addresses, taken after address calculation. This is the single
/// biggest advantage of modelling timing inside a functional simulator rather
/// than over a static instruction list: coalescing, cache behaviour and bank
/// conflicts all depend on values the kernel computed, and no static analysis
/// recovers them.
struct MemoryAccess {
  MemorySpace space = MemorySpace::None;
  bool is_load = true;
  /// @brief Bytes each active lane transfers.
  std::uint32_t bytes_per_lane = 4;
  /// @brief Byte address per active lane, in lane order.
  ///
  /// @details Empty for a scalar access, which uses scalar_address, and for a
  /// tensor transfer, which uses ranges. Also empty when the observer could not
  /// recover addresses; see `addresses_known`.
  std::vector<std::uint64_t> lane_addresses;
  std::uint64_t scalar_address = 0;
  std::uint32_t scalar_bytes = 4;
  /// @brief Global byte runs a tensor transfer touched.
  std::vector<AddressRange> ranges;
  /// @brief Elements a tensor transfer moved, its analogue of a lane access.
  std::uint64_t elements = 0;
  /// @brief Bytes a tensor transfer moved through the LDS, a separate bandwidth
  ///        from the global side that need not match it.
  std::uint64_t lds_bytes = 0;
  /// @brief Whether the access bypasses the first-level cache.
  bool non_temporal = false;

  /// @brief Whether the address fields describe the whole access.
  ///
  /// @details False when the observer issued the access but could not recover
  /// where it went. A model must then charge it as an uncoalesced miss to the
  /// farthest level it models rather than skipping it: an access with no
  /// addresses is exactly where guessing cheap is most tempting and most wrong,
  /// because it costs nothing and silently deletes real traffic.
  ///
  /// This is about recoverability, not about what the model asked for. A model
  /// that did not request Interest::lane_addresses gets an empty address list
  /// and this still true, because the byte counts are exact either way — it
  /// declined detail it does not use, which is not the same as the observer
  /// having lost track of the access.
  bool addresses_known = true;

  /// @brief The counter the wavefront will wait on for this operation.
  ///
  /// @details Reported by the simulator rather than derived from the class,
  /// because which counter an operation lands on is a per-target ISA decision.
  /// Vector stores are the case that matters: older compute targets have no
  /// separate store counter and post stores to the load counter, so a model
  /// that assumed a store counter would park those completions in a queue no
  /// wait instruction on that target can name, and the wait would cost nothing.
  ///
  /// WaitCounter::Count means the observer had none to report, and the model
  /// falls back to deriving one from the class.
  WaitCounter wait_counter = WaitCounter::Count;

  bool valid() const { return space != MemorySpace::None; }
};

/// @brief Thresholds an s_waitcnt-family instruction waits down to.
///
/// @details One entry per counter, holding the count the wavefront is willing
/// to leave outstanding. kUnconstrained is the common case, since most waits
/// name a single counter.
struct WaitThresholds {
  /// @brief High enough that the counter is never the constraint.
  static constexpr std::uint32_t kUnconstrained = 0xFFFFFFFFu;

  std::array<std::uint32_t, kNumWaitCounters> values{};

  WaitThresholds() { values.fill(kUnconstrained); }

  void set(WaitCounter counter, std::uint32_t threshold) {
    values[static_cast<std::size_t>(counter)] = threshold;
  }
  std::uint32_t get(WaitCounter counter) const { return values[static_cast<std::size_t>(counter)]; }
  bool constrains_anything() const {
    for (std::uint32_t v : values)
      if (v != kUnconstrained)
        return true;
    return false;
  }
};

/// @brief Properties of one instruction that do not change between executions.
///
/// @details Derived once per program counter and shared by every execution of
/// it. A kernel executes the same instruction many times, and re-deriving its
/// operand list on every issue would dominate the observer's cost. Only what
/// varies per execution — addresses, branch outcome, active lanes — travels in
/// the event.
///
/// Owned by the observer's per-kernel cache, which outlives every event that
/// points at it. Several wavefronts read one entry concurrently, so a model
/// must treat it as immutable.
struct StaticInstInfo {
  InstClass inst_class = InstClass::Unknown;
  std::uint32_t size_bytes = 4;
  std::vector<RegisterRange> reads;
  std::vector<RegisterRange> writes;
  /// @brief Raw immediate of an s_delay_alu / s_wait_alu, which a model
  ///        interprets as a scheduling hint. Meaningful for InstClass::DelayAlu.
  std::uint32_t delay_immediate = 0;
  /// @brief Mnemonic, for a model's own hot-spot reporting.
  std::string mnemonic;
};

/// @brief One observed instruction execution on one wavefront.
struct InstructionEvent {
  /// @brief Program counter of the instruction, before it executed.
  std::uint64_t pc = 0;
  /// @brief Static properties. Never null when the event reaches a model.
  const StaticInstInfo *info = nullptr;
  /// @brief The class to cost *this* execution as.
  ///
  /// @details Normally a copy of info->inst_class, but a FLAT access can only
  /// be told apart from an LDS access by the addresses it produced, so the
  /// observer may sharpen the class per execution. It lives on the event rather
  /// than being written back into the shared static entry, which other
  /// wavefronts are reading concurrently.
  InstClass effective_class = InstClass::Unknown;
  /// @brief Lanes that were active, for costing per-lane work.
  std::uint32_t active_lanes = 0;
  /// @brief Wavefront width, so a wave wider than the SIMD costs extra passes.
  std::uint32_t wave_lanes = 64;
  /// @brief Whether control flow left the fall-through path.
  bool branch_taken = false;
  /// @brief Meaningful when the class is WaitCounter.
  WaitThresholds wait;
  /// @brief Meaningful when the class is a memory class.
  MemoryAccess memory;
};

/// @brief Identifies a dispatch across the whole device.
///
/// @details Both halves are needed. Dispatch ids are allocated per command
/// processor, and a multi-XCD part has one command processor per XCD, so ids
/// collide across them. Keyed on dispatch_id alone, two unrelated kernels land
/// in the same report entry and their wavefronts are mixed together.
struct DispatchKey {
  std::uint32_t dispatch_id = 0;
  std::uint32_t queue_id = 0;

  friend bool operator==(const DispatchKey &, const DispatchKey &) = default;
};

/// @brief Identity and shape of a wavefront, fixed for its lifetime.
struct WaveRef {
  DispatchKey dispatch;
  std::uint32_t workgroup_id = 0;
  std::uint32_t wave_slot = 0;
  /// @brief Compute unit the wavefront is resident on. A model shards its
  ///        per-CU state on this; the host serializes calls that share it.
  std::uint32_t compute_unit_id = 0;
  std::uint32_t wave_lanes = 64;
  std::uint32_t vector_registers = 256;
  std::uint32_t scalar_registers = 102;

  friend bool operator==(const WaveRef &, const WaveRef &) = default;
};

/// @brief Shape of a kernel dispatch, delivered before its wavefronts start.
struct DispatchInfo {
  DispatchKey key;
  std::string kernel_name;
  std::uint32_t grid_size[3] = {0, 0, 0};
  std::uint32_t workgroup_size[3] = {0, 0, 0};
  std::uint32_t workgroup_count = 0;
  std::uint32_t waves_per_workgroup = 0;
  std::uint32_t vector_registers_per_wave = 0;
  std::uint32_t scalar_registers_per_wave = 0;
  /// @brief Group segment reserved per workgroup, in bytes, as allocated rather
  ///        than as requested: LDS is handed out in granules, and occupancy is
  ///        decided by what was taken, not by what was asked for.
  std::uint32_t lds_bytes_per_workgroup = 0;
  std::uint32_t wave_size = 0;
};

} // namespace rocjitsu::timing
