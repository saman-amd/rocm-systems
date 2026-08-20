// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file observer.h
/// @brief The one place rocjitsu execution becomes a timing event stream.
///
/// @details There is exactly one observation layer, and this is it. Turning
/// hooks into a coherent per-wavefront event stream is the fiddly part of the
/// feature — sticky wait targets, a hook that never fires for a terminating
/// wavefront, a program counter that means something different in the after
/// hook than a reader expects, dispatch ids that collide across command
/// processors — and every one of those is a silent under-count if it is got
/// wrong. Re-deriving it per model would mean re-deriving those bugs, so
/// models see events (event.h) and never a rocjitsu type.
///
/// The observer is a plugin only in the mechanical sense that it consumes
/// ExecutionPlugin hooks. It is not selected through the `plugins` block and
/// carries no configuration of its own; the `timing` block names the model and
/// the loader builds both.

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/plugins/wavefront_state.h"
#include "rocjitsu/vm/timing/event.h"
#include "rocjitsu/vm/timing/time_source.h"
#include "rocjitsu/vm/timing/timing_model.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu::timing {

class TimingHost;

/// @brief Presents a TimingModel to SimulatedClock as a bare clock.
///
/// @details A separate object rather than making TimingObserver itself a
/// TimeSource: the clock is polled from guest timestamp paths — a guest thread
/// inside an ioctl, the completion tracker writing a signal, an in-kernel
/// s_memtime — on threads that have nothing to do with execution and that may
/// run concurrently with any hook. Narrowing the type to these two values is
/// how that boundary is stated in the type system rather than in a comment.
///
/// Neither call clamps or locks. Monotonicity is TimingModel::device_cycles()'s
/// documented obligation and SimulatedClock clamps again on the way out, so a
/// third clamp here would only hide a model that is violating its contract.
class ModelTimeSource final : public TimeSource {
public:
  explicit ModelTimeSource(const TimingModel &model) : model_(model) {}

  std::uint64_t current_cycles() const override { return model_.device_cycles(); }
  double clock_ghz() const override { return model_.clock_ghz(); }

private:
  const TimingModel &model_;
};

/// @brief Everything the observer keeps on a wavefront.
///
/// @details Lives in the wavefront's plugin slot so the hot path never looks a
/// wavefront up in a map. The pending fields carry an instruction from the
/// before hook, which knows the program counter it issued at and the lanes that
/// were live going in, to the after hook, which knows where control flow
/// actually went and what addresses were produced.
struct ObservedWave final : rocjitsu::WavefrontState {
  WaveRef ref;
  /// @brief Program counter the pending instruction issued at.
  std::uint64_t pending_pc = 0;
  /// @brief Static properties of the pending instruction, owned by the
  ///        observer's per-pc cache, which outlives every wavefront.
  const StaticInstInfo *pending_info = nullptr;
  /// @brief Label to charge to the unmodelled ledger for the pending
  ///        instruction, or empty. Held as a pointer into the cache entry so
  ///        the hot path allocates nothing to declare a coverage gap.
  const std::string *pending_gap = nullptr;
  /// @brief Lanes live *before* the instruction executed.
  ///
  /// @details Sampled in the before hook because an instruction may rewrite
  /// EXEC — v_cmpx and the saveexec family do — and the work the machine
  /// actually performed is the work of the lanes that were live going in.
  /// Sampling after would cost a diverging branch nothing on the iteration
  /// that narrowed the mask.
  std::uint32_t pending_active_lanes = 0;
  bool has_pending = false;
};

/// @brief Turns rocjitsu execution hooks into the event stream a model eats.
///
/// @details Borrows the model and the host; the loader owns both and must
/// destroy this object before either. That ordering is load-bearing rather than
/// tidy: the destructor calls TimingModel::on_finalize(), because a local run
/// under the interposer never tears the VM down and onShutdown() may therefore
/// never fire at all.
class TimingObserver final : public rocjitsu::ExecutionPlugin {
public:
  /// @param model The model to drive. Borrowed; must outlive this observer.
  /// @param host Tuning and the coverage ledger. Borrowed; outlives the model.
  TimingObserver(TimingModel &model, const TimingHost &host);
  ~TimingObserver() override;

  TimingObserver(const TimingObserver &) = delete;
  TimingObserver &operator=(const TimingObserver &) = delete;

  /// @brief The clock to hand SimulatedClock::set_time_source().
  ///
  /// @details Offered rather than installed: making the model the guest's clock
  /// changes what the program under test measures, which is a much larger
  /// change than adding a report and belongs to whoever configured the run.
  TimeSource *time_source() { return &time_source_; }

  /// @brief Run the model's terminal callback and emit both reports, at most
  ///        once however many teardown paths reach it.
  ///
  /// @details Public because the paths that must reach it are not all inside
  /// this class: onShutdown() and the destructor both call it, and so does an
  /// exit handler, because a local run under the interposer reaches neither.
  void finalize_once();

  /// @brief Whether the group must serialize the hot hooks for this plugin.
  ///
  /// @details No. Every piece of state the hot hooks share is protected inside
  /// the observer — the per-pc cache by a shared mutex, the model by the
  /// per-compute-unit shard locks below, the pending instruction by living on
  /// the wavefront that is executing it. Returning true would put every
  /// wavefront in the device behind one mutex, which is exactly the bottleneck
  /// timing_model.h's threading contract exists to avoid: the contract promises
  /// a model serialization per compute unit *and nothing more*, precisely so
  /// that units can be costed in parallel.
  bool requires_serial_hot_hooks() const override { return false; }

  void onShutdown() override;

  void onAmdgpuDispatchPacketProcessed(const rocjitsu::KernelDispatchInfo &info) override;
  void onAmdgpuDispatchExecutionEnd(std::uint32_t dispatch_id) override;

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;
  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) override;

  void onAmdgpuBeforeExecuteInstruction(std::uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;
  void onAmdgpuAfterExecuteInstruction(std::uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override;

private:
  /// @brief One per-pc cache entry.
  ///
  /// @details The coverage label sits beside the static info rather than in it
  /// because StaticInstInfo is the model-facing contract and this is the
  /// observer's own bookkeeping. Deriving it once per program counter is what
  /// lets an unclassified opcode be charged to the ledger on *every* execution
  /// — which is the number that says how much of the run is uncovered —
  /// without formatting a string on the hot path.
  struct CacheEntry {
    StaticInstInfo info;
    std::string gap_label;
  };

  /// @brief Static properties for @p pc, derived on first sight.
  ///
  /// @details Cached because a kernel executes each instruction many times and
  /// walking its operand list every time would dominate the observer's cost.
  /// The cached mnemonic is checked against the instruction's: binary
  /// translation can place different code at an address seen earlier in the
  /// run, and a stale entry would silently cost the new code as the old.
  const CacheEntry &static_info(std::uint64_t pc, const Instruction &inst);

  /// @brief The compute unit this wavefront is reported as contending on.
  ///
  /// @details The one it ran on, unless `timing.machine.compute_units` says how
  /// many the part gives a dispatch. It usually does, because the emulator
  /// confines a dispatch to the compute units of a single accelerator die while
  /// the device it presents advertises every die's worth. Reporting the
  /// simulator's placement unchanged therefore crowds a whole grid onto one
  /// die's units and over-states contention; declaring the real count spreads
  /// the workgroups round-robin the way a shader-processor input does.
  ///
  /// Nothing else moves — the same wavefronts execute the same instructions
  /// against the same addresses. This decides only which of them queue behind
  /// each other, and which shard lock serializes them.
  std::uint32_t placed_compute_unit(const amdgpu::Wavefront &wf) const;

  ObservedWave *wave_state(const amdgpu::Wavefront &wf) const {
    return static_cast<ObservedWave *>(wf.plugin_state(slot_index()));
  }

  /// @brief Announce @p key to the model if it has not been announced yet.
  ///
  /// @details Dispatch packets are parsed ahead of execution, so the normal
  /// order is announcement first and wavefronts later. A wavefront that appears
  /// for a dispatch nobody announced — the command processor reached it by a
  /// path that does not fire the packet hook — would otherwise be attributed to
  /// a dispatch the model has never heard of, and the report's first entries
  /// are exactly the kernels a run is usually about. Synthesizing a shaped-only
  /// announcement keeps it attributable, named `?` rather than nameless.
  void ensure_dispatch_announced(const DispatchKey &key, const amdgpu::Wavefront &wf);

  /// @brief Complete and emit the wavefront's pending instruction.
  ///
  /// @param branch_taken Whether control flow left the fall-through path.
  /// @param inst The executing instruction, or null when it is no longer
  ///        reachable — the terminal path, where the memory and wait payloads
  ///        are moot because the instruction that halted the wave is not one.
  void emit_pending(ObservedWave &state, const amdgpu::Wavefront &wf, bool branch_taken,
                    const Instruction *inst);

  /// @brief The lock covering @p compute_unit_id.
  ///
  /// @details timing_model.h promises a model that calls naming one compute
  /// unit are serialized and that calls naming different ones are not. Sharding
  /// on the id is how: the shard count is fixed and small, so two units can
  /// collide onto one lock — which costs concurrency and never correctness,
  /// while the reverse would corrupt the model's per-unit state.
  std::mutex &shard_for(std::uint32_t compute_unit_id) {
    return shards_[compute_unit_id & (kNumShards - 1)];
  }

  /// @brief Pack a dispatch key into one integer, for the announced set.
  static std::uint64_t packed(const DispatchKey &key) {
    return (static_cast<std::uint64_t>(key.dispatch_id) << 32) | key.queue_id;
  }

  TimingModel &model_;
  const TimingHost &host_;
  /// @brief Sampled once, as TimingModel::Interest requires. The payloads it
  ///        gates — 64 lane addresses per access, a walk of the operand list —
  ///        are the observer's dominant per-instruction cost.
  TimingModel::Interest interest_;
  ModelTimeSource time_source_;

  /// @brief Compute units the config says a dispatch is spread over, or 0 to
  ///        report the simulator's own placement.
  ///
  /// @details Zero is the pessimistic default deliberately. A config that does
  /// not declare the count leaves the grid crammed onto the die's worth of
  /// units the emulator actually uses, which over-states contention and reads
  /// slow; inventing a larger number would read fast on no evidence.
  std::uint32_t declared_compute_units_ = 0;

  /// @brief Number of shard locks. A power of two so the map is a mask.
  static constexpr std::size_t kNumShards = 64;
  std::array<std::mutex, kNumShards> shards_;

  /// @brief Read on every instruction, written once per distinct program
  ///        counter, so concurrent wavefronts read it in parallel and only a
  ///        first sighting takes the exclusive lock.
  mutable std::shared_mutex info_mutex_;
  std::unordered_map<std::uint64_t, std::unique_ptr<CacheEntry>> info_cache_;
  /// @brief Entries displaced from info_cache_, kept alive for the run.
  ///
  /// @details An entry is only ever displaced when translated code reuses an
  /// address, and a wavefront or a model can be holding the old one at that
  /// moment. Nothing tracks those references, so the only safe lifetime is the
  /// observer's own.
  std::vector<std::unique_ptr<CacheEntry>> retired_info_;

  /// @brief Guards the dispatch bookkeeping below.
  ///
  /// @details Its writers are the infrequent hooks, which the plugin group
  /// already serializes against each other. The lock is still taken, because
  /// depending on a policy that belongs to another class — and that a later
  /// `requires_serial_hot_hooks()` change would silently alter — is how a data
  /// race gets introduced by an edit that looks unrelated.
  std::mutex dispatch_mutex_;
  /// @brief dispatch_id to queue_id, because onAmdgpuDispatchExecutionEnd
  ///        carries only the id and a DispatchKey needs both halves.
  std::unordered_map<std::uint32_t, std::uint32_t> dispatch_queue_;
  /// @brief Keys already delivered to on_dispatch_begin.
  std::unordered_set<std::uint64_t> announced_;

  /// @brief Whether on_finalize() has run. The destructor and onShutdown() both
  ///        reach for it and exactly one may win.
  std::atomic<bool> finalized_{false};
};

} // namespace rocjitsu::timing
