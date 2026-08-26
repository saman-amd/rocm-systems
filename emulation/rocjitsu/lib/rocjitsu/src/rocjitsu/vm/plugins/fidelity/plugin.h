// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_PLUGINS_FIDELITY_PLUGIN_H_
#define ROCJITSU_VM_PLUGINS_FIDELITY_PLUGIN_H_

/// @file Semantic-fidelity reporting plugin.
///
/// Reports how trustworthy a run's results are, as distinct from whether the
/// run completed. A kernel that executes to completion under the emulator is
/// structurally valid; that says nothing about whether the values it produced
/// match hardware. This plugin supplies the missing axis: it classifies every
/// retired instruction as exact, approximate or unsupported, and reports
/// whether a non-exact value went on to steer control flow, form an address,
/// or gate synchronization.
///
/// The distinction matters for CI. A numeric approximation that only reaches a
/// stored result bounds the error, and an output tolerance can absorb it. The
/// same value reaching control flow or addressing means the emulated execution
/// may have taken a path hardware would not, which no output tolerance can
/// detect.

#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/fidelity/core/semantic_fidelity.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace rocjitsu::plugins::fidelity {

using FidelityCounts = std::array<uint64_t, kNumFidelityClasses>;
using TaintCounts = std::array<uint64_t, kNumTaintSinks>;

/// Instruction counts and taint observations accumulated over one scope.
struct FidelityTotals {
  FidelityCounts instructions{};
  /// Instructions that consumed a non-exact value, per sink they exposed it to.
  TaintCounts tainted_sinks{};
  /// Distinct non-exact mnemonics retired, and how many times each retired.
  std::map<std::string, uint64_t> inexact_mnemonics;

  void merge(const FidelityTotals &other);
  uint64_t total_instructions() const;
  bool any_tainted_sink() const;
};

/// Per-wavefront fidelity accounting and taint state.
///
/// Taint is tracked per wavefront because registers are: two waves running the
/// same code reach different conclusions about their own values.
struct FidelityWavefrontState final : WavefrontState {
  FidelityTotals totals;
  std::unordered_set<uint32_t> tainted_vgprs;
  std::unordered_set<uint32_t> tainted_sgprs;
  /// Sinks the in-flight instruction exposes a consumed value to, and whether
  /// it has already been charged, so one instruction counts once per sink.
  uint32_t active_sinks = 0;
  uint32_t charged_sinks = 0;
  /// Whether the in-flight instruction's result is itself non-exact, either
  /// because its own semantics are inexact or because it consumed taint.
  bool active_inexact = false;
};

/// Emits a per-dispatch and end-of-run semantic-fidelity report as JSONL.
class FidelityPlugin final : public ExecutionPlugin {
public:
  explicit FidelityPlugin(const char *config_json = nullptr);
  ~FidelityPlugin() override;

  /// Taint state is per wavefront, but a wavefront's register hooks must not
  /// interleave with another partition's while an instruction is in flight.
  bool requires_serial_hot_hooks() const override { return true; }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override;
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;
  void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override;
  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override;
  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                              uint64_t lane_mask, uint8_t byte_mask) override;
  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;
  void onShutdown() override;

private:
  struct DispatchState {
    KernelDispatchInfo info{};
    FidelityTotals totals;
  };

  /// Charge the in-flight instruction for consuming a non-exact value.
  void observe_tainted_read(FidelityWavefrontState &state);

  void emit_record(std::string_view record, const KernelDispatchInfo *info,
                   const FidelityTotals &totals);

  std::unordered_map<uint32_t, DispatchState> dispatches_;
  FidelityTotals aggregate_;
  uint64_t completed_dispatches_ = 0;
  bool summary_emitted_ = false;
};

} // namespace rocjitsu::plugins::fidelity

#endif // ROCJITSU_VM_PLUGINS_FIDELITY_PLUGIN_H_
