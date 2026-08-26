// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/fidelity/plugin.h"

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <format>
#include <string>

namespace rocjitsu::plugins::fidelity {
namespace {

constexpr size_t index_of(Fidelity fidelity) { return static_cast<size_t>(fidelity); }

std::string json_escape(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char character : text) {
    if (character == '"' || character == '\\')
      escaped += '\\';
    escaped += character;
  }
  return escaped;
}

} // namespace

void FidelityTotals::merge(const FidelityTotals &other) {
  for (size_t i = 0; i < instructions.size(); ++i)
    instructions[i] += other.instructions[i];
  for (size_t i = 0; i < tainted_sinks.size(); ++i)
    tainted_sinks[i] += other.tainted_sinks[i];
  for (const auto &[mnemonic, count] : other.inexact_mnemonics)
    inexact_mnemonics[mnemonic] += count;
}

uint64_t FidelityTotals::total_instructions() const {
  uint64_t total = 0;
  for (const uint64_t count : instructions)
    total += count;
  return total;
}

bool FidelityTotals::any_tainted_sink() const {
  for (const uint64_t count : tainted_sinks)
    if (count != 0)
      return true;
  return false;
}

FidelityPlugin::FidelityPlugin(const char * /*config_json*/) : ExecutionPlugin("fidelity") {}

FidelityPlugin::~FidelityPlugin() { onShutdown(); }

void FidelityPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  dispatches_[info.dispatch_id].info = info;
}

void FidelityPlugin::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
  wf.set_plugin_state(slot_index(), std::make_unique<FidelityWavefrontState>());
}

void FidelityPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                      amdgpu::Wavefront &wf) {
  auto *state = static_cast<FidelityWavefrontState *>(wf.plugin_state(slot_index()));
  const std::string_view mnemonic = inst.mnemonic();
  const Fidelity fidelity = classify(mnemonic);

  ++state->totals.instructions[index_of(fidelity)];
  if (fidelity != Fidelity::kExact)
    ++state->totals.inexact_mnemonics[std::string(mnemonic)];

  // Register hooks fire between here and the after-execute hook, so stage what
  // they need: which sinks this instruction would expose a consumed value to,
  // and whether its result is non-exact regardless of what it consumes.
  state->active_sinks = taint_sink_mask(inst);
  state->charged_sinks = 0;
  state->active_inexact = fidelity != Fidelity::kExact;
}

void FidelityPlugin::observe_tainted_read(FidelityWavefrontState &state) {
  // Consuming a non-exact value makes this instruction's own result non-exact,
  // which is what propagates taint to its destinations.
  state.active_inexact = true;
  const uint32_t uncharged = state.active_sinks & ~state.charged_sinks;
  if (uncharged == 0)
    return;
  for (size_t sink = 0; sink < kNumTaintSinks; ++sink) {
    if (uncharged & taint_sink_bit(static_cast<TaintSink>(sink)))
      ++state.totals.tainted_sinks[sink];
  }
  state.charged_sinks |= uncharged;
}

void FidelityPlugin::onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                           uint64_t /*lane_mask*/, uint8_t /*byte_mask*/) {
  if (wf == nullptr)
    return;
  auto *state = static_cast<FidelityWavefrontState *>(wf->plugin_state(slot_index()));
  if (state == nullptr || !state->tainted_vgprs.contains(physical_reg))
    return;
  observe_tainted_read(*state);
}

void FidelityPlugin::onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) {
  if (wf == nullptr)
    return;
  auto *state = static_cast<FidelityWavefrontState *>(wf->plugin_state(slot_index()));
  if (state == nullptr || !state->tainted_sgprs.contains(physical_reg))
    return;
  observe_tainted_read(*state);
}

void FidelityPlugin::onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                            uint64_t /*lane_mask*/, uint8_t /*byte_mask*/) {
  if (wf == nullptr)
    return;
  auto *state = static_cast<FidelityWavefrontState *>(wf->plugin_state(slot_index()));
  if (state == nullptr)
    return;
  // An exact instruction reading only exact inputs restores confidence in the
  // register, so clear rather than leave stale taint behind.
  if (state->active_inexact)
    state->tainted_vgprs.insert(physical_reg);
  else
    state->tainted_vgprs.erase(physical_reg);
}

void FidelityPlugin::onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                                     amdgpu::Wavefront &wf) {
  auto *state = static_cast<FidelityWavefrontState *>(wf.plugin_state(slot_index()));
  state->active_sinks = 0;
  state->charged_sinks = 0;
  state->active_inexact = false;
}

void FidelityPlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  auto *state = static_cast<FidelityWavefrontState *>(wf.plugin_state(slot_index()));
  dispatches_[wf.dispatch_id()].totals.merge(state->totals);
}

void FidelityPlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  auto iter = dispatches_.find(dispatch_id);
  if (iter == dispatches_.end())
    return;

  DispatchState &state = iter->second;
  emit_record("dispatch", &state.info, state.totals);
  aggregate_.merge(state.totals);
  ++completed_dispatches_;
  dispatches_.erase(iter);
}

void FidelityPlugin::onShutdown() {
  if (summary_emitted_)
    return;
  summary_emitted_ = true;
  emit_record("summary", nullptr, aggregate_);
}

void FidelityPlugin::emit_record(std::string_view record, const KernelDispatchInfo *info,
                                 const FidelityTotals &totals) {
  const uint64_t instruction_count = totals.total_instructions();
  std::string output =
      std::format("{{\"schema\":\"rocjitsu.fidelity.v1\",\"record\":\"{}\"", record);
  if (info) {
    output += std::format(",\"dispatch_id\":{},\"kernel_name\":\"{}\",\"kernel_symbol\":\"{}\"",
                          info->dispatch_id, json_escape(info->kernelNameOrUnknown()),
                          json_escape(info->kernelSymbolOrUnknown()));
  } else {
    output += std::format(",\"dispatches\":{}", completed_dispatches_);
  }

  // A run is numerically validated only if every instruction was exact and no
  // non-exact value reached a sink; report that verdict rather than making
  // each consumer re-derive it.
  const bool exact_only = totals.instructions[index_of(Fidelity::kExact)] == instruction_count;
  output += std::format(",\"wave_instructions\":{},\"numerically_validated\":{}", instruction_count,
                        exact_only && !totals.any_tainted_sink() ? "true" : "false");

  output += ",\"fidelity\":{";
  for (size_t i = 0; i < totals.instructions.size(); ++i) {
    if (i != 0)
      output += ',';
    output +=
        std::format("\"{}\":{}", fidelity_name(static_cast<Fidelity>(i)), totals.instructions[i]);
  }
  output += '}';

  output += ",\"tainted_sinks\":{";
  for (size_t i = 0; i < totals.tainted_sinks.size(); ++i) {
    if (i != 0)
      output += ',';
    output += std::format("\"{}\":{}", taint_sink_name(static_cast<TaintSink>(i)),
                          totals.tainted_sinks[i]);
  }
  output += '}';

  output += ",\"inexact_instructions\":{";
  bool first = true;
  for (const auto &[mnemonic, count] : totals.inexact_mnemonics) {
    if (!first)
      output += ',';
    first = false;
    output += std::format("\"{}\":{{\"count\":{},\"reason\":\"{}\"}}", json_escape(mnemonic), count,
                          inexactness_name(inexactness_of(mnemonic)));
  }
  output += "}}\n";
  sink().write(output);
}

} // namespace rocjitsu::plugins::fidelity
