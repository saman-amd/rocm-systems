// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/fidelity/core/semantic_fidelity.h"

#include <algorithm>
#include <array>

namespace rocjitsu::plugins::fidelity {
namespace {

/// Mnemonics whose generated execute body produces its result through
/// isa/arch/amdgpu/shared/transcendental.h (ISA-manual ULP bounds) or
/// shared/pseudo_scalar.h (host libm) rather than reproducing the hardware's
/// exact rounding.
constexpr std::array kNumericApproximations = {
    std::string_view{"v_cos_f16"},   std::string_view{"v_cos_f32"},
    std::string_view{"v_exp_f16"},   std::string_view{"v_exp_f32"},
    std::string_view{"v_log_f16"},   std::string_view{"v_log_f32"},
    std::string_view{"v_rcp_f16"},   std::string_view{"v_rcp_f32"},
    std::string_view{"v_rcp_f64"},   std::string_view{"v_rcp_iflag_f32"},
    std::string_view{"v_rsq_f16"},   std::string_view{"v_rsq_f32"},
    std::string_view{"v_rsq_f64"},   std::string_view{"v_s_exp_f32"},
    std::string_view{"v_s_log_f32"}, std::string_view{"v_s_rcp_f32"},
    std::string_view{"v_s_rsq_f32"}, std::string_view{"v_s_sqrt_f32"},
    std::string_view{"v_sin_f16"},   std::string_view{"v_sin_f32"},
    std::string_view{"v_sqrt_f16"},  std::string_view{"v_sqrt_f32"},
    std::string_view{"v_sqrt_f64"},
};

/// Mnemonics that retire with an empty execute body even though hardware
/// performs an architecturally observable action.
///
/// Deliberately excludes opcodes that are no-ops on hardware too (s_nop,
/// v_nop, ds_nop, s_clause, s_delay_alu, prefetch and perf-counter hints):
/// eliding those loses nothing, so reporting them would only add noise.
/// What remains either moves data, changes numeric state, or gates execution.
constexpr std::array kElidedSemantics = {
    // Produce a result that is never written, so consumers read stale values.
    std::string_view{"image_bvh_intersect_ray"},
    std::string_view{"lds_direct_load"},
    std::string_view{"lds_param_load"},
    // Change FP behaviour for every later instruction in the wave.
    std::string_view{"s_denorm_mode"},
    std::string_view{"s_round_mode"},
    std::string_view{"v_clrexcp"},
    // Control flow that silently falls through.
    std::string_view{"s_cbranch_cdbgsys"},
    std::string_view{"s_cbranch_cdbgsys_and_user"},
    std::string_view{"s_cbranch_cdbgsys_or_user"},
    std::string_view{"s_cbranch_cdbguser"},
    std::string_view{"s_cbranch_join"},
    std::string_view{"s_trap"},
    // Execution state and synchronization that never takes effect.
    std::string_view{"s_alloc_vgpr"},
    std::string_view{"s_sethalt"},
    std::string_view{"s_setkill"},
    std::string_view{"s_setprio"},
    std::string_view{"s_set_valu_coexec_mode"},
    std::string_view{"s_wait_alu"},
    std::string_view{"s_wait_event"},
    std::string_view{"s_wait_idle"},
    std::string_view{"s_waitcnt_depctr"},
    std::string_view{"s_wakeup"},
};

/// Strip encoding suffixes that select an operand form rather than a different
/// execute body.
std::string_view canonical(std::string_view mnemonic) {
  constexpr std::array kSuffixes = {
      std::string_view{"_e32"},  std::string_view{"_e64"},   std::string_view{"_sdwa"},
      std::string_view{"_dpp8"}, std::string_view{"_dpp16"}, std::string_view{"_dpp"},
  };
  for (bool stripped = true; stripped;) {
    stripped = false;
    for (std::string_view suffix : kSuffixes) {
      if (mnemonic.size() > suffix.size() && mnemonic.ends_with(suffix)) {
        mnemonic.remove_suffix(suffix.size());
        stripped = true;
        break;
      }
    }
  }
  return mnemonic;
}

bool contains(const auto &table, std::string_view mnemonic) {
  return std::find(table.begin(), table.end(), mnemonic) != table.end();
}

} // namespace

Inexactness inexactness_of(std::string_view mnemonic) {
  const std::string_view name = canonical(mnemonic);
  if (contains(kNumericApproximations, name))
    return Inexactness::kNumericApproximation;
  if (contains(kElidedSemantics, name))
    return Inexactness::kElidedSemantics;
  return Inexactness::kNone;
}

Fidelity classify(std::string_view mnemonic) {
  return inexactness_of(mnemonic) == Inexactness::kNone ? Fidelity::kExact : Fidelity::kApproximate;
}

uint32_t taint_sink_mask(const Instruction &inst) {
  const uint64_t flags = inst.flags();
  uint32_t mask = 0;
  if (flags & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL))
    mask |= taint_sink_bit(TaintSink::kControlFlow);
  if (flags & MEMORY_OP)
    mask |= taint_sink_bit(TaintSink::kAddressing);
  if (flags & (WAITCNT | BARRIER))
    mask |= taint_sink_bit(TaintSink::kSynchronization);
  return mask;
}

} // namespace rocjitsu::plugins::fidelity
