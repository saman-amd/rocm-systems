// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_PLUGINS_FIDELITY_CORE_SEMANTIC_FIDELITY_H_
#define ROCJITSU_VM_PLUGINS_FIDELITY_CORE_SEMANTIC_FIDELITY_H_

/// @file Per-opcode semantic fidelity classification.
///
/// The emulator already decides, per opcode, how faithfully it reproduces
/// hardware semantics; that verdict is made where the execute body is
/// generated and is otherwise discarded. This header names the verdict so it
/// can travel with a run and reach a result.
///
/// The classification mirrors the three ways an execute body comes into
/// existence:
///
/// * `kExact` — the body computes the architected result bit-for-bit.
/// * `kApproximate` — the body substitutes a numeric approximation
///   (isa/arch/amdgpu/shared/transcendental.h, which targets the ISA manual's
///   ULP bounds, or shared/pseudo_scalar.h, which defers to host libm), or
///   elides a semantic the generator could not model. The instruction retires,
///   but its result is not guaranteed to equal hardware.
/// * `kUnsupported` — no semantics exist at all; the generated body throws
///   util::UnimplementedInst.

#include "rocjitsu/isa/instruction.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rocjitsu::plugins::fidelity {

/// How faithfully an executed instruction reproduces hardware semantics.
enum class Fidelity : uint8_t { kExact = 0, kApproximate = 1, kUnsupported = 2 };

inline constexpr size_t kNumFidelityClasses = 3;

/// Stable JSON spelling for a fidelity class.
constexpr std::string_view fidelity_name(Fidelity fidelity) {
  switch (fidelity) {
  case Fidelity::kExact:
    return "exact";
  case Fidelity::kApproximate:
    return "approximate";
  case Fidelity::kUnsupported:
    return "unsupported";
  }
  return "exact";
}

/// Why an instruction is not emulated exactly.
///
/// Distinguishes a result that is numerically close from a semantic that was
/// not modelled at all: a consumer that tolerates ULP error may still care
/// very much that a barrier did nothing.
enum class Inexactness : uint8_t {
  kNone = 0,
  /// Result computed by an approximation of the hardware's numeric behaviour.
  kNumericApproximation = 1,
  /// The opcode's architected side effect is not modelled and was skipped.
  kElidedSemantics = 2,
};

constexpr std::string_view inexactness_name(Inexactness reason) {
  switch (reason) {
  case Inexactness::kNone:
    return "none";
  case Inexactness::kNumericApproximation:
    return "numeric_approximation";
  case Inexactness::kElidedSemantics:
    return "elided_semantics";
  }
  return "none";
}

/// The consumer of a non-exact value that determines whether the inexactness
/// stayed numeric or changed what the program did.
///
/// A non-exact value that is only ever stored is a numeric accuracy question.
/// The same value steering a branch, forming an address, or gating
/// synchronization can invalidate the run's structure, which is a different
/// and far stronger claim.
enum class TaintSink : uint8_t {
  kControlFlow = 0,
  kAddressing = 1,
  kSynchronization = 2,
};

inline constexpr size_t kNumTaintSinks = 3;

constexpr std::string_view taint_sink_name(TaintSink sink) {
  switch (sink) {
  case TaintSink::kControlFlow:
    return "control_flow";
  case TaintSink::kAddressing:
    return "addressing";
  case TaintSink::kSynchronization:
    return "synchronization";
  }
  return "control_flow";
}

/// Classify an instruction's emulation fidelity from its mnemonic.
///
/// Keyed on the mnemonic because that is what identifies an execute body
/// across every arch that shares it. Encoding suffixes (_e32/_e64/_dpp/_sdwa)
/// select an operand form, not a different numeric implementation, so they are
/// stripped before matching.
Fidelity classify(std::string_view mnemonic);

/// Why `classify` returned a non-exact verdict; kNone when it returned kExact.
Inexactness inexactness_of(std::string_view mnemonic);

/// The taint sinks an instruction would expose a consumed value to.
///
/// Derived from InstFlags rather than a mnemonic list so it stays correct for
/// every arch: BRANCH/COND_BRANCH/INDIRECT_BRANCH/INDIRECT_CALL are control
/// flow, MEMORY_OP is addressing, and WAITCNT/BARRIER are synchronization.
uint32_t taint_sink_mask(const Instruction &inst);

/// Bit position of `sink` within the mask returned by `taint_sink_mask`.
constexpr uint32_t taint_sink_bit(TaintSink sink) { return 1u << static_cast<uint32_t>(sink); }

} // namespace rocjitsu::plugins::fidelity

#endif // ROCJITSU_VM_PLUGINS_FIDELITY_CORE_SEMANTIC_FIDELITY_H_
