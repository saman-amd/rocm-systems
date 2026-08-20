// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_host.h
/// @brief What rocjitsu offers a timing model: tuning, and a place to admit
///        what was not modelled.
///
/// @details A model is handed one of these at construction and may keep the
/// reference for its lifetime. It is an abstract interface rather than a
/// concrete class because a model lives in its own shared object and resolves
/// no simulator symbols; calling a virtual on a host-constructed object needs
/// only the declaration, since the vtable travels with the object.
///
/// Everything here exists to serve one rule: a timing model must fail slow.

#pragma once

#include "rocjitsu/vm/timing/inst_class.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace rocjitsu::timing {

/// @brief The pessimistic default for `machine.compute_units`.
///
/// @details Shared so that every reader of the key falls back to the same
/// number. The coverage report records the last value asked for, so two readers
/// disagreeing would let it name a value nothing actually used — and that
/// report is the whole basis for trusting what a run computed.
///
/// One compute unit: the whole grid crammed onto a single one. Absurd, and
/// deliberately so — a config that does not describe the part cannot be used to
/// claim a number about it.
inline constexpr std::uint64_t kPessimisticComputeUnits = 1;

class TimingHost {
public:
  virtual ~TimingHost() = default;

  // -- Tuning ---------------------------------------------------------------
  //
  // A model contains no numbers. Every latency, rate and capacity is read from
  // the `timing` block of the architecture config file, so retargeting a model
  // to another part is a config edit rather than a rebuild.

  /// @brief Read an integer from `timing.machine`, by dotted path.
  ///
  /// @param key Path below `timing.machine`, e.g. `"vector_alu.issue_cycles"`.
  /// @param pessimistic The value to use when the config does not name @p key.
  ///        This must be the *slowest reasonable* value for the parameter, not
  ///        the typical one. A forgotten parameter then makes the run read slow
  ///        rather than fast, which is the difference between a gap that is
  ///        noticed and a gap that is mistaken for accuracy.
  ///
  /// @details Every fallback is recorded and printed at shutdown, so a model
  /// asking for something the config never provided is visible in the run
  /// rather than only in the numbers.
  virtual std::uint64_t tune(std::string_view key, std::uint64_t pessimistic) const = 0;

  /// @brief As tune(), for a non-integral parameter such as a rate.
  virtual double tune_real(std::string_view key, double pessimistic) const = 0;

  /// @brief Cycles to charge one instruction of @p cls on its issue port.
  ///
  /// @details Reads `timing.machine.<class>.issue_cycles`. When the config does
  /// not name that class the result is the largest issue cost the config gives
  /// *any* class, which is what makes InstClass::Unknown expensive without the
  /// config having to anticipate it. An opcode nobody classified is therefore
  /// charged as much as the most expensive thing the part can do.
  virtual std::uint64_t class_issue_cycles(InstClass cls) const = 0;

  /// @brief The model's private configuration, from `timing.model_config`, as a
  ///        JSON object string. Never null; `"{}"` when absent.
  ///
  /// @details Kept separate from `timing.machine` so one model's invented knob
  /// never looks like a property of the hardware. The host does not interpret
  /// it.
  virtual std::string model_config_json() const = 0;

  /// @brief The shader clock, in GHz, from `timing.clock_mhz`.
  virtual double clock_ghz() const = 0;

  // -- Coverage -------------------------------------------------------------

  /// @brief Declare that an effect was observed but not modelled.
  ///
  /// @param effect A short stable label, e.g. `"lds bank conflicts"`.
  ///
  /// @details Counted per distinct label. Declare a *structural* gap — a whole
  /// subsystem the model does not have — once at construction, since it is true
  /// of every instruction and counting it per instruction says nothing extra.
  /// Declare a *data-dependent* gap — an access whose addresses were not
  /// recoverable, a case the model fell through on — every time it happens, so
  /// the count is a measure of how much of the run it covers.
  ///
  /// The host aggregates these and prints them at shutdown. A run
  /// whose ledger is non-empty is not a validated run, which is the point:
  /// silence has to mean coverage, or a report is not evidence. Charging an
  /// unmodelled effect zero and saying nothing is the failure this exists to
  /// prevent — it always reads fast, and on a benchmark dominated by something
  /// else it looks exactly like accuracy.
  ///
  /// Safe to call from any thread and cheap enough for a per-instruction path.
  virtual void note_unmodeled(std::string_view effect) const = 0;

  /// @brief Write a diagnostic line to the run's plugin sink.
  virtual void log(std::string_view message) const = 0;

  /// @brief Append the run's coverage record to @p out.
  ///
  /// @details Every parameter a model asked for and what it resolved to, which
  /// of those the config did not name, and the unmodelled ledger with counts.
  /// The host emits this next to the model's own report at the end of a run.
  ///
  /// Pure rather than defaulted on purpose: a host that quietly wrote nothing
  /// here would turn every report into a bare set of numbers with no statement
  /// of what produced them, which is the shape this API exists to prevent.
  virtual void write_coverage_report(std::string &out) const = 0;
};

} // namespace rocjitsu::timing
