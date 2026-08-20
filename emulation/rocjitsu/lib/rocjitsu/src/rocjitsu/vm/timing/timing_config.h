// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_config.h
/// @brief The `timing` block of the architecture config file, and the
///        TimingHost that serves it to a model.
///
/// @details The block is read in a second, schema-free pass over the same
/// config file — the same way the `plugins` and `sinks` blocks are read. It has
/// to be: the typed FlatBuffers load runs with unexpected fields skipped, so a
/// `timing` block added to the schema-typed path would be dropped silently and
/// the model would run entirely on fallbacks with nothing to say it had.
///
/// The namespace is kept disjoint from `plugins` so that an observing plugin
/// can never be selected as the timing authority, or the reverse.
///
/// @code
///   "timing": {
///     "model": "leaky",
///     "clock_mhz": 2100,
///     "machine": { "vector_alu": { "issue_cycles": 4 }, "compute_units": 256 },
///     "model_config": { }
///   }
/// @endcode

#pragma once

#include "rocjitsu/vm/timing/timing_host.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rocjitsu {
class PluginSink;
}

namespace rocjitsu::timing {

/// @brief A parsed `timing` block, and the TimingHost view of it.
///
/// @details Also the run's coverage record. Every parameter a model asks for is
/// noted with the value it got and whether the config named it, and every
/// effect a model declares unmodelled is counted. Both are printed at shutdown,
/// which is what makes the fail-slow rule auditable rather than aspirational:
/// a report that does not list its gaps is not evidence that there are none.
class TimingConfig final : public TimingHost {
public:
  /// @brief Parse the `timing` block out of a full rocjitsu config document.
  /// @returns Null when the document has no `timing` block, which is the normal
  ///          case for a run with no timing model.
  static std::unique_ptr<TimingConfig> parse(const std::string &config_json);

  ~TimingConfig() override;

  /// @brief The model to load, from `timing.model`. Never empty in a parsed
  ///        config; parse() rejects a block without one.
  const std::string &model_name() const { return model_name_; }

  /// @brief Where diagnostics and the coverage report go. Optional; without one
  ///        they go to the log.
  void set_sink(PluginSink *sink) { sink_ = sink; }

  /// @brief Replace `timing.model_config` with the schema-resolved object.
  ///
  /// @details Written by TimingModelLoader once it has merged in the defaults
  /// the model's metadata declares. Only the loader has seen that schema, and
  /// the model reads its configuration back out through the host, so the
  /// resolved object has to land here or the defaults reach nobody. Called
  /// before the model is constructed, hence before any model thread exists.
  void set_model_config_json(std::string json) { model_config_json_ = std::move(json); }

  // -- TimingHost -----------------------------------------------------------

  std::uint64_t tune(std::string_view key, std::uint64_t pessimistic) const override;
  double tune_real(std::string_view key, double pessimistic) const override;
  std::uint64_t class_issue_cycles(InstClass cls) const override;
  std::string model_config_json() const override { return model_config_json_; }
  double clock_ghz() const override { return clock_ghz_; }
  void note_unmodeled(std::string_view effect) const override;
  void log(std::string_view message) const override;
  void write_coverage_report(std::string &out) const override;

  // -- Coverage -------------------------------------------------------------

  /// @brief Whether any parameter fell back to its pessimistic default.
  bool has_fallbacks() const;

  /// @brief Whether any effect was declared unmodelled.
  bool has_unmodeled() const;

private:
  TimingConfig() = default;

  /// @brief What a model asked for and what it got.
  struct Resolved {
    std::string value;
    /// @brief False when the config did not name the key and the model's
    ///        pessimistic default was used instead.
    bool from_config = false;
  };

  void record(std::string_view key, const std::string &value, bool from_config) const;

  std::string model_name_;
  double clock_ghz_ = 1.0;
  std::string model_config_json_ = "{}";

  /// @brief The flattened `timing.machine` block: dotted path to scalar.
  ///
  /// @details Flattened at parse time so a lookup is one map probe rather than
  /// a walk, and so an unreadable nested shape is diagnosed once rather than on
  /// every access.
  std::map<std::string, std::string, std::less<>> machine_;

  /// @brief Guards the two ledgers below, which are written from model threads.
  mutable std::mutex ledger_mutex_;
  mutable std::map<std::string, Resolved, std::less<>> queried_;
  mutable std::map<std::string, std::uint64_t, std::less<>> unmodeled_;

  PluginSink *sink_ = nullptr;
};

} // namespace rocjitsu::timing
