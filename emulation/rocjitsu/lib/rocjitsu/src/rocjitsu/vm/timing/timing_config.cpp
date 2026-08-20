// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/timing_config.h"

#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"

#include "flatbuffers/flexbuffers.h"
#include "util/log.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>

namespace rocjitsu::timing {
namespace {

/// @brief Flatten a nested map into dotted paths, keeping scalars only.
///
/// @details Values are stored as strings rather than as a variant because the
/// only consumers are tune() and tune_real(), which know the type they want,
/// and because the coverage report has to print them anyway.
void flatten(const flexbuffers::Reference &value, const std::string &prefix,
             std::map<std::string, std::string, std::less<>> &out) {
  // An absent block is an empty map, not a map with one null entry under the
  // empty key. Nothing looks that key up, but a machine_ that is non-empty when
  // the config named nothing would make any future "did the config say
  // anything?" check quietly wrong.
  if (value.IsNull())
    return;
  if (value.IsMap()) {
    auto map = value.AsMap();
    auto keys = map.Keys();
    auto values = map.Values();
    for (std::size_t i = 0; i < keys.size(); ++i) {
      const std::string key = keys[i].AsKey();
      flatten(values[i], prefix.empty() ? key : prefix + "." + key, out);
    }
    return;
  }
  if (value.IsVector())
    return; // No parameter is a list; ignoring one is better than inventing a shape.
  out[prefix] = value.ToString();
}

/// @brief Re-emit a flexbuffers value as JSON, for the model's private config.
///
/// @details The model parses it itself, so the host does no validation beyond
/// producing something parseable. flexbuffers::Reference::ToString() already
/// emits JSON syntax for maps and vectors.
std::string to_json(const flexbuffers::Reference &value) {
  if (value.IsNull())
    return "{}";
  return value.ToString();
}

bool parse_u64(const std::string &text, std::uint64_t &out) {
  const char *begin = text.data();
  const char *end = begin + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, out);
  if (ec == std::errc() && ptr == end)
    return true;
  // A number that came through as a double ("256.0") still names an integer.
  // A number that came through as a double ("256.0") still names an integer,
  // but the range has to be checked before the cast rather than after: "1e400"
  // parses to infinity, and converting that is undefined rather than merely
  // wrong.
  char *tail = nullptr;
  const double value = std::strtod(text.c_str(), &tail);
  if (tail == text.c_str() || *tail != '\0' || !std::isfinite(value) || value < 0.0 ||
      value >= 9.0e18)
    return false;
  out = static_cast<std::uint64_t>(value);
  return true;
}

bool parse_double(const std::string &text, double &out) {
  char *tail = nullptr;
  const double value = std::strtod(text.c_str(), &tail);
  if (tail == text.c_str() || *tail != '\0')
    return false;
  out = value;
  return true;
}

} // namespace

TimingConfig::~TimingConfig() = default;

std::unique_ptr<TimingConfig> TimingConfig::parse(const std::string &config_json) {
  flexbuffers::Builder builder;
  if (!plugin_detail::flexbuffer_from_json(config_json, builder))
    return nullptr;

  auto root = flexbuffers::GetRoot(builder.GetBuffer());
  if (!root.IsMap())
    return nullptr;
  auto timing = root.AsMap()["timing"];
  if (!timing.IsMap())
    return nullptr;

  auto map = timing.AsMap();
  auto model = map["model"];
  if (model.IsNull() || model.ToString().empty()) {
    // A timing block naming no model is a configuration error rather than a
    // reason to pick one: silently defaulting is how a run ends up reporting
    // numbers from a model nobody chose.
    util::Logger::warn("timing: config has a 'timing' block with no 'model'; ignoring it");
    return nullptr;
  }

  std::unique_ptr<TimingConfig> config(new TimingConfig());
  config->model_name_ = model.ToString();

  auto clock_mhz = map["clock_mhz"];
  if (!clock_mhz.IsNull()) {
    config->clock_ghz_ = clock_mhz.AsDouble() / 1000.0;
  } else {
    // 1 GHz, so cycles and nanoseconds coincide and a config that forgot the
    // clock produces numbers that are obviously unscaled rather than subtly so.
    util::Logger::warn("timing: 'timing.clock_mhz' is not set; assuming 1000 MHz");
  }

  flatten(map["machine"], "", config->machine_);
  config->model_config_json_ = to_json(map["model_config"]);
  return config;
}

void TimingConfig::record(std::string_view key, const std::string &value, bool from_config) const {
  std::lock_guard lock(ledger_mutex_);
  queried_.insert_or_assign(std::string(key), Resolved{value, from_config});
}

std::uint64_t TimingConfig::tune(std::string_view key, std::uint64_t pessimistic) const {
  auto it = machine_.find(key);
  std::uint64_t value = pessimistic;
  bool from_config = false;
  if (it != machine_.end() && parse_u64(it->second, value)) {
    from_config = true;
  } else if (it != machine_.end()) {
    // Present but unreadable. Falling back is right, but silently doing so
    // would hide a typo'd value behind a plausible number.
    util::Logger::warn("timing: machine.", std::string(key), " = '", it->second,
                       "' is not an integer; using ", std::to_string(pessimistic));
    value = pessimistic;
  }
  record(key, std::to_string(value), from_config);
  return value;
}

double TimingConfig::tune_real(std::string_view key, double pessimistic) const {
  auto it = machine_.find(key);
  double value = pessimistic;
  bool from_config = false;
  if (it != machine_.end() && parse_double(it->second, value)) {
    from_config = true;
  } else if (it != machine_.end()) {
    util::Logger::warn("timing: machine.", std::string(key), " = '", it->second,
                       "' is not a number; using ", std::to_string(pessimistic));
    value = pessimistic;
  }
  record(key, std::to_string(value), from_config);
  return value;
}

std::uint64_t TimingConfig::class_issue_cycles(InstClass cls) const {
  const std::string key = std::string(inst_class_name(cls)) + ".issue_cycles";
  auto it = machine_.find(key);
  std::uint64_t value = 0;
  if (it != machine_.end() && parse_u64(it->second, value)) {
    record(key, std::to_string(value), true);
    return value;
  }

  // The class is not named. Charge it the most expensive issue cost the config
  // gives anything, so an unclassified or newly added class is never the cheap
  // one. This is what makes InstClass::Unknown expensive without every config
  // file having to anticipate it.
  // One cycle, reached only when the config names no class at all. Not a
  // pessimistic value — there is nothing to be pessimistic relative to — but
  // deliberately nonzero, so a config that describes no machine still costs
  // every instruction something and the run does not report a kernel as free.
  // The coverage report marks every class as a fallback in that case, which is
  // the signal that the number means nothing.
  std::uint64_t worst = 1;
  for (std::size_t i = 0; i < kNumInstClasses; ++i) {
    auto named =
        machine_.find(std::string(inst_class_name(static_cast<InstClass>(i))) + ".issue_cycles");
    std::uint64_t candidate = 0;
    if (named != machine_.end() && parse_u64(named->second, candidate))
      worst = std::max(worst, candidate);
  }
  record(key, std::to_string(worst), false);
  return worst;
}

void TimingConfig::note_unmodeled(std::string_view effect) const {
  std::lock_guard lock(ledger_mutex_);
  ++unmodeled_[std::string(effect)];
}

void TimingConfig::log(std::string_view message) const {
  if (sink_)
    sink_->write(std::string(message));
  else
    util::Logger::warn("timing: ", std::string(message));
}

bool TimingConfig::has_fallbacks() const {
  std::lock_guard lock(ledger_mutex_);
  return std::any_of(queried_.begin(), queried_.end(),
                     [](const auto &entry) { return !entry.second.from_config; });
}

bool TimingConfig::has_unmodeled() const {
  std::lock_guard lock(ledger_mutex_);
  return !unmodeled_.empty();
}

void TimingConfig::write_coverage_report(std::string &out) const {
  std::lock_guard lock(ledger_mutex_);

  out += "timing model: " + model_name_ + "\n";
  out += "shader clock: " + std::to_string(clock_ghz_ * 1000.0) + " MHz\n";

  out += "parameters in effect (" + std::to_string(queried_.size()) + "):\n";
  for (const auto &[key, resolved] : queried_) {
    out += "  machine." + key + " = " + resolved.value;
    if (!resolved.from_config)
      out += "   [NOT IN CONFIG - pessimistic default]";
    out += "\n";
  }

  if (unmodeled_.empty()) {
    out += "unmodelled effects: none declared\n";
    return;
  }
  out += "unmodelled effects (" + std::to_string(unmodeled_.size()) +
         " kinds; these are NOT in the reported time):\n";
  for (const auto &[effect, count] : unmodeled_)
    out += "  " + effect + "   x" + std::to_string(count) + "\n";
}

} // namespace rocjitsu::timing
