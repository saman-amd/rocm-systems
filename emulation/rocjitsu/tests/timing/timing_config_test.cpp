// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_config_test.cpp
/// @brief TimingConfig: the `timing` block, and the ledger that makes a run
///        auditable.
///
/// @details Two things are being tested here and they matter for different
/// reasons. The parsing half is ordinary: a block is found or it is not. The
/// resolution half is the fail-slow rule made mechanical — a parameter the
/// config forgot resolves to the caller's pessimistic value, an instruction
/// class the config never named is charged the most expensive cost in the file,
/// and both facts are written into the coverage report rather than absorbed.
/// The class_issue_cycles() test with a cheap class, an expensive class and a
/// question about a third is the centre of that: it is the mechanism by which
/// InstClass::Unknown becomes expensive without any config file anticipating it.

#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/timing/timing_config.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace rocjitsu::timing {
namespace {

bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

/// @brief A full config document with a timing block inside it, so the tests
///        exercise the same "find our block among the others" path production
///        does rather than a bare fragment.
constexpr const char *kFullDocument = R"({
  "plugins": { "logging": {} },
  "sinks": { "types": ["stderr"] },
  "timing": {
    "model": "leaky",
    "clock_mhz": 2100,
    "machine": {
      "vector_alu": { "issue_cycles": 4 },
      "matrix_multiply": { "issue_cycles": 64 },
      "compute_units": 256,
      "global": { "bytes_per_cycle": 128.5 }
    },
    "model_config": { "alpha": 3 }
  }
})";

std::unique_ptr<TimingConfig> parse_full() {
  auto config = TimingConfig::parse(kFullDocument);
  return config;
}

TEST(TimingConfigTest, ParsesTheTimingBlockOutOfAFullDocument) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->model_name(), "leaky");
  EXPECT_DOUBLE_EQ(config->clock_ghz(), 2.1);
  EXPECT_TRUE(contains(config->model_config_json(), "alpha"));

  // The nested machine block is addressed by dotted path.
  EXPECT_EQ(config->tune("vector_alu.issue_cycles", 999), 4u);
  EXPECT_EQ(config->tune("compute_units", 1), 256u);
  EXPECT_DOUBLE_EQ(config->tune_real("global.bytes_per_cycle", 1.0), 128.5);
}

TEST(TimingConfigTest, ReturnsNullWhenTheDocumentHasNoTimingBlock) {
  EXPECT_EQ(TimingConfig::parse(R"({"plugins": {"logging": {}}})"), nullptr);
  EXPECT_EQ(TimingConfig::parse(R"({})"), nullptr);
}

/// @brief A block naming no model is a configuration error, not an invitation
///        to pick one: a run must never report numbers from a model nobody
///        chose. The refusal is logged; only the null is observable here.
TEST(TimingConfigTest, ReturnsNullWhenTheBlockNamesNoModel) {
  EXPECT_EQ(TimingConfig::parse(R"({"timing": {"clock_mhz": 2100}})"), nullptr);
  EXPECT_EQ(TimingConfig::parse(R"({"timing": {"model": "", "clock_mhz": 2100}})"), nullptr);
}

/// @brief Without a clock the numbers must be obviously unscaled rather than
///        subtly so, which is what 1 GHz buys: cycles and nanoseconds coincide.
TEST(TimingConfigTest, MissingClockDefaultsToOneGigahertz) {
  auto config = TimingConfig::parse(R"({"timing": {"model": "leaky"}})");
  ASSERT_NE(config, nullptr);
  EXPECT_DOUBLE_EQ(config->clock_ghz(), 1.0);
  EXPECT_EQ(config->model_config_json(), "{}");
}

TEST(TimingConfigTest, TuneReturnsTheConfiguredValueAndRecordsNoFallback) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->tune("compute_units", 1), 256u);
  EXPECT_FALSE(config->has_fallbacks());

  std::string report;
  config->write_coverage_report(report);
  EXPECT_TRUE(contains(report, "machine.compute_units = 256"));
  EXPECT_FALSE(contains(report, "NOT IN CONFIG"));
}

/// @brief The caller's value is the *pessimistic* one, and using it is recorded
///        either way — a parameter the config never named has to be visible in
///        the run, not only in the numbers it produced.
TEST(TimingConfigTest, TuneFallsBackToTheCallersPessimisticValueAndRecordsIt) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->tune("simd_lanes", 1), 1u);
  EXPECT_TRUE(config->has_fallbacks());

  std::string report;
  config->write_coverage_report(report);
  EXPECT_TRUE(contains(report, "machine.simd_lanes = 1"));
  EXPECT_TRUE(contains(report, "NOT IN CONFIG"));
}

TEST(TimingConfigTest, TuneRealFollowsTheSameRule) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);
  EXPECT_DOUBLE_EQ(config->tune_real("global.bytes_per_cycle", 1.0), 128.5);
  EXPECT_FALSE(config->has_fallbacks());
  EXPECT_DOUBLE_EQ(config->tune_real("lds.bytes_per_cycle", 0.5), 0.5);
  EXPECT_TRUE(config->has_fallbacks());
}

/// @brief A number written as a double still names an integer parameter; a
///        config author writing 256.0 should not silently lose the part.
TEST(TimingConfigTest, TuneAcceptsAnIntegerWrittenAsADouble) {
  auto config =
      TimingConfig::parse(R"({"timing": {"model": "leaky", "machine": {"compute_units": 256.0}}})");
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->tune("compute_units", 1), 256u);
  EXPECT_FALSE(config->has_fallbacks());
}

/// @brief A key that is present but unreadable falls back, and the fallback is
///        recorded: a typo'd value must not hide behind a plausible number.
TEST(TimingConfigTest, TuneFallsBackWhenTheConfiguredValueIsNotANumber) {
  auto config = TimingConfig::parse(
      R"({"timing": {"model": "leaky", "machine": {"compute_units": "many"}}})");
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->tune("compute_units", 7), 7u);
  EXPECT_TRUE(config->has_fallbacks());
}

TEST(TimingConfigTest, ClassIssueCyclesReturnsTheConfiguredValueForANamedClass) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);
  EXPECT_EQ(config->class_issue_cycles(InstClass::VectorAlu), 4u);
  EXPECT_EQ(config->class_issue_cycles(InstClass::MatrixMultiply), 64u);
  EXPECT_FALSE(config->has_fallbacks());
}

/// @brief The central fail-slow mechanism.
///
/// @details The config names a cheap class and an expensive one and says
/// nothing about a third. The answer for the third must be the expensive cost,
/// never the cheap one and never a fresh default: that is what makes
/// InstClass::Unknown — and any class added to the taxonomy after a config file
/// was written — cost as much as the most expensive thing the part can do,
/// without the file having to anticipate it.
TEST(TimingConfigTest, UnnamedClassIsChargedTheMostExpensiveNamedCost) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);

  const std::uint64_t cheap = config->class_issue_cycles(InstClass::VectorAlu);
  const std::uint64_t expensive = config->class_issue_cycles(InstClass::MatrixMultiply);
  ASSERT_LT(cheap, expensive);

  EXPECT_EQ(config->class_issue_cycles(InstClass::Transcendental), expensive);
  EXPECT_EQ(config->class_issue_cycles(InstClass::Unknown), expensive);
  EXPECT_TRUE(config->has_fallbacks());

  std::string report;
  config->write_coverage_report(report);
  EXPECT_TRUE(contains(report, "machine.unknown.issue_cycles = 64"));
  EXPECT_TRUE(contains(report, "NOT IN CONFIG"));
}

/// @brief With no class named at all there is no "most expensive" to borrow,
///        and the answer still must not be zero — a class that costs nothing
///        drains against an idle port and disappears from the model entirely.
TEST(TimingConfigTest, UnnamedClassIsNeverFreeEvenWithAnEmptyMachineBlock) {
  auto config = TimingConfig::parse(R"({"timing": {"model": "leaky"}})");
  ASSERT_NE(config, nullptr);
  EXPECT_GE(config->class_issue_cycles(InstClass::Unknown), 1u);
  EXPECT_GE(config->class_issue_cycles(InstClass::VectorAlu), 1u);
}

TEST(TimingConfigTest, CoverageReportListsEveryParameterAskedForAndMarksTheGaps) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);

  config->tune("compute_units", 1);              // named
  config->tune("simd_lanes", 1);                 // absent
  config->tune_real("lds.bytes_per_cycle", 0.5); // absent

  std::string report;
  config->write_coverage_report(report);

  EXPECT_TRUE(contains(report, "timing model: leaky"));
  EXPECT_TRUE(contains(report, "parameters in effect (3)"));
  EXPECT_TRUE(contains(report, "machine.compute_units = 256"));
  EXPECT_TRUE(contains(report, "machine.simd_lanes = 1   [NOT IN CONFIG - pessimistic default]"));
  EXPECT_TRUE(contains(report, "machine.lds.bytes_per_cycle"));
  EXPECT_TRUE(contains(report, "unmodelled effects: none declared"));
}

TEST(TimingConfigTest, CoverageReportListsTheUnmodelledLedgerWithCounts) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);
  EXPECT_FALSE(config->has_unmodeled());

  config->note_unmodeled("lds bank conflicts");
  config->note_unmodeled("memory access with unrecoverable addresses");
  config->note_unmodeled("memory access with unrecoverable addresses");
  config->note_unmodeled("memory access with unrecoverable addresses");

  EXPECT_TRUE(config->has_unmodeled());

  std::string report;
  config->write_coverage_report(report);
  EXPECT_TRUE(contains(report, "unmodelled effects (2 kinds"));
  EXPECT_TRUE(contains(report, "NOT in the reported time"));
  EXPECT_TRUE(contains(report, "lds bank conflicts   x1"));
  EXPECT_TRUE(contains(report, "memory access with unrecoverable addresses   x3"));
}

TEST(TimingConfigTest, DiagnosticsGoToTheSinkWhenOneIsInstalled) {
  auto config = parse_full();
  ASSERT_NE(config, nullptr);

  StringSink sink;
  config->set_sink(&sink);
  config->log("model says hello");
  EXPECT_TRUE(contains(sink.str(), "model says hello"));
}

} // namespace
} // namespace rocjitsu::timing
