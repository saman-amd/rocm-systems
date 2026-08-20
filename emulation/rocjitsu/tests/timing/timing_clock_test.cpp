// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_clock_test.cpp
/// @brief SimulatedClock: the only clock a guest under rocjitsu can observe.
///
/// @details The clock is a process-wide singleton with monotonic floors that
/// are never reset, which is a deliberate property rather than an inconvenience
/// — a guest must never see time retreat, including across a plugin being
/// installed halfway through a run. Every test here is therefore written
/// against *deltas* it establishes itself rather than against absolute values,
/// and every one restores the unbound state before the source it installed goes
/// out of scope: SimulatedClock never frees a binding, so a reader could still
/// hold a pointer to a source whose storage has been reclaimed.
///
/// The source stub is local and two lines long, which is the point of
/// TimeSource being a separate type from TimingModel.

#include "rocjitsu/vm/timing/event.h"
#include "rocjitsu/vm/timing/simulated_clock.h"
#include "rocjitsu/vm/timing/time_source.h"
#include "timing/mock_timing_model.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

/// @brief A time source whose cycle count the test writes directly.
class StubSource final : public timing::TimeSource {
public:
  explicit StubSource(std::uint64_t cycles = 0, double clock_ghz = 1.0)
      : cycles_(cycles), clock_ghz_(clock_ghz) {}

  void set_cycles(std::uint64_t cycles) { cycles_.store(cycles, std::memory_order_relaxed); }
  void advance(std::uint64_t delta) { cycles_.fetch_add(delta, std::memory_order_relaxed); }

  std::uint64_t current_cycles() const override { return cycles_.load(std::memory_order_relaxed); }
  double clock_ghz() const override { return clock_ghz_; }

private:
  std::atomic<std::uint64_t> cycles_;
  double clock_ghz_;
};

/// @brief Installs a source and takes it out again before it can die.
///
/// @details Order is the whole point, and it is why this is a guard rather than
/// fixture teardown: SimulatedClock never frees a binding and reads through it
/// on the way out, so the uninstall has to happen while the source is still
/// alive. Declared after the source it wraps, it is destroyed before it.
/// Getting this wrong in a test is a segfault; getting it wrong in the
/// simulator would be a use-after-free on a guest thread inside an ioctl.
class InstalledSource {
public:
  explicit InstalledSource(const timing::TimeSource &source) {
    SimulatedClock::instance().set_time_source(&source);
  }
  ~InstalledSource() { SimulatedClock::instance().set_time_source(nullptr); }

  InstalledSource(const InstalledSource &) = delete;
  InstalledSource &operator=(const InstalledSource &) = delete;
};

class TimingClockTest : public ::testing::Test {
protected:
  /// @details Start every test from the unbound state, so each one measures
  /// deltas against host time rather than against whatever the last test left
  /// behind. The floors themselves are process-wide and never reset, which is
  /// deliberate — see the file comment.
  void SetUp() override { clock().set_time_source(nullptr); }

  static SimulatedClock &clock() { return SimulatedClock::instance(); }

  static void rest() { std::this_thread::sleep_for(std::chrono::milliseconds(3)); }
};

TEST_F(TimingClockTest, ReportsHostTimeAndAdvancesWithNoSourceInstalled) {
  EXPECT_FALSE(clock().is_simulated());

  const std::uint64_t first = clock().nanoseconds();
  rest();
  const std::uint64_t second = clock().nanoseconds();

  EXPECT_GT(second, first);
  EXPECT_GE(second - first, 1'000'000u) << "host time should have moved by at least a millisecond";

  // Shader cycles must keep moving too: a kernel spinning until the counter
  // changes would otherwise never make progress on an unmodelled run.
  const std::uint64_t cycles = clock().shader_cycles();
  rest();
  EXPECT_GT(clock().shader_cycles(), cycles);
}

/// @brief Installing a source continues the timeline instead of restarting it,
///        in both the nanosecond and the cycle domain.
///
/// @details This is the regression test for a prototype defect worth stating in
/// full. The cycle domain clamped to its monotonic floor without rebasing, and
/// the floor already held a host-derived magnitude far above anything a model
/// starting at zero could produce. shader_cycles() therefore returned that same
/// constant forever, so every in-kernel s_memtime pair differed by exactly
/// zero and self-timing kernels reported no elapsed time at all — a failure
/// that looks like a working clock right up until someone subtracts two
/// readings. The exact-delta assertions below are what a clamp-without-rebase
/// cannot satisfy: under that bug both deltas are 0.
TEST_F(TimingClockTest, InstallingASourceRebasesRatherThanClamps) {
  const std::uint64_t ns_before = clock().nanoseconds();
  const std::uint64_t cycles_before = clock().shader_cycles();

  StubSource source(0, 2.0);
  const InstalledSource installed(source);
  EXPECT_TRUE(clock().is_simulated());

  const std::uint64_t ns_at_install = clock().nanoseconds();
  const std::uint64_t cycles_at_install = clock().shader_cycles();

  EXPECT_GE(ns_at_install, ns_before) << "the clock must not retreat across an install";
  EXPECT_GE(cycles_at_install, cycles_before);
  EXPECT_LT(ns_at_install - ns_before, 1'000'000'000u) << "continued, rather than leaping";
  EXPECT_LT(cycles_at_install - cycles_before, 1'000'000'000u);

  source.set_cycles(1'000);
  EXPECT_EQ(clock().shader_cycles() - cycles_at_install, 1'000u)
      << "the cycle domain must track the model exactly, not sit on a stale floor";
  EXPECT_EQ(clock().nanoseconds() - ns_at_install, 500u) << "1000 cycles at 2 GHz is 500 ns";

  source.advance(3'000);
  EXPECT_EQ(clock().shader_cycles() - cycles_at_install, 4'000u);
  EXPECT_EQ(clock().nanoseconds() - ns_at_install, 2'000u);
}

TEST_F(TimingClockTest, UninstallingRebasesBackOntoHostTimeWithoutJumpingOrFreezing) {
  StubSource source(0, 1.0);
  const InstalledSource installed(source);
  source.set_cycles(1'000'000); // a millisecond of simulated time

  const std::uint64_t simulated = clock().nanoseconds();
  clock().set_time_source(nullptr);
  EXPECT_FALSE(clock().is_simulated());

  const std::uint64_t after = clock().nanoseconds();
  EXPECT_GE(after, simulated);
  // Without an origin for the host domain the absolute host nanosecond count
  // would be added to the base and the clock would leap forward by decades.
  EXPECT_LT(after - simulated, 1'000'000'000u) << "returning to host time must not jump";

  rest();
  EXPECT_GT(clock().nanoseconds(), after) << "returning to host time must not freeze";
}

/// @brief A model that retreats must not be able to make a guest compute a
///        negative duration; the clock is the last line of defence.
TEST_F(TimingClockTest, IsMonotonicUnderASourceThatMovesBackwards) {
  timing::test::RetreatingModel model(0, 1.0);
  timing::test::ModelTimeSource source(model);
  const InstalledSource installed(source);

  const std::uint64_t base_ns = clock().nanoseconds();
  const std::uint64_t base_cycles = clock().shader_cycles();

  model.set_cycles(10'000);
  const std::uint64_t high_ns = clock().nanoseconds();
  const std::uint64_t high_cycles = clock().shader_cycles();
  EXPECT_EQ(high_cycles - base_cycles, 10'000u);
  EXPECT_EQ(high_ns - base_ns, 10'000u);

  model.set_cycles(1'000);
  EXPECT_EQ(clock().nanoseconds(), high_ns) << "the retreat must be absorbed, not exposed";
  EXPECT_EQ(clock().shader_cycles(), high_cycles);

  model.set_cycles(0);
  EXPECT_EQ(clock().nanoseconds(), high_ns);
  EXPECT_EQ(clock().shader_cycles(), high_cycles);
}

/// @brief ROCR calibrates by reading the clock counters ioctl twice and
///        dividing by the difference, so two reads that agree are a divide by
///        zero rather than a harmless duplicate.
TEST_F(TimingClockTest, CounterNanosecondsStrictlyIncreasesWithAFrozenSource) {
  StubSource source(4'096, 1.0);
  const InstalledSource installed(source);

  std::uint64_t previous = clock().counter_nanoseconds();
  for (int i = 0; i < 64; ++i) {
    const std::uint64_t now = clock().counter_nanoseconds();
    ASSERT_GT(now, previous) << "iteration " << i;
    previous = now;
  }

  // The plain reading is deliberately *not* strictly increasing: everywhere
  // else, two reads at the same instant should agree.
  EXPECT_EQ(clock().nanoseconds(), clock().nanoseconds());
}

/// @brief The wall clock is a separate counter because the guest divides it by
///        a separately advertised rate. Reporting shader cycles here would
///        over-report elapsed time by the ratio of the two clocks: correctly
///        typed, plausibly shaped and silently wrong.
TEST_F(TimingClockTest, WallClockTicksAdvanceAtTheWallClockRateNotTheShaderClock) {
  StubSource source(0, 2.0);
  const InstalledSource installed(source);

  const std::uint64_t ns_before = clock().nanoseconds();
  const std::uint64_t ticks_before = clock().wall_clock_ticks();
  const std::uint64_t cycles_before = clock().shader_cycles();

  source.set_cycles(2'000'000);

  const std::uint64_t ns_delta = clock().nanoseconds() - ns_before;
  const std::uint64_t ticks_delta = clock().wall_clock_ticks() - ticks_before;
  const std::uint64_t cycles_delta = clock().shader_cycles() - cycles_before;

  EXPECT_EQ(cycles_delta, 2'000'000u);
  EXPECT_EQ(ns_delta, 1'000'000u) << "2,000,000 cycles at 2 GHz is 1 ms";

  const std::uint64_t ratio =
      SimulatedClock::kTimestampFrequencyHz / SimulatedClock::kWallClockFrequencyHz;
  const std::uint64_t expected = ns_delta / ratio;
  // Off by at most one tick: wall_clock_ticks() truncates an absolute value,
  // and the two endpoints need not fall on tick boundaries.
  EXPECT_GE(ticks_delta + 1, expected);
  EXPECT_LE(ticks_delta, expected + 1);
  EXPECT_NE(ticks_delta, cycles_delta) << "the wall clock is not the shader clock";
}

TEST_F(TimingClockTest, ShaderClockHzIsZeroWithoutAModelAndTheModelsRateWithOne) {
  EXPECT_EQ(clock().shader_clock_hz(), 0u);

  StubSource source(0, 2.5);
  {
    const InstalledSource installed(source);
    EXPECT_EQ(clock().shader_clock_hz(), 2'500'000'000u);
  }
  EXPECT_EQ(clock().shader_clock_hz(), 0u);
}

/// @brief Readers race installs for real: in daemon mode the uninstall runs on
///        the engine thread while client threads are still answering driver
///        queries. Nothing here checks a value — only that no reader ever sees
///        a domain retreat and that the singleton survives the churn. Under a
///        sanitizer build this is also the data-race check.
TEST_F(TimingClockTest, ConcurrentReadersNeverSeeTimeRetreatWhileSourcesChange) {
  StubSource fast(0, 2.0);
  StubSource slow(0, 0.5);

  std::atomic<bool> stop{false};
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&stop] {
      std::uint64_t last_ns = 0;
      std::uint64_t last_cycles = 0;
      std::uint64_t last_counter = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        const std::uint64_t ns = SimulatedClock::instance().nanoseconds();
        ASSERT_GE(ns, last_ns);
        last_ns = ns;

        const std::uint64_t cycles = SimulatedClock::instance().shader_cycles();
        ASSERT_GE(cycles, last_cycles);
        last_cycles = cycles;

        const std::uint64_t counter = SimulatedClock::instance().counter_nanoseconds();
        ASSERT_GT(counter, last_counter);
        last_counter = counter;
      }
    });
  }

  for (int i = 0; i < 300; ++i) {
    fast.advance(1'000);
    slow.advance(1'000);
    if (i % 3 == 0)
      clock().set_time_source(nullptr);
    else if (i % 3 == 1)
      clock().set_time_source(&fast);
    else
      clock().set_time_source(&slow);
  }

  stop.store(true, std::memory_order_relaxed);
  for (std::thread &reader : readers)
    reader.join();

  // Uninstall while the sources are still alive and every reader has stopped:
  // a binding outlives its installation and a reader may hold one across a swap.
  clock().set_time_source(nullptr);
}

/// @brief With a model whose cost is a constant, guest-visible time is exactly
///        predictable: N instructions at C cycles each, converted at the
///        model's clock.
///
/// @details Asserting the exact figure rather than a bound is the only way to
/// catch a conversion that is off by the clock ratio. Such an error leaves the
/// numbers plausibly shaped — right unit, right order of magnitude, wrong by a
/// constant factor — and a bound would accept it.
TEST_F(TimingClockTest, GuestTimeIsExactlyWhatTheModelCharged) {
  constexpr std::uint64_t kCyclesPerInstruction = 7;
  constexpr std::uint64_t kInstructions = 1'000;

  timing::test::FixedCostModel model(kCyclesPerInstruction, /*clock_ghz=*/2.0);
  timing::test::ModelTimeSource source(model);
  const InstalledSource installed(source);

  const std::uint64_t ns_before = clock().nanoseconds();
  const std::uint64_t cycles_before = clock().shader_cycles();

  const timing::WaveRef wave;
  const timing::InstructionEvent event;
  for (std::uint64_t i = 0; i < kInstructions; ++i)
    model.on_instruction(wave, event);

  EXPECT_EQ(model.instructions(), kInstructions);
  EXPECT_EQ(clock().shader_cycles() - cycles_before, kInstructions * kCyclesPerInstruction);
  EXPECT_EQ(static_cast<double>(clock().nanoseconds() - ns_before),
            model.nanoseconds_for(kInstructions));
  EXPECT_EQ(clock().shader_clock_hz(), 2'000'000'000u);
}

} // namespace
} // namespace rocjitsu::amdgpu
