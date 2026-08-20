// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_leaky_model_test.cpp
/// @brief The sample model, driven entirely from hand-built events.
///
/// @details No simulator, no compiled kernel and no GPU appears anywhere in
/// this file, which is the property event.h was shaped to give. It buys
/// precision that a kernel-driven test cannot: a leaky bucket is arithmetic
/// over four accumulators, and the way to test arithmetic is to state the
/// inputs and the answer, not to hope a kernel happens to reach the case.
///
/// Two of these tests are regressions for defects the prototype shipped with,
/// and both are called out where they sit. The one worth reading first is
/// ReportedCyclesAndTheClockAgree: a per-dispatch report and the guest-visible
/// clock disagreeing by a factor of three is the failure mode this whole API is
/// arranged to make impossible, and nothing catches it except asserting the two
/// against each other.

#include "rocjitsu/vm/timing/event.h"
#include "rocjitsu/vm/timing/models/leaky/model.h"
#include "timing/mock_timing_model.h"

#include <gtest/gtest.h>

#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::timing::leaky {
namespace {

/// @brief Every "N cycles" figure the report attributes to a dispatch.
std::vector<std::uint64_t> reported_cycles(const std::string &report) {
  std::vector<std::uint64_t> values;
  const std::string marker = " cycles (";
  std::size_t pos = 0;
  while ((pos = report.find(marker, pos)) != std::string::npos) {
    std::size_t begin = pos;
    while (begin > 0 && std::isdigit(static_cast<unsigned char>(report[begin - 1])))
      --begin;
    if (begin < pos)
      values.push_back(std::stoull(report.substr(begin, pos - begin)));
    pos += marker.size();
  }
  return values;
}

class TimingLeakyModelTest : public ::testing::Test {
protected:
  /// @details A deliberately boring machine: one compute unit, one port each,
  /// a 64-wide SIMD and bandwidth so large it never binds. That leaves the
  /// issue buckets as the only thing under test, so each assertion below is
  /// about one term of the arithmetic rather than about which term won.
  void SetUp() override { describe_machine(host); }

  /// @brief The machine above, written into @p out. A MockTimingHost holds a
  ///        mutex and cannot be copied, so a test that needs a second ledger
  ///        builds a second host from here rather than duplicating one.
  static void describe_machine(timing::test::MockTimingHost &out) {
    out.set_clock_ghz(1.0);
    out.set_int("compute_units", 1);
    out.set_int("simd_lanes", 64);
    out.set_int("dispatch_latency_cycles", 0);
    out.set_real("global.bytes_per_cycle", 1.0e9);
    out.set_real("lds.bytes_per_cycle", 1.0e9);
    for (std::size_t i = 0; i < kNumFunctionalUnits; ++i)
      out.set_int(std::string(functional_unit_name(static_cast<FunctionalUnit>(i))) + ".ports", 1);
    for (std::size_t i = 0; i < kNumInstClasses; ++i)
      out.set_class_issue_cycles(static_cast<InstClass>(i), 1);
  }

  static WaveRef wave_of(DispatchKey key) {
    WaveRef wave;
    wave.dispatch = key;
    wave.wave_lanes = 64;
    return wave;
  }

  InstructionEvent event_of(InstClass cls) {
    InstructionEvent event;
    event.info = &info;
    event.effective_class = cls;
    event.active_lanes = 64;
    event.wave_lanes = 64;
    return event;
  }

  static DispatchInfo dispatch_of(DispatchKey key, std::string name) {
    DispatchInfo info;
    info.key = key;
    info.kernel_name = std::move(name);
    return info;
  }

  timing::test::MockTimingHost host;
  StaticInstInfo info;
};

// -- drain_cycles arithmetic -------------------------------------------------

TEST_F(TimingLeakyModelTest, DrainCyclesTakesTheFullestBucket) {
  host.set_int("compute_units", 2);
  host.set_int("vector_alu.ports", 2);
  host.set_int("matrix_multiply.ports", 1);
  host.set_real("global.bytes_per_cycle", 10.0);
  host.set_real("lds.bytes_per_cycle", 1.0);
  LeakyBucketModel model(host);

  Buckets buckets;
  buckets.unit_cycles[static_cast<std::size_t>(FunctionalUnit::VectorAlu)] = 100;     // /4 = 25
  buckets.unit_cycles[static_cast<std::size_t>(FunctionalUnit::MatrixMultiply)] = 30; // /2 = 15
  buckets.global_bytes = 100;                                                         // /10 = 10
  buckets.lds_bytes = 5;                                                              // /1  = 5
  EXPECT_EQ(model.drain_cycles(buckets), 25u);

  buckets.unit_cycles[static_cast<std::size_t>(FunctionalUnit::MatrixMultiply)] = 200; // /2 = 100
  EXPECT_EQ(model.drain_cycles(buckets), 100u);

  buckets.global_bytes = 3'000; // /10 = 300, now the widest
  EXPECT_EQ(model.drain_cycles(buckets), 300u);

  buckets.lds_bytes = 900; // /1 = 900
  EXPECT_EQ(model.drain_cycles(buckets), 900u);
}

TEST_F(TimingLeakyModelTest, DrainCyclesDividesAUnitBucketByComputeUnitsTimesPorts) {
  Buckets buckets;
  buckets.unit_cycles[static_cast<std::size_t>(FunctionalUnit::VectorAlu)] = 100;

  {
    LeakyBucketModel model(host);
    EXPECT_EQ(model.drain_cycles(buckets), 100u);
  }
  host.set_int("compute_units", 4);
  {
    LeakyBucketModel model(host);
    EXPECT_EQ(model.drain_cycles(buckets), 25u);
  }
  host.set_int("vector_alu.ports", 2);
  {
    LeakyBucketModel model(host);
    EXPECT_EQ(model.drain_cycles(buckets), 13u) << "ceil(100 / 8): a partial cycle still costs one";
  }
}

/// @brief The floor covers everything between the packet and the first wave
///        that this model does not model, so it applies even to a dispatch
///        whose buckets are empty.
TEST_F(TimingLeakyModelTest, DrainCyclesAppliesTheDispatchLatencyFloor) {
  host.set_int("dispatch_latency_cycles", 500);
  LeakyBucketModel model(host);

  EXPECT_EQ(model.drain_cycles(Buckets{}), 500u);

  Buckets buckets;
  buckets.unit_cycles[static_cast<std::size_t>(FunctionalUnit::VectorAlu)] = 100;
  EXPECT_EQ(model.drain_cycles(buckets), 500u) << "the floor wins while it is the larger term";

  buckets.unit_cycles[static_cast<std::size_t>(FunctionalUnit::VectorAlu)] = 900;
  EXPECT_EQ(model.drain_cycles(buckets), 900u) << "and stops winning when it is not";
}

/// @brief Work parked on FunctionalUnit::None is charged like any other bucket.
///
/// @details None is not "no cost" — it is the front-end issue slot every
/// instruction occupies before it reaches a unit, which is why waits, barriers
/// and no-ops land there. Excluding it from the drain would make an s_nop free
/// and a kernel padded with them arbitrarily fast, which is the unbounded
/// optimistic bias the fail-slow rule exists to prevent.
TEST_F(TimingLeakyModelTest, WorkOnTheNonIssuingUnitIsStillCharged) {
  Buckets buckets;
  buckets.unit_cycles[static_cast<std::size_t>(FunctionalUnit::None)] = 1'000;

  {
    LeakyBucketModel model(host);
    EXPECT_EQ(model.drain_cycles(buckets), 1'000u);
  }
  host.set_int("compute_units", 4);
  {
    LeakyBucketModel model(host);
    EXPECT_EQ(model.drain_cycles(buckets), 250u) << "and it scales with the part like any other";
  }
}

/// @brief A no-op is not free, end to end.
TEST_F(TimingLeakyModelTest, ANoOpCostsItsIssueSlot) {
  host.set_class_issue_cycles(InstClass::Nop, 1);
  LeakyBucketModel model(host);

  const DispatchKey key{.dispatch_id = 1, .queue_id = 0};
  for (int i = 0; i < 100; ++i)
    model.on_instruction(wave_of(key), event_of(InstClass::Nop));
  model.on_dispatch_end(key);

  EXPECT_EQ(model.device_cycles(), 100u);
}

// -- A dispatch, end to end --------------------------------------------------

TEST_F(TimingLeakyModelTest, DispatchOfHandBuiltEventsCostsWhatItSaysOnTheTin) {
  host.set_class_issue_cycles(InstClass::VectorAlu, 4);
  LeakyBucketModel model(host);

  const DispatchKey key{.dispatch_id = 1, .queue_id = 0};
  model.on_dispatch_begin(dispatch_of(key, "hand_built"));
  const WaveRef wave = wave_of(key);
  for (int i = 0; i < 10; ++i)
    model.on_instruction(wave, event_of(InstClass::VectorAlu));
  model.on_dispatch_end(key);

  // Ten instructions at four issue cycles, one pass each on a 64-wide SIMD,
  // draining through one port on one compute unit.
  EXPECT_EQ(model.device_cycles(), 40u);

  std::string report;
  model.write_report(report);
  EXPECT_NE(report.find("hand_built"), std::string::npos);
  ASSERT_EQ(reported_cycles(report).size(), 1u);
  EXPECT_EQ(reported_cycles(report)[0], 40u);
}

/// @brief A wave wider than the SIMD costs a pass per extra width, and the pass
///        count is taken from the wave width rather than the active lanes so
///        that divergence never reads as free.
TEST_F(TimingLeakyModelTest, AWaveWiderThanTheSimdCostsExtraPasses) {
  host.set_int("simd_lanes", 16);
  host.set_class_issue_cycles(InstClass::VectorAlu, 4);
  LeakyBucketModel model(host);

  const DispatchKey key{.dispatch_id = 1, .queue_id = 0};
  WaveRef wave = wave_of(key);
  wave.wave_lanes = 64;
  InstructionEvent event = event_of(InstClass::VectorAlu);
  event.active_lanes = 1; // heavily divergent, and charged as if it were not
  model.on_instruction(wave, event);
  model.on_dispatch_end(key);

  EXPECT_EQ(model.device_cycles(), 16u) << "4 issue cycles x ceil(64/16) passes";
}

/// @brief A scalar instruction is not lane-parallel, so widening the wave must
///        not multiply its cost.
TEST_F(TimingLeakyModelTest, ScalarWorkDoesNotPayPerSimdPass) {
  host.set_int("simd_lanes", 16);
  host.set_class_issue_cycles(InstClass::ScalarAlu, 4);
  LeakyBucketModel model(host);

  const DispatchKey key{.dispatch_id = 1, .queue_id = 0};
  model.on_instruction(wave_of(key), event_of(InstClass::ScalarAlu));
  model.on_dispatch_end(key);

  EXPECT_EQ(model.device_cycles(), 4u);
}

/// @brief Dispatch ids collide across command processors, so the queue half of
///        the key has to keep two unrelated kernels apart. Mixing them would
///        merge their buckets and cost one dispatch the other's work.
TEST_F(TimingLeakyModelTest, DispatchesWithTheSameIdOnDifferentQueuesStaySeparate) {
  host.set_class_issue_cycles(InstClass::VectorAlu, 1);
  LeakyBucketModel model(host);

  const DispatchKey left{.dispatch_id = 7, .queue_id = 0};
  const DispatchKey right{.dispatch_id = 7, .queue_id = 1};
  model.on_dispatch_begin(dispatch_of(left, "left"));
  model.on_dispatch_begin(dispatch_of(right, "right"));
  for (int i = 0; i < 3; ++i)
    model.on_instruction(wave_of(left), event_of(InstClass::VectorAlu));
  for (int i = 0; i < 5; ++i)
    model.on_instruction(wave_of(right), event_of(InstClass::VectorAlu));
  model.on_dispatch_end(left);
  model.on_dispatch_end(right);

  std::string report;
  model.write_report(report);
  const std::vector<std::uint64_t> cycles = reported_cycles(report);
  ASSERT_EQ(cycles.size(), 2u);
  EXPECT_EQ(cycles[0], 3u);
  EXPECT_EQ(cycles[1], 5u);
  EXPECT_EQ(model.device_cycles(), 8u);
}

/// @brief The report and the guest-visible clock must be the same number.
///
/// @details The prototype's per-dispatch report said 9.99 us for a kernel that
/// hipEventElapsedTime measured at 3.53 us, because a duration adjustment was
/// applied to the report and never reached the clock. Both numbers were
/// individually plausible, which is exactly why the disagreement survived: only
/// comparing them finds it. device_cycles() must advance by precisely the total
/// the report attributes to the dispatches, with nothing added and nothing lost.
TEST_F(TimingLeakyModelTest, ReportedCyclesAndTheClockAgree) {
  host.set_class_issue_cycles(InstClass::VectorAlu, 3);
  host.set_class_issue_cycles(InstClass::ScalarAlu, 7);
  LeakyBucketModel model(host);

  EXPECT_EQ(model.device_cycles(), 0u) << "an idle model has not moved the clock";

  std::uint64_t before = 0;
  for (std::uint32_t id = 1; id <= 3; ++id) {
    const DispatchKey key{.dispatch_id = id, .queue_id = 0};
    model.on_dispatch_begin(dispatch_of(key, "k" + std::to_string(id)));
    const WaveRef wave = wave_of(key);
    for (std::uint32_t i = 0; i < id * 4; ++i)
      model.on_instruction(wave, event_of(InstClass::VectorAlu));
    for (std::uint32_t i = 0; i < id; ++i)
      model.on_instruction(wave, event_of(InstClass::ScalarAlu));

    const std::uint64_t at_begin = model.device_cycles();
    model.on_dispatch_end(key);
    const std::uint64_t charged = model.device_cycles() - at_begin;

    std::string report;
    model.write_report(report);
    const std::vector<std::uint64_t> cycles = reported_cycles(report);
    ASSERT_EQ(cycles.size(), id);
    EXPECT_EQ(cycles.back(), charged)
        << "dispatch " << id << " was reported and charged differently";
    before += charged;
    EXPECT_EQ(model.device_cycles(), before);
  }

  std::string report;
  model.write_report(report);
  std::uint64_t total = 0;
  for (std::uint64_t cycles : reported_cycles(report))
    total += cycles;
  EXPECT_EQ(total, model.device_cycles());

  // And the same figure in the unit the guest actually reads.
  EXPECT_DOUBLE_EQ(model.clock_ghz(), 1.0);
}

// -- Fail-slow ---------------------------------------------------------------

/// @brief The structural gaps are on the ledger before a single event arrives,
///        because they are true of every instruction the model will ever see.
TEST_F(TimingLeakyModelTest, StructuralGapsAreDeclaredAtConstruction) {
  LeakyBucketModel model(host);
  EXPECT_GE(host.unmodeled_kinds(), 5u);
  EXPECT_GE(host.unmodeled_count("cache hierarchy (every access charged at the global rate)"), 1u);
  EXPECT_GE(host.unmodeled_count("latency and its exposure (throughput bound only)"), 1u);
}

/// @brief An opcode nobody classified costs as much as the most expensive thing
///        the part can do, and says so.
TEST_F(TimingLeakyModelTest, UnknownClassIsChargedTheMostExpensiveCostAndDeclared) {
  timing::test::MockTimingHost sparse;
  sparse.set_int("compute_units", 1);
  sparse.set_int("simd_lanes", 64);
  sparse.set_int("vector_alu.ports", 1);
  sparse.set_real("global.bytes_per_cycle", 1.0e9);
  sparse.set_real("lds.bytes_per_cycle", 1.0e9);
  // The floor is pinned to zero because this test is about the issue-cycle term
  // alone. Left unnamed it resolves to its own pessimistic default, which is
  // larger than either cost here and would floor both dispatches to the same
  // number — the fail-slow rule working correctly, and hiding what is under
  // test.
  sparse.set_int("dispatch_latency_cycles", 0);
  // Only two classes named: one cheap, one expensive, and nothing about Unknown.
  sparse.set_class_issue_cycles(InstClass::VectorAlu, 1);
  sparse.set_class_issue_cycles(InstClass::MatrixMultiply, 64);

  LeakyBucketModel model(sparse);
  const std::uint64_t declared_before = sparse.unmodeled_count("unclassified instruction");

  const DispatchKey known{.dispatch_id = 1, .queue_id = 0};
  model.on_instruction(wave_of(known), event_of(InstClass::VectorAlu));
  model.on_dispatch_end(known);
  const std::uint64_t cheap = model.device_cycles();

  const DispatchKey unknown{.dispatch_id = 2, .queue_id = 0};
  model.on_instruction(wave_of(unknown), event_of(InstClass::Unknown));
  model.on_dispatch_end(unknown);
  const std::uint64_t expensive = model.device_cycles() - cheap;

  EXPECT_EQ(cheap, 1u);
  EXPECT_EQ(expensive, 64u) << "an unclassified opcode costs the worst cost in the config";
  EXPECT_EQ(sparse.unmodeled_count("unclassified instruction"), declared_before + 1);
}

/// @brief An access whose addresses the observer could not recover must cost
///        more than the same access with addresses, never less.
///
/// @details It is the case where guessing cheap is most tempting and most
/// wrong: it costs nothing to compute and it silently deletes real traffic from
/// the run. Billing it as a full-width global transfer is the pessimistic
/// reading, and it goes on the ledger so the count says how much of the run the
/// model was guessing about.
TEST_F(TimingLeakyModelTest, UnrecoverableAddressesCostMoreThanKnownOnes) {
  timing::test::MockTimingHost known_host;
  timing::test::MockTimingHost unknown_host;
  for (timing::test::MockTimingHost *sink : {&known_host, &unknown_host}) {
    describe_machine(*sink);
    sink->set_real("global.bytes_per_cycle", 1.0);
    sink->set_class_issue_cycles(InstClass::VectorMemoryRead, 1);
  }

  const auto cost_of = [this](bool addresses_known, timing::test::MockTimingHost &sink) {
    LeakyBucketModel model(sink);
    const DispatchKey key{.dispatch_id = 1, .queue_id = 0};
    InstructionEvent event = event_of(InstClass::VectorMemoryRead);
    event.active_lanes = 1; // one lane's worth of addresses were recovered
    event.wave_lanes = 64;
    event.memory.space = MemorySpace::Global;
    event.memory.is_load = true;
    event.memory.bytes_per_lane = 4;
    event.memory.addresses_known = addresses_known;
    if (addresses_known)
      event.memory.lane_addresses.assign(1, 0x1000);
    model.on_instruction(wave_of(key), event);
    model.on_dispatch_end(key);
    return model.device_cycles();
  };

  const std::uint64_t known = cost_of(true, known_host);
  const std::uint64_t unknown = cost_of(false, unknown_host);

  EXPECT_EQ(known, 4u) << "one active lane at four bytes, one byte per cycle";
  EXPECT_EQ(unknown, 256u) << "billed as a full-width transfer instead";
  EXPECT_GT(unknown, known);
  EXPECT_EQ(known_host.unmodeled_count("memory access with unrecoverable addresses"), 0u);
  EXPECT_EQ(unknown_host.unmodeled_count("memory access with unrecoverable addresses"), 1u);
}

/// @brief A barrier costs the spread between the wavefronts that reach it, and
///        this model has no per-wavefront timeline to take a spread of. Nothing
///        is charged, so the ledger entry is the only honest record of it.
TEST_F(TimingLeakyModelTest, BarrierIsDeclaredRatherThanCharged) {
  LeakyBucketModel model(host);
  const DispatchKey key{.dispatch_id = 1, .queue_id = 0};
  const WaveRef waves[] = {wave_of(key), wave_of(key)};

  model.on_dispatch_begin(dispatch_of(key, "barrier"));
  model.on_barrier(waves);
  model.on_dispatch_end(key);

  EXPECT_EQ(model.device_cycles(), 0u);
  EXPECT_EQ(host.unmodeled_count("barrier (no per-wavefront timeline to spread over)"), 1u);
}

/// @brief A run that stops without a completion notification still has to
///        report the work it saw, rather than dropping the dispatch on the floor.
TEST_F(TimingLeakyModelTest, FinalizeClosesADispatchThatNeverEnded) {
  host.set_class_issue_cycles(InstClass::VectorAlu, 4);
  LeakyBucketModel model(host);

  const DispatchKey key{.dispatch_id = 9, .queue_id = 2};
  model.on_dispatch_begin(dispatch_of(key, "abandoned"));
  for (int i = 0; i < 10; ++i)
    model.on_instruction(wave_of(key), event_of(InstClass::VectorAlu));

  EXPECT_EQ(model.device_cycles(), 0u) << "nothing is charged until the dispatch closes";
  std::string before;
  model.write_report(before);
  EXPECT_TRUE(reported_cycles(before).empty());

  model.on_finalize();

  EXPECT_EQ(model.device_cycles(), 40u);
  std::string after;
  model.write_report(after);
  ASSERT_EQ(reported_cycles(after).size(), 1u);
  EXPECT_EQ(reported_cycles(after)[0], 40u);
  EXPECT_NE(after.find("abandoned"), std::string::npos);

  // Finalising twice must not double-charge; the straggler is gone by then.
  model.on_finalize();
  EXPECT_EQ(model.device_cycles(), 40u);
}

TEST_F(TimingLeakyModelTest, ClockRateComesFromTheHost) {
  host.set_clock_ghz(2.1);
  LeakyBucketModel model(host);
  EXPECT_DOUBLE_EQ(model.clock_ghz(), 2.1);
  EXPECT_EQ(model.name(), "leaky");
}

/// @brief A dispatch drains at the width of its workgroups, not of the part.
///
/// @details One workgroup does not run on two compute units, so a grid smaller
/// than the machine cannot use all of it. Without this bound a four-workgroup
/// grid divides by every compute unit the config declares and comes out faster
/// than a single workgroup's work can possibly be retired — which reads as a
/// very fast kernel rather than as a modelling error.
TEST_F(TimingLeakyModelTest, ADispatchDrainsAtItsOwnWidthNotThePartsWidth) {
  host.set_int("compute_units", 256);
  LeakyBucketModel model(host);

  Buckets narrow;
  narrow.unit_cycles[static_cast<std::size_t>(FunctionalUnit::VectorAlu)] = 1024;
  narrow.workgroups = 4;

  Buckets wide = narrow;
  wide.workgroups = 256;

  // Same work, different width: four workgroups get four compute units.
  EXPECT_EQ(model.drain_cycles(narrow), 256u);
  EXPECT_EQ(model.drain_cycles(wide), 4u);

  // An unannounced shape is not a shape of zero. It falls back to the whole
  // part rather than dividing by nothing.
  Buckets unannounced = narrow;
  unannounced.workgroups = 0;
  EXPECT_EQ(model.drain_cycles(unannounced), 4u);
}

} // namespace
} // namespace rocjitsu::timing::leaky
