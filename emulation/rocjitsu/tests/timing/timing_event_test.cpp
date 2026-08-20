// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_event_test.cpp
/// @brief Invariants of the event vocabulary itself.
///
/// @details These are the properties every model is entitled to assume and no
/// model can restore once they are broken. Most of them are about defaults:
/// the event structs are aggregates that observers fill in field by field, so
/// whatever a field means when nobody assigned it is what a partially written
/// event costs. The fail-slow rule therefore has to be encoded in the zero
/// values, and that is only true by construction — reordering an enum or
/// changing an initialiser would silently invert it, which is what these tests
/// exist to catch.

#include "rocjitsu/vm/timing/event.h"
#include "rocjitsu/vm/timing/inst_class.h"
#include "rocjitsu/vm/timing/timing_model.h"
#include "timing/mock_timing_model.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

namespace rocjitsu::timing {
namespace {

/// @brief The zero value of InstClass must be the expensive one.
///
/// @details A default-constructed StaticInstInfo or InstructionEvent is what an
/// observer produces when it has not classified an opcode, and what memset-like
/// initialisation produces everywhere else. If Unknown were not first, that
/// state would be VectorAlu — a cheap, entirely plausible class — and a
/// coverage gap would be indistinguishable from a fast kernel.
TEST(TimingEventTest, UnknownIsTheZeroValueSoDefaultsAreExpensive) {
  EXPECT_EQ(static_cast<int>(InstClass::Unknown), 0);

  const StaticInstInfo info{};
  EXPECT_EQ(info.inst_class, InstClass::Unknown);

  const InstructionEvent event{};
  EXPECT_EQ(event.effective_class, InstClass::Unknown);

  // The same must hold for a value-initialised array of classes, which is how a
  // per-class cost table starts life.
  const std::array<InstClass, 4> table{};
  for (InstClass cls : table)
    EXPECT_EQ(cls, InstClass::Unknown);
}

/// @brief An unclassified opcode must contend for a port that is already busy.
///
/// @details Mapping Unknown to FunctionalUnit::None would let it drain against
/// an idle unit and cost nothing in a throughput model, which is the same
/// optimistic bias as charging it zero cycles, arriving by a different route.
TEST(TimingEventTest, UnknownContendsForARealUnit) {
  EXPECT_EQ(unit_for_class(InstClass::Unknown), FunctionalUnit::VectorAlu);
  EXPECT_NE(unit_for_class(InstClass::Unknown), FunctionalUnit::None);
}

/// @brief Only the classes that genuinely occupy no issue port map to None.
TEST(TimingEventTest, OnlyNonIssuingClassesMapToNoUnit) {
  const std::set<InstClass> non_issuing = {
      InstClass::WaitCounter, InstClass::DelayAlu, InstClass::Barrier,
      InstClass::Message,     InstClass::Nop,      InstClass::Terminate,
  };
  for (std::size_t i = 0; i < kNumInstClasses; ++i) {
    const auto cls = static_cast<InstClass>(i);
    const bool none = unit_for_class(cls) == FunctionalUnit::None;
    EXPECT_EQ(none, non_issuing.count(cls) == 1) << "class " << inst_class_name(cls);
  }
}

/// @brief Class names are the config file's vocabulary, so they must be unique:
///        two classes sharing a name would make one of them unconfigurable.
TEST(TimingEventTest, ClassNamesAreDistinct) {
  std::set<std::string> names;
  for (std::size_t i = 0; i < kNumInstClasses; ++i)
    EXPECT_TRUE(names.insert(inst_class_name(static_cast<InstClass>(i))).second)
        << "duplicate class name at index " << i;
  EXPECT_EQ(names.size(), kNumInstClasses);
}

TEST(TimingEventTest, MemoryClassesAreExactlyTheOnesThatPostToACounter) {
  const std::set<InstClass> memory = {
      InstClass::LdsRead,
      InstClass::LdsWrite,
      InstClass::VectorMemoryRead,
      InstClass::VectorMemoryWrite,
      InstClass::VectorMemoryAtomic,
      InstClass::ScalarMemory,
      InstClass::TensorMemory,
      InstClass::Export,
  };
  for (std::size_t i = 0; i < kNumInstClasses; ++i) {
    const auto cls = static_cast<InstClass>(i);
    EXPECT_EQ(class_is_memory(cls), memory.count(cls) == 1) << "class " << inst_class_name(cls);
  }
}

TEST(TimingEventTest, WaitThresholdsDefaultToUnconstrained) {
  const WaitThresholds wait;
  for (std::size_t i = 0; i < kNumWaitCounters; ++i)
    EXPECT_EQ(wait.get(static_cast<WaitCounter>(i)), WaitThresholds::kUnconstrained);
  EXPECT_FALSE(wait.constrains_anything());
}

TEST(TimingEventTest, WaitThresholdsRoundTripAndConstrainOnlyWhatWasSet) {
  WaitThresholds wait;
  wait.set(WaitCounter::VectorLoad, 0);
  EXPECT_EQ(wait.get(WaitCounter::VectorLoad), 0u);
  EXPECT_TRUE(wait.constrains_anything());

  // A threshold of zero is a full wait, not an absent one, so every other
  // counter must still read as unconstrained.
  EXPECT_EQ(wait.get(WaitCounter::LgkmCombined), WaitThresholds::kUnconstrained);

  wait.set(WaitCounter::LgkmCombined, 3);
  EXPECT_EQ(wait.get(WaitCounter::LgkmCombined), 3u);
  EXPECT_EQ(wait.get(WaitCounter::VectorLoad), 0u);
}

/// @brief A default MemoryAccess is not an access at all.
///
/// @details addresses_known defaults to true, which would be the optimistic
/// default if it stood alone; it is safe only because space defaults to None
/// and no model looks at the address fields of an access that reports itself
/// invalid. Pinning both here keeps that pairing from drifting apart.
TEST(TimingEventTest, DefaultMemoryAccessIsNotAnAccess) {
  const MemoryAccess access;
  EXPECT_EQ(access.space, MemorySpace::None);
  EXPECT_FALSE(access.valid());
  EXPECT_TRUE(access.lane_addresses.empty());

  // No counter reported: a model derives one from the class rather than
  // parking the completion on a counter the target may not have.
  EXPECT_EQ(access.wait_counter, WaitCounter::Count);

  const InstructionEvent event{};
  EXPECT_FALSE(event.memory.valid());
}

/// @brief Dispatch ids are allocated per command processor, so they collide
///        across the XCDs of a multi-die part. The queue half is what keeps two
///        unrelated kernels out of one report entry.
TEST(TimingEventTest, DispatchKeyDistinguishesQueues) {
  const DispatchKey a{.dispatch_id = 7, .queue_id = 0};
  const DispatchKey b{.dispatch_id = 7, .queue_id = 1};
  const DispatchKey c{.dispatch_id = 7, .queue_id = 0};

  const DispatchKey other_id{.dispatch_id = 8, .queue_id = 0};

  EXPECT_NE(a, b);
  EXPECT_EQ(a, c);
  EXPECT_NE(other_id, a);
}

TEST(TimingEventTest, WaveRefEqualityFollowsItsDispatchKey) {
  WaveRef wave;
  wave.dispatch = {.dispatch_id = 3, .queue_id = 0};
  wave.workgroup_id = 1;
  wave.wave_slot = 2;
  wave.compute_unit_id = 4;

  WaveRef other = wave;
  EXPECT_EQ(wave, other);

  other.dispatch.queue_id = 1;
  EXPECT_NE(wave, other);

  other = wave;
  other.compute_unit_id = 5;
  EXPECT_NE(wave, other) << "the resident compute unit is part of a wavefront's identity";
}

/// @brief A default wavefront is a full-width one, so a model that costs by
///        wave width never bills zero for an event whose shape was not filled in.
TEST(TimingEventTest, DefaultWaveShapeIsFullWidth) {
  const InstructionEvent event{};
  EXPECT_EQ(event.wave_lanes, 64u);
  EXPECT_EQ(event.active_lanes, 0u) << "no lane count reported; a model falls back to wave_lanes";

  const WaveRef wave;
  EXPECT_EQ(wave.wave_lanes, 64u);
}

/// @brief A model that declares no Interest gets no optional payloads built.
///
/// @details The default has to be "nothing", because filling lane addresses and
/// operand ranges dominates the observer's per-instruction cost and a
/// throughput model needs neither. What makes that safe is the other half of
/// the contract, exercised in the leaky-model tests: a field a model did not
/// ask for arrives empty rather than absent, and an empty field is charged as
/// the expensive case.
TEST(TimingModelTest, DefaultInterestRequestsNothing) {
  const TimingModel::Interest interest;
  EXPECT_FALSE(interest.lane_addresses);
  EXPECT_FALSE(interest.register_ranges);

  const test::FixedCostModel model;
  EXPECT_FALSE(model.interest().lane_addresses);
  EXPECT_FALSE(model.interest().register_ranges);
}

/// @brief The observation contract, asserted against the recording double.
///
/// @details Call order for one wavefront is on_wave_begin, then on_instruction
/// once per executed instruction in program order, then on_wave_end, with the
/// dispatch callbacks bracketing the wavefronts that belong to the dispatch.
/// Driving that by hand here is what lets a model's own tests assume it.
TEST(TimingModelTest, RecordingModelPreservesOrderAndIdentity) {
  using Callback = test::RecordingModel::Callback;

  StaticInstInfo info;
  info.inst_class = InstClass::VectorAlu;
  info.mnemonic = "v_add_f32";

  const DispatchKey key{.dispatch_id = 4, .queue_id = 1};
  DispatchInfo dispatch;
  dispatch.key = key;
  dispatch.kernel_name = "k";

  WaveRef first;
  first.dispatch = key;
  first.wave_slot = 0;
  WaveRef second = first;
  second.wave_slot = 1;

  test::RecordingModel model;
  model.on_dispatch_begin(dispatch);
  for (const WaveRef &wave : {first, second}) {
    model.on_wave_begin(wave);
    for (std::uint64_t pc : {0x100u, 0x108u}) {
      InstructionEvent event;
      event.pc = pc;
      event.info = &info;
      event.effective_class = info.inst_class;
      model.on_instruction(wave, event);
    }
  }
  model.on_barrier(std::array<WaveRef, 2>{first, second});
  model.on_wave_end(first);
  model.on_wave_end(second);
  model.on_dispatch_end(key);
  model.on_finalize();

  const std::vector<Callback> expected = {
      Callback::DispatchBegin, Callback::WaveBegin,   Callback::Instruction, Callback::Instruction,
      Callback::WaveBegin,     Callback::Instruction, Callback::Instruction, Callback::Barrier,
      Callback::WaveEnd,       Callback::WaveEnd,     Callback::DispatchEnd, Callback::Finalize,
  };
  EXPECT_EQ(model.sequence(), expected);
  EXPECT_EQ(model.count(Callback::WaveBegin), 2u) << "exactly one per wavefront";
  EXPECT_EQ(model.count(Callback::WaveBegin), model.count(Callback::WaveEnd));

  const std::vector<test::RecordingModel::Call> calls = model.calls();
  EXPECT_EQ(calls.front().dispatch.key, key);
  EXPECT_EQ(calls[1].wave, first);
  EXPECT_EQ(calls[2].event.pc, 0x100u);
  EXPECT_EQ(calls[2].event.info, &info) << "static info is borrowed, not copied";
  EXPECT_EQ(calls[3].event.pc, 0x108u);
  EXPECT_EQ(calls[4].wave, second);
  ASSERT_EQ(calls[7].barrier_waves.size(), 2u);
  EXPECT_EQ(calls[7].barrier_waves[1], second);
  EXPECT_EQ(calls[10].key, key);
}

/// @brief The host double must resolve an unnamed class the way TimingConfig
///        does, or every model test that depends on fail-slow proves nothing.
TEST(TimingHostTest, MockHostMirrorsTheFailSlowResolution) {
  test::MockTimingHost host;
  host.set_class_issue_cycles(InstClass::VectorAlu, 2);
  host.set_class_issue_cycles(InstClass::MatrixMultiply, 64);

  EXPECT_EQ(host.class_issue_cycles(InstClass::VectorAlu), 2u);
  EXPECT_EQ(host.class_issue_cycles(InstClass::Unknown), 64u);
  EXPECT_EQ(host.class_issue_cycles(InstClass::Branch), 64u);
  EXPECT_TRUE(host.fell_back("unknown.issue_cycles"));
  EXPECT_FALSE(host.fell_back("vector_alu.issue_cycles"));

  EXPECT_EQ(host.tune("compute_units", 1), 1u);
  EXPECT_TRUE(host.fell_back("compute_units"));

  test::MockTimingHost empty;
  EXPECT_GE(empty.class_issue_cycles(InstClass::Unknown), 1u) << "never free";
}

} // namespace
} // namespace rocjitsu::timing
