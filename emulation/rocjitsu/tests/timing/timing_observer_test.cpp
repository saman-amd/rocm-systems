// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_observer_test.cpp
/// @brief The parts of TimingObserver that do not need a running simulator.
///
/// @details Most of the observer is instruction decoding, and testing that
/// needs wavefronts. What is left is the bookkeeping around the edges — the
/// clock adapter, the terminal call, and the dispatch key — and all three are
/// reachable from the infrequent hooks alone, which take plain structs. They
/// are also the parts where a defect is silent: a finalize that runs twice
/// double-reports, a finalize that never runs reports nothing at all, and a
/// dispatch key missing its queue half merges two unrelated kernels into one
/// row on a multi-XCD part. None of those announce themselves in a number.

#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/timing/observer.h"
#include "timing/mock_timing_model.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace rocjitsu::timing {
namespace {

using Callback = test::RecordingModel::Callback;

/// @brief Total count of ledger entries whose label contains @p needle.
///
/// @details Matched on a fragment rather than the full label so a reworded
/// diagnostic does not fail a test about behaviour.
std::uint64_t declared(const test::MockTimingHost &host, const std::string &needle) {
  std::uint64_t total = 0;
  for (const auto &[effect, count] : host.unmodeled())
    if (effect.find(needle) != std::string::npos)
      total += count;
  return total;
}

rocjitsu::KernelDispatchInfo packet(std::uint32_t dispatch_id, std::uint32_t queue_id,
                                    std::string name) {
  rocjitsu::KernelDispatchInfo info;
  info.dispatch_id = dispatch_id;
  info.queue_id = queue_id;
  info.kernel_name = std::move(name);
  info.grid_size_x = 1024;
  info.grid_size_y = 2;
  info.grid_size_z = 3;
  info.workgroup_size_x = 256;
  info.workgroup_size_y = 1;
  info.workgroup_size_z = 1;
  info.workgroup_count = 4;
  info.wfs_per_workgroup = 4;
  info.sgprs_per_wf = 32;
  info.vgprs_per_wf = 64;
  info.lds_bytes_per_workgroup = 8192;
  info.wave_size = 64;
  return info;
}

/// @brief Declare the compute-unit count, so the observer's placement gap does
///        not appear in tests that are not about it. A MockTimingHost holds a
///        mutex and cannot be returned by value, hence the out-parameter.
void declare_machine(test::MockTimingHost &host) { host.set_int("compute_units", 256); }

TEST(TimingObserverTest, TimeSourceAdapterTracksTheModel) {
  test::MockTimingHost host;
  host.set_int("compute_units", 256);
  test::FixedCostModel model(/*cycles_per_instruction=*/5, /*clock_ghz=*/2.0);
  TimingObserver observer(model, host);

  TimeSource *source = observer.time_source();
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source->current_cycles(), 0u);
  EXPECT_DOUBLE_EQ(source->clock_ghz(), 2.0);

  const WaveRef wave;
  const InstructionEvent event;
  for (int i = 0; i < 10; ++i)
    model.on_instruction(wave, event);

  EXPECT_EQ(source->current_cycles(), model.device_cycles());
  EXPECT_EQ(source->current_cycles(), 50u);
}

/// @brief The terminal call must happen exactly once.
///
/// @details Both the destructor and onShutdown() reach for it, and both have to
/// exist: a local run under the interposer never tears the VM down, so
/// onShutdown() may never fire, while a run that does shut down would otherwise
/// close the report only when the object happened to be freed. Running it twice
/// would let a model close and re-close, which for anything accumulating a
/// report means printing it twice.
TEST(TimingObserverTest, FinalizeRunsExactlyOnceWhenShutdownAndDestructionBothFire) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  {
    TimingObserver observer(model, host);
    observer.onShutdown();
    EXPECT_EQ(model.count(Callback::Finalize), 1u);
    observer.onShutdown();
    EXPECT_EQ(model.count(Callback::Finalize), 1u) << "a second shutdown must not re-close";
  }
  EXPECT_EQ(model.count(Callback::Finalize), 1u) << "and neither must destruction after one";
}

TEST(TimingObserverTest, FinalizeRunsOnDestructionWhenShutdownNeverFires) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  {
    TimingObserver observer(model, host);
    EXPECT_EQ(model.count(Callback::Finalize), 0u);
  }
  EXPECT_EQ(model.count(Callback::Finalize), 1u);
}

/// @brief The completion hook carries only a dispatch id, so the queue half of
///        the key has to be remembered from the packet that opened it.
///
/// @details Losing it is not a crash: the dispatch closes on queue zero, which
/// on a multi-XCD part is some other command processor's dispatch, and the two
/// merge into one report row with their wavefronts mixed together.
TEST(TimingObserverTest, DispatchIdMapsBackToItsQueueIdAtCompletion) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  TimingObserver observer(model, host);

  observer.onAmdgpuDispatchPacketProcessed(packet(7, 3, "kernel_on_queue_three"));
  observer.onAmdgpuDispatchExecutionEnd(7);

  const std::vector<test::RecordingModel::Call> calls = model.calls();
  ASSERT_EQ(calls.size(), 2u);
  ASSERT_EQ(calls[0].kind, Callback::DispatchBegin);
  EXPECT_EQ(calls[0].dispatch.key.dispatch_id, 7u);
  EXPECT_EQ(calls[0].dispatch.key.queue_id, 3u);
  EXPECT_EQ(calls[0].dispatch.kernel_name, "kernel_on_queue_three");

  ASSERT_EQ(calls[1].kind, Callback::DispatchEnd);
  EXPECT_EQ(calls[1].key.dispatch_id, 7u);
  EXPECT_EQ(calls[1].key.queue_id, 3u) << "the queue half must survive to completion";
  EXPECT_EQ(declared(host, "no recoverable queue id"), 0u);
}

/// @brief Two dispatches sharing an id on different queues stay two dispatches.
TEST(TimingObserverTest, CollidingDispatchIdsOnDifferentQueuesStaySeparate) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  TimingObserver observer(model, host);

  observer.onAmdgpuDispatchPacketProcessed(packet(7, 0, "left"));
  observer.onAmdgpuDispatchPacketProcessed(packet(7, 1, "right"));

  const std::vector<test::RecordingModel::Call> calls = model.calls();
  ASSERT_EQ(calls.size(), 2u);
  EXPECT_EQ(calls[0].dispatch.key.queue_id, 0u);
  EXPECT_EQ(calls[1].dispatch.key.queue_id, 1u);
  EXPECT_NE(calls[0].dispatch.key, calls[1].dispatch.key);
}

TEST(TimingObserverTest, ARepeatedPacketAnnouncesItsDispatchOnlyOnce) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  TimingObserver observer(model, host);

  observer.onAmdgpuDispatchPacketProcessed(packet(2, 1, "k"));
  observer.onAmdgpuDispatchPacketProcessed(packet(2, 1, "k"));
  EXPECT_EQ(model.count(Callback::DispatchBegin), 1u);
}

/// @brief A completion for a dispatch nobody announced is still delivered — a
///        model that never hears an end holds the dispatch open forever — but
///        the guessed queue is declared rather than passed off as known.
/// @brief A completion for a dispatch nobody announced is dropped, not guessed.
///
/// @details The completion tracker retires every queue entry it holds, and only
/// some of them are kernel dispatches the observer announced. Forwarding one of
/// the others would have to invent the queue half of the key, and (0, id) is a
/// real dispatch on a multi-XCD part — the model would close a different
/// kernel. Nothing is open for an unannounced dispatch, so dropping it leaves
/// nothing dangling, and the count is what would reveal real dispatches going
/// unannounced.
TEST(TimingObserverTest, ACompletionForAnUnannouncedDispatchIsNotForwarded) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  TimingObserver observer(model, host);

  observer.onAmdgpuDispatchExecutionEnd(99);

  EXPECT_EQ(model.count(Callback::DispatchEnd), 0u);
  EXPECT_EQ(declared(host, "never announced"), 1u);
}

/// @brief The mapping is consumed by the completion it belongs to, so a
///        dispatch id reused later cannot inherit the old queue.
TEST(TimingObserverTest, TheQueueMappingIsForgottenOnceItsDispatchCompletes) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  TimingObserver observer(model, host);

  observer.onAmdgpuDispatchPacketProcessed(packet(5, 2, "k"));
  observer.onAmdgpuDispatchExecutionEnd(5);
  EXPECT_EQ(model.count(Callback::DispatchEnd), 1u);
  EXPECT_EQ(declared(host, "never announced"), 0u);

  // The second completion finds no mapping, so it is a completion for a
  // dispatch this observer no longer knows about and is dropped rather than
  // forwarded onto queue zero.
  observer.onAmdgpuDispatchExecutionEnd(5);
  EXPECT_EQ(model.count(Callback::DispatchEnd), 1u);
  EXPECT_EQ(declared(host, "never announced"), 1u);
}

TEST(TimingObserverTest, DispatchShapeReachesTheModelIntact) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  TimingObserver observer(model, host);

  observer.onAmdgpuDispatchPacketProcessed(packet(1, 0, "shaped"));

  ASSERT_EQ(model.count(Callback::DispatchBegin), 1u);
  const DispatchInfo &info = model.calls().front().dispatch;
  EXPECT_EQ(info.grid_size[0], 1024u);
  EXPECT_EQ(info.grid_size[1], 2u);
  EXPECT_EQ(info.grid_size[2], 3u);
  EXPECT_EQ(info.workgroup_size[0], 256u);
  EXPECT_EQ(info.workgroup_count, 4u);
  EXPECT_EQ(info.waves_per_workgroup, 4u);
  EXPECT_EQ(info.vector_registers_per_wave, 64u);
  EXPECT_EQ(info.scalar_registers_per_wave, 32u);
  EXPECT_EQ(info.lds_bytes_per_workgroup, 8192u);
  EXPECT_EQ(info.wave_size, 64u);
}

/// @brief A nameless kernel still lands in a named row rather than a blank one.
TEST(TimingObserverTest, ANamelessKernelIsAttributedRatherThanDropped) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  TimingObserver observer(model, host);

  observer.onAmdgpuDispatchPacketProcessed(packet(1, 0, ""));

  ASSERT_EQ(model.count(Callback::DispatchBegin), 1u);
  EXPECT_EQ(model.calls().front().dispatch.kernel_name, rocjitsu::kUnknownKernelIdentity);
}

/// @brief A config that does not say how many compute units the part gives a
///        dispatch leaves the grid crammed onto the units the emulator actually
///        uses, which over-states contention. That is the pessimistic reading
///        and it is the one taken, but it goes on the ledger either way.
TEST(TimingObserverTest, UndeclaredComputeUnitsAreDeclaredAsAPlacementGap) {
  test::RecordingModel model;
  {
    test::MockTimingHost silent;
    TimingObserver observer(model, silent);
    EXPECT_EQ(declared(silent, "dispatch placement"), 1u);
    EXPECT_TRUE(silent.fell_back("compute_units"));
  }
  {
    test::MockTimingHost declared_host;
    declare_machine(declared_host);
    TimingObserver observer(model, declared_host);
    EXPECT_EQ(declared(declared_host, "dispatch placement"), 0u);
  }
}

/// @brief The observer must not ask the plugin group to serialize the hot
///        hooks. timing_model.h promises a model serialization per compute unit
///        and nothing more, precisely so units can be costed in parallel;
///        answering true here would put every wavefront in the device behind
///        one mutex and make the model the simulator's bottleneck.
TEST(TimingObserverTest, HotHooksAreNotSerializedByTheGroup) {
  test::MockTimingHost host;
  declare_machine(host);
  test::RecordingModel model;
  TimingObserver observer(model, host);
  EXPECT_FALSE(observer.requires_serial_hot_hooks());
}

} // namespace
} // namespace rocjitsu::timing
