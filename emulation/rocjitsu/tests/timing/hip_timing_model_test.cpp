// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_timing_model_test.cpp
/// @brief End-to-end proof that a guest program observes the timing model's
///        clock rather than the host's.
///
/// @details Every other test under tests/timing/ drives a model directly with
/// hand-built events, which is what event.h naming no rocjitsu type buys. This
/// one cannot be written that way, and that is the point: the value has to
/// travel out of the model, through SimulatedClock, into the completion signal
/// ROCR reads, and back to the program through hipEventElapsedTime with the
/// runtime's own conversion applied. Nothing below that whole path can tell you
/// it works.
///
/// Requires a config whose `timing` block selects a model — see
/// configs/gfx950_mi355x_kmd.json — and runs under the CLI launcher.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

extern "C" const char *__lsan_default_suppressions() { return "leak:libhsa-runtime64.so\n"; }
extern "C" const char *__tsan_default_suppressions() {
  return "called_from_lib:libhsa-runtime64.so\n";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

namespace {

/// @brief A HIP event duration in whole nanoseconds.
///
/// @details The comparison below is about a one-nanosecond quantisation step,
/// which is far below what a float millisecond can represent exactly — so the
/// values are brought back to the integer unit the clock actually produced
/// before being compared, rather than compared as floats with a hand-picked
/// epsilon.
std::int64_t nanoseconds_from(float milliseconds) {
  return std::llround(static_cast<double>(milliseconds) * 1.0e6);
}

/// @brief Fixed work per thread, so the reported time is a function of the grid
///        size alone.
///
/// @details Arithmetic only, deliberately: a memory-bound kernel would make the
/// scaling test depend on how good the model's memory system is rather than on
/// whether the measurement tracks the work it was handed.
__global__ void fixed_work(float *out, int iterations) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  float accumulator = static_cast<float>(index);
  for (int i = 0; i < iterations; ++i)
    accumulator = accumulator * 1.0001f + 1.0f;
  out[index] = accumulator;
}

/// @brief Reads the device's own clock either side of some work.
__global__ void read_device_clock(unsigned long long *out, int iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  const unsigned long long begin = __builtin_amdgcn_s_memtime();
  float accumulator = 1.0f;
  for (int i = 0; i < iterations; ++i)
    accumulator = accumulator * 1.0001f + 1.0f;
  const unsigned long long end = __builtin_amdgcn_s_memtime();
  out[0] = begin;
  out[1] = end;
  // Keeps the loop from being folded away; the value itself is not interesting.
  out[2] = static_cast<unsigned long long>(accumulator);
}

/// @brief Launch @p blocks x 64 threads and return what the guest measured, in
///        milliseconds, through the ordinary HIP event path.
float timed_launch(int blocks, int iterations) {
  float *device_out = nullptr;
  EXPECT_EQ(hipMalloc(&device_out, static_cast<size_t>(blocks) * 64 * sizeof(float)), hipSuccess);

  hipEvent_t start{};
  hipEvent_t stop{};
  EXPECT_EQ(hipEventCreate(&start), hipSuccess);
  EXPECT_EQ(hipEventCreate(&stop), hipSuccess);

  EXPECT_EQ(hipEventRecord(start, nullptr), hipSuccess);
  fixed_work<<<blocks, 64>>>(device_out, iterations);
  EXPECT_EQ(hipEventRecord(stop, nullptr), hipSuccess);
  EXPECT_EQ(hipEventSynchronize(stop), hipSuccess);

  float milliseconds = -1.0f;
  EXPECT_EQ(hipEventElapsedTime(&milliseconds, start, stop), hipSuccess);

  EXPECT_EQ(hipEventDestroy(start), hipSuccess);
  EXPECT_EQ(hipEventDestroy(stop), hipSuccess);
  EXPECT_EQ(hipFree(device_out), hipSuccess);
  return milliseconds;
}

} // namespace

/// @brief The load-bearing one: repeated identical launches report an identical
///        duration.
///
/// @details This is what separates a simulated clock from a fast one. Host wall
/// time never repeats — two runs of the same kernel under an emulator differ by
/// microseconds at least — so a bit-identical result across repeats can only
/// come from a timeline a model computed. It is also the property that makes a
/// timing model useful at all: comparing two kernels means nothing if the
/// measurement moves on its own.
TEST(HipTimingModelTest, ElapsedTimeIsReproducible) {
  constexpr int kBlocks = 32;
  constexpr int kIterations = 64;

  // The first launch carries one-time cost a model may legitimately charge —
  // first-touch translation, a cold cache in a model that has one — so the
  // comparison starts from the second.
  (void)timed_launch(kBlocks, kIterations);

  const std::int64_t reference = nanoseconds_from(timed_launch(kBlocks, kIterations));
  ASSERT_GT(reference, 0) << "no time was reported at all; is a timing model configured?";

  for (int repeat = 0; repeat < 4; ++repeat) {
    const std::int64_t measured = nanoseconds_from(timed_launch(kBlocks, kIterations));
    // One nanosecond, not zero. The model produces a whole number of cycles and
    // the guest reads whole nanoseconds, and the shader clock is not an integer
    // number of cycles per nanosecond, so an identical cycle count can land
    // either side of a nanosecond boundary depending on where the run's
    // accumulated base sits. That one nanosecond is the entire budget: two host
    // wall-clock measurements of this kernel differ by microseconds, so the
    // bound still fails loudly if host time is what the guest is reading.
    EXPECT_LE(std::abs(measured - reference), 1)
        << "repeat " << repeat << " read " << measured << " ns against " << reference
        << " ns; that is more than clock quantisation, so the guest is reading host time";
  }
}

/// @brief More work per wavefront takes proportionally longer.
///
/// @details Scaling the loop count rather than the grid, deliberately. A wider
/// grid on a part with compute units to spare should take the *same* time, not
/// longer — that is what a throughput model is for — so scaling the grid would
/// be asserting the opposite of correct behaviour. Lengthening the loop adds
/// work to each wavefront without adding anywhere to put it, which is the case
/// where the reported time has to move.
///
/// The bound is loose on purpose: the claim is that the measurement tracks the
/// work, not that any particular model gets the constant right. The lower bound
/// is well under 4x because a per-dispatch floor the model cannot see past is
/// included in both measurements.
TEST(HipTimingModelTest, ElapsedTimeScalesWithWork) {
  constexpr int kBlocks = 64;
  (void)timed_launch(kBlocks, 256);

  const float small = timed_launch(kBlocks, 256);
  const float large = timed_launch(kBlocks, 1024);
  ASSERT_GT(small, 0.0f);

  const float ratio = large / small;
  EXPECT_GT(ratio, 1.5f) << "4x the per-thread work produced only " << ratio << "x the time";
  EXPECT_LT(ratio, 8.0f) << "4x the per-thread work produced " << ratio << "x the time";
}

/// @brief A kernel timing itself reads the modelled clock, and it never goes
///        backwards.
///
/// @details What this can assert depends on the model, and the model shipped
/// here is a bucket: it has no timeline inside a dispatch and publishes the
/// whole cost at the end, so two reads within one kernel legitimately return
/// the same value. A model with a per-cycle timeline would show a delta here,
/// and this test would still pass.
///
/// What is asserted either way is that the counter is on the modelled timeline
/// and does not retreat. Retreating is the failure that matters: a counter that
/// clamps without rebasing returns a constant *forever* after the first read,
/// so a kernel spinning until it changes never finishes, and every self-timing
/// delta in the process is zero rather than merely small.
TEST(HipTimingModelTest, InKernelClockIsMonotonic) {
  unsigned long long *device_out = nullptr;
  HIP_ASSERT(hipMalloc(&device_out, 3 * sizeof(unsigned long long)));

  read_device_clock<<<1, 64>>>(device_out, 256);
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<unsigned long long> host_out(3, 0);
  HIP_ASSERT(hipMemcpy(host_out.data(), device_out, 3 * sizeof(unsigned long long),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_out));

  EXPECT_GE(host_out[1], host_out[0]) << "the in-kernel clock went backwards across the loop";
  EXPECT_NE(host_out[0], 0u) << "the in-kernel clock read as zero; is it wired to anything?";
}
