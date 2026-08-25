/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

// Unit tests for the pure large BAR eligibility logic (no HSA/GPU) that gates the
// device-memory queue placement flags; regression coverage for APUs with no VRAM
// being admitted and then timing out on a device-memory ring buffer dispatch.

#include <cstdint>

#include "gtest/gtest.h"

#include "core/util/large_bar.h"

using rocr::core::LargeBarEligible;

namespace {
constexpr uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
}  // namespace

// A discrete GPU: reachable over a link and exposing VRAM.
TEST(LargeBarTest, DiscreteGpuWithVramIsEligible) {
  EXPECT_TRUE(LargeBarEligible(1, 16 * kGiB));
  EXPECT_TRUE(LargeBarEligible(2, 1));
}

// An APU reports no device-local memory, so there is no BAR to map through
// even though the CPU->GPU hop count looks like a discrete link. This is the
// gfx1151 case that regressed.
TEST(LargeBarTest, ZeroLocalMemoryIsNotEligible) {
  EXPECT_FALSE(LargeBarEligible(1, 0));
  EXPECT_FALSE(LargeBarEligible(4, 0));
}

// The hop count still has to indicate a real CPU->GPU link.
TEST(LargeBarTest, NoLinkIsNotEligible) {
  EXPECT_FALSE(LargeBarEligible(0, 16 * kGiB));
  EXPECT_FALSE(LargeBarEligible(0, 0));
}

// Both conditions are required, so eligibility tracks local memory once a link
// is present and never depends on hop count alone.
TEST(LargeBarTest, RequiresLinkAndLocalMemory) {
  for (uint32_t hop = 0; hop <= 3; ++hop) {
    EXPECT_EQ(LargeBarEligible(hop, kGiB), hop >= 1);
    EXPECT_FALSE(LargeBarEligible(hop, 0));
  }
}