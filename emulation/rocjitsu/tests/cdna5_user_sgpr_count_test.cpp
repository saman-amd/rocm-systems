// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

namespace rocjitsu::test::cdna5 {
namespace {

using namespace ::rocjitsu;

// gfx1250 widens COMPUTE_PGM_RSRC2.USER_SGPR_COUNT to six bits at [6:1], so a
// declared count of 32 must survive an encode/decode round trip.
TEST(Gfx1250SimulationTest, UserSgprCount32RoundTrips) {
  // gfx1250 spends bit 6 on the count, so 32 occupies the full six-bit field.
  rocr::llvm::amdhsa::kernel_descriptor_t kd{};
  set_descriptor_user_sgpr_count(ROCJITSU_CODE_ARCH_CDNA5, kd, 32u);
  EXPECT_EQ(descriptor_user_sgpr_count(ROCJITSU_CODE_ARCH_CDNA5, kd), 32u);
  EXPECT_EQ(kd.compute_pgm_rsrc2, 32u << 1);

  // Pre-gfx1250 targets keep the five-bit field, whose maximum is 31, and must
  // not disturb ENABLE_TRAP_HANDLER at bit 6.
  rocr::llvm::amdhsa::kernel_descriptor_t legacy{};
  set_descriptor_user_sgpr_count(ROCJITSU_CODE_ARCH_RDNA4, legacy, 31u);
  EXPECT_EQ(descriptor_user_sgpr_count(ROCJITSU_CODE_ARCH_RDNA4, legacy), 31u);
  EXPECT_EQ(legacy.compute_pgm_rsrc2 & rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_TRAP_HANDLER,
            0u);
}

// A kernel declaring 32 user SGPRs (kernarg pointer + 30 preloaded dwords) must
// dispatch and see its preloaded kernarg values in s[2:31].
TEST(Gfx1250SimulationTest, KernargPreload30DwordsWith32UserSgprs) {
  using namespace rocr::llvm::amdhsa;
  Gfx1250Sim sim;

  constexpr uint32_t kPreloadLength = 30;
  constexpr uint32_t kUserSgprs = 32;
  constexpr uint64_t kKernargAddr = 0x40000;
  std::array<uint32_t, kPreloadLength> kernargs{};
  for (uint32_t i = 0; i < kPreloadLength; ++i)
    kernargs[i] = 0xA5A50000u + i;
  sim.memory->load_image(reinterpret_cast<const uint8_t *>(kernargs.data()),
                         kernargs.size() * sizeof(uint32_t), kKernargAddr);

  const std::array<uint32_t, 1> code{S_ENDPGM_GFX12};
  const uint64_t kernel_object = sim.write_kernel(
      0x10000, code.data(), code.size(), /*sgprs=*/104, /*vgprs=*/32, kUserSgprs,
      /*enable_wg_id_x=*/false, /*enable_wg_id_y=*/false, /*enable_wg_id_z=*/false,
      /*kernel_code_properties=*/KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
          KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
      /*kernarg_size=*/kPreloadLength * 4, kPreloadLength, /*kernarg_preload_offset=*/0);

  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32, kKernargAddr);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_FALSE(sim.snapshot->snapshots().empty()) << "kernel did not dispatch/halt";
  const auto &snap = sim.snapshot->snapshots().back();
  for (uint32_t i = 0; i < kPreloadLength; ++i)
    EXPECT_EQ(snap.sgprs[2 + i], 0xA5A50000u + i) << "preloaded kernarg dword " << i;
}

} // namespace
} // namespace rocjitsu::test::cdna5
