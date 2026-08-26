// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

namespace rocjitsu::test::cdna5 {
namespace {

using namespace ::rocjitsu;

// DS_ADD_U64 has no architectural destination: only DS_ADD_RTN_U64 returns the
// pre-op LDS value. The encoded VDST field must therefore be ignored.
TEST(Gfx1250SimulationTest, DsAddU64NonReturningDoesNotWriteVdst) {
  Gfx1250Sim sim;
  auto *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(256));
  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData0 = 2;
  constexpr uint32_t kDst = 6;
  constexpr uint32_t kAddress = 0x20;
  constexpr uint64_t kOld = 0x1122334455667788ull;
  constexpr uint64_t kAddend = 0x0000000000000011ull;
  constexpr uint32_t kSentinelLo = 0xDEADBEEFu;
  constexpr uint32_t kSentinelHi = 0xCAFEF00Du;
  const uint32_t vb = wf->vgpr_alloc().base;

  cu.write_vgpr(vb + kAddr, 0, kAddress);
  cu.write_vgpr(vb + kData0, 0, static_cast<uint32_t>(kAddend));
  cu.write_vgpr(vb + kData0 + 1, 0, static_cast<uint32_t>(kAddend >> 32));
  // Live, unrelated values in the VGPRs the encoded vdst field happens to name.
  cu.write_vgpr(vb + kDst, 0, kSentinelLo);
  cu.write_vgpr(vb + kDst + 1, 0, kSentinelHi);
  cu.lds().write64(wf->lds_base() + kAddress, kOld);

  ::rocjitsu::cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData0;
  raw.vdst = kDst;
  auto *inst = new ::rocjitsu::cdna5::DsAddU64Vds(
      reinterpret_cast<const ::rocjitsu::cdna5::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(state->is_load) << "non-returning DS atomic must not be a load";

  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(inst, *wf);

  EXPECT_EQ(cu.lds().read64(wf->lds_base() + kAddress), kOld + kAddend);
  EXPECT_EQ(cu.read_vgpr(vb + kDst, 0), kSentinelLo);
  EXPECT_EQ(cu.read_vgpr(vb + kDst + 1, 0), kSentinelHi);
}

// The returning form must keep working.
TEST(Gfx1250SimulationTest, DsAddRtnU64WritesVdst) {
  Gfx1250Sim sim;
  auto *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(256));
  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData0 = 2;
  constexpr uint32_t kDst = 6;
  constexpr uint32_t kAddress = 0x20;
  constexpr uint64_t kOld = 0x1122334455667788ull;
  constexpr uint64_t kAddend = 0x0000000000000011ull;
  const uint32_t vb = wf->vgpr_alloc().base;

  cu.write_vgpr(vb + kAddr, 0, kAddress);
  cu.write_vgpr(vb + kData0, 0, static_cast<uint32_t>(kAddend));
  cu.write_vgpr(vb + kData0 + 1, 0, static_cast<uint32_t>(kAddend >> 32));
  cu.lds().write64(wf->lds_base() + kAddress, kOld);

  ::rocjitsu::cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData0;
  raw.vdst = kDst;
  auto *inst = new ::rocjitsu::cdna5::DsAddRtnU64Vds(
      reinterpret_cast<const ::rocjitsu::cdna5::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->is_load);

  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(inst, *wf);

  EXPECT_EQ(cu.lds().read64(wf->lds_base() + kAddress), kOld + kAddend);
  const uint64_t returned =
      cu.read_vgpr(vb + kDst, 0) | (static_cast<uint64_t>(cu.read_vgpr(vb + kDst + 1, 0)) << 32);
  EXPECT_EQ(returned, kOld);
}

} // namespace
} // namespace rocjitsu::test::cdna5
