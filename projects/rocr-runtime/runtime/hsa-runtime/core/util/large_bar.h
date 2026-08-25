////////////////////////////////////////////////////////////////////////////////
//
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Large BAR eligibility for device-memory queue placement.
//
// Large BAR is only meaningful when the GPU exposes device-local memory that
// the CPU can reach through a BAR, so it gates the device-memory queue
// placement flags of hsa_amd_queue_create. The node's local memory size is the
// discriminator rather than the Integrated flag: local memory size comes from
// the KFD topology and reads 0 when there is no dedicated VRAM, while
// Integrated is filled in from a libdrm query that is skipped under the
// topology model and left clear when the DRM render node cannot be opened.
// Trusting Integrated there admits a device-memory ring buffer on an APU whose
// command processor cannot fetch packets from it, which surfaces as a dispatch
// timeout instead of HSA_STATUS_ERROR_INVALID_QUEUE_CREATION.

#ifndef HSA_RUNTIME_CORE_UTIL_LARGE_BAR_H_
#define HSA_RUNTIME_CORE_UTIL_LARGE_BAR_H_

#include <cstdint>

namespace rocr {
namespace core {

// num_hop is the CPU to GPU link hop count; local_mem_size is the agent's
// device-local memory in bytes (HsaNodeProperties::LocalMemSize).
inline constexpr bool LargeBarEligible(uint32_t num_hop, uint64_t local_mem_size) {
  return num_hop >= 1 && local_mem_size > 0;
}

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_LARGE_BAR_H_
