/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_SDMA_PKT_BUILDERS_H_
#define HSA_RUNTIME_CORE_INC_SDMA_PKT_BUILDERS_H_

#include <cstddef>
#include <cstdint>

// Byte-layout builders for the SDMA FENCE and TRAP packets. These are the single
// source of truth shared by BlitSdma (copy/fill/barrier path) and the native
// SDMA user queue (progress-fence emission), so the two cannot drift.

namespace rocr {
namespace AMD {

/// @brief Write an SDMA FENCE packet that stores @p fence_value to @p fence_addr.
/// @details The packet format is selected by @p gfx_major_version (pre-gfx12 vs
///   gfx12). @p scope_fields mirrors the BlitSdma template flag and sets the
///   system memory scope on gfx12+.
/// @return Number of bytes written.
size_t BuildSdmaFencePacket(void* dst, uint32_t gfx_major_version, bool scope_fields,
                            void* fence_addr, uint32_t fence_value);

/// @brief Write a 64-bit SDMA FENCE packet (gfx1250) storing @p fence_value.
/// @return Number of bytes written.
size_t BuildSdmaFence64bPacket(void* dst, bool scope_fields, void* fence_addr,
                               uint64_t fence_value);

/// @brief Write an SDMA TRAP packet carrying interrupt context @p event_id.
/// @return Number of bytes written.
size_t BuildSdmaTrapPacket(void* dst, uint32_t event_id);

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_SDMA_PKT_BUILDERS_H_
