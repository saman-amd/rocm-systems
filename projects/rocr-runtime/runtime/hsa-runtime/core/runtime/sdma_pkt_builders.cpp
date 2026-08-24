/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/sdma_pkt_builders.h"

#include <cstring>

#include "core/inc/sdma_registers.h"

namespace rocr {
namespace AMD {

namespace {
inline uint32_t PtrLow32(const void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

inline uint32_t PtrHigh32(const void* p) {
#if defined(HSA_LARGE_MODEL)
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) >> 32);
#else
  return 0;
#endif
}
}  // namespace

size_t BuildSdmaFencePacket(void* dst, uint32_t gfx_major_version, bool scope_fields,
                            void* fence_addr, uint32_t fence_value) {
  // GFX12 or later use a different packet format that is incompatible (fields
  // changed in size and location).
  if (gfx_major_version >= 12) {
    SDMA_PKT_FENCE_GFX12* packet_addr = reinterpret_cast<SDMA_PKT_FENCE_GFX12*>(dst);

    memset(packet_addr, 0, sizeof(SDMA_PKT_FENCE_GFX12));

    packet_addr->HEADER_UNION.op = SDMA_OP_FENCE;
    packet_addr->HEADER_UNION.mtype = 3;

    /* We only use fence on signals and they are in system memory */
    packet_addr->HEADER_UNION.sys = 1;

    if (scope_fields)
      packet_addr->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

    packet_addr->ADDR_LO_UNION.addr_31_0 = PtrLow32(fence_addr);
    packet_addr->ADDR_HI_UNION.addr_63_32 = PtrHigh32(fence_addr);

    packet_addr->DATA_UNION.data = fence_value;
    return sizeof(SDMA_PKT_FENCE_GFX12);
  }

  SDMA_PKT_FENCE* packet_addr = reinterpret_cast<SDMA_PKT_FENCE*>(dst);

  memset(packet_addr, 0, sizeof(SDMA_PKT_FENCE));

  packet_addr->HEADER_UNION.op = SDMA_OP_FENCE;

  if (gfx_major_version >= 10) {
    packet_addr->HEADER_UNION.mtype = 3;
  }

  packet_addr->ADDR_LO_UNION.addr_31_0 = PtrLow32(fence_addr);
  packet_addr->ADDR_HI_UNION.addr_63_32 = PtrHigh32(fence_addr);

  packet_addr->DATA_UNION.data = fence_value;
  return sizeof(SDMA_PKT_FENCE);
}

size_t BuildSdmaFence64bPacket(void* dst, bool scope_fields, void* fence_addr,
                               uint64_t fence_value) {
  SDMA_PKT_FENCE_64B_GFX1250* pkt = reinterpret_cast<SDMA_PKT_FENCE_64B_GFX1250*>(dst);

  memset(pkt, 0, sizeof(SDMA_PKT_FENCE_64B_GFX1250));

  pkt->HEADER_UNION.op = SDMA_OP_FENCE;
  pkt->HEADER_UNION.sub_op = SDMA_SUBOP_FENCE_64B;
  pkt->HEADER_UNION.mtype = 3;
  // Signal memory is in system memory.
  pkt->HEADER_UNION.sys = 1;

  if (scope_fields)
    pkt->HEADER_UNION.scope = SDMA_MEMORY_SCOPE_SYS;

  pkt->ADDR_LO_UNION.addr_31_3 = PtrLow32(fence_addr) >> 3;
  pkt->ADDR_HI_UNION.addr_63_32 = PtrHigh32(fence_addr);

  pkt->DATA_LO_UNION.data_31_0 = static_cast<uint32_t>(fence_value);
  pkt->DATA_HI_UNION.data_63_32 = static_cast<uint32_t>(fence_value >> 32);
  return sizeof(SDMA_PKT_FENCE_64B_GFX1250);
}

size_t BuildSdmaTrapPacket(void* dst, uint32_t event_id) {
  SDMA_PKT_TRAP* packet_addr = reinterpret_cast<SDMA_PKT_TRAP*>(dst);

  memset(packet_addr, 0, sizeof(SDMA_PKT_TRAP));

  packet_addr->HEADER_UNION.op = SDMA_OP_TRAP;
  packet_addr->INT_CONTEXT_UNION.int_ctx = event_id;
  return sizeof(SDMA_PKT_TRAP);
}

}  // namespace AMD
}  // namespace rocr
