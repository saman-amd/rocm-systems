/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-gather device kernel for the DDA fabric path (gfx1250).
 * A warp owns one 2 KiB slice: eight uint64 registers per lane, with the flag
 * lanes' odd registers carrying the per-line flag so the payload stays dense and
 * coalesced in the user buffer (see ll128_pack.h). No GPU barrier; staging uses
 * comm->ddaScratch reached via comm->ddaPeerPtrsDev.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "algorithms/CollCommon.h"
#include "algorithms/ll128_pack.h"

#ifndef RCCL_LL128_AG_LOCAL_COPY_NT
#define RCCL_LL128_AG_LOCAL_COPY_NT 1
#endif

namespace meta::comms {

// Slot geometry is derived from the scratch allocation rather than pinned to a
// compile-time budget, so the reach grows with the buffer instead of capping the
// all-gather well below the LL128 size threshold. Scratch holds 2 banks of
// nRanks slots, and both inputs are identical on every rank of the comm, so all
// ranks agree on the layout without exchanging it.
constexpr size_t ddaLL128AgSlotSlices(int nRanks, size_t scratchBytes) {
  return nRanks < 1 ? 0
                    : scratchBytes / ((size_t)2 * (size_t)nRanks * (size_t)ll128::kWireBytesPerSlice);
}

// Slot stride in 8B words. A whole number of slices, so also a whole number of
// 128B lines.
constexpr size_t ddaLL128AgSlotWords(int nRanks, size_t scratchBytes) {
  return ddaLL128AgSlotSlices(nRanks, scratchBytes) * (size_t)ll128::kWireWordsPerSlice;
}

// Payload the slot carries. Whole slices only: a partial trailing slice would
// run past the slot into the next rank's.
constexpr size_t ddaLL128AgMaxPerRankBytes(int nRanks, size_t scratchBytes) {
  return ddaLL128AgSlotSlices(nRanks, scratchBytes) * (size_t)ll128::kDataBytesPerSlice;
}

__device__ __forceinline__ uint32_t ddaLL128AgEpochBegin(const uint32_t* __restrict__ epochDev, int flatBlockId) {
  uint32_t f = epochDev[flatBlockId] + 1u;
  if (f == 0u) f = 2u; // skip 0 sentinel; keep bank parity
  return f;
}

__device__ __forceinline__ void ddaLL128AgEpochEnd(uint32_t* __restrict__ epochDev, int flatBlockId, int total,
                                                   int epochLen, uint32_t flag) {
  for (int e = flatBlockId + (int)threadIdx.x * total; e < epochLen; e += total * (int)blockDim.x) {
    epochDev[e] = flag;
  }
}

// LL128 all-gather. 2D grid: grid.x == nRanks - 1 places one column per remote
// peer and none for self; grid.y splits that peer's slices across blocks, one
// warp per slice. Each column packs this rank's payload into its peer's slot,
// then polls its own slot for that peer's payload and unpacks it. The local
// sendbuff -> recvbuff[self] copy is spread over the whole grid between the two
// phases.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllGatherFabricLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                          T* __restrict__ recvbuff, // local user output
                                          const T* __restrict__ sendbuff, // local user input
                                          size_t perRankBytes, // per-rank payload; multiple of 16
                                          int selfRank, int nRanksRt,
                                          uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                          int epochLen, // number of cells in epochDev
                                          size_t slicesTotal, // slices this call uses
                                          size_t slotWords) { // per-rank slot stride, in 8B words

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  // XOR keeps the pairing index-symmetric -- rank r's column c owns peer p
  // exactly when p's column c owns r -- so both halves of a pair are dispatched
  // in the same wave, which matters because a phase-2 poll waits on one specific
  // block of the peer's grid. XOR enumerates the peers exactly only for
  // power-of-two rank counts; the rotation fallback is correct but not symmetric.
  const int col = (int)blockIdx.x;
  const int peer =
    ((nRanks & (nRanks - 1)) == 0) ? (selfRank ^ (col + 1)) : ((selfRank + 1 + col) % nRanks);

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % ll128::kWarp;
  const int warp = tid / ll128::kWarp;
  const int nwarps = nthreads / ll128::kWarp;
  const bool flagLane = ll128::isFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaLL128AgEpochBegin(epochDev, flatBlockId);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)nRanks * (uint64_t)slotWords;

  // Slices stride by warp within this peer's column only.
  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords +
    (uint64_t)selfRank * (uint64_t)slotWords;
  const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
    (uint64_t)peer * (uint64_t)slotWords;
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;

  // Phase 1: pack and push this column's slices to the one peer it owns.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)ll128::kDataBytesPerSlice;
    const size_t rem = perRankBytes - dataByte;
    const int eltInSlice =
      rem < (size_t)ll128::kDataBytesPerSlice ? (int)rem : ll128::kDataBytesPerSlice;
    uint64_t regs[ll128::kWordsPerThread];
    ll128::loadRegs<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);
    ll128::storeWire(
      scatterSlot + s * (size_t)ll128::kWireWordsPerSlice + 2 * lane, regs, flag, flagLane);
  }

  // Local copy sendbuff -> recvbuff[selfRank]. With no self column this is spread
  // over the whole grid, and it sits between the phases deliberately: after phase
  // 1 so it never delays the stores peers are waiting on, and before phase 2 so
  // it fills the fabric latency the poll would otherwise spend spinning.
  {
    v4u_gptr s4 = (v4u_gptr)sendbuff;
    v4u_gptr d4 = (v4u_gptr)(reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes);
    const size_t nVec = perRankBytes >> 4; // 16B chunks
    const size_t gtid = (size_t)flatBlockId * (size_t)nthreads + (size_t)tid;
    const size_t stride = (size_t)total * (size_t)nthreads;
    for (size_t i = gtid; i < nVec; i += stride) {
#if RCCL_LL128_AG_LOCAL_COPY_NT
      __builtin_nontemporal_store(__builtin_nontemporal_load(s4 + i), d4 + i);
#else
      d4[i] = s4[i];
#endif
    }
  }

  // Phase 2: poll the same slices in that peer's slot and unpack.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)ll128::kDataBytesPerSlice;
    const size_t rem = perRankBytes - dataByte;
    const int eltInSlice =
      rem < (size_t)ll128::kDataBytesPerSlice ? (int)rem : ll128::kDataBytesPerSlice;
    uint64_t vr[ll128::kWordsPerThread];
    ll128::pollWire(gatherSlot + s * (size_t)ll128::kWireWordsPerSlice + 2 * lane, vr, flag, lane);
    ll128::storeRegs<int8_t>(dstBytes + dataByte, vr, eltInSlice, lane, flagLane);
  }

  // Load-bearing: no thread may advance a cell until every thread in the block
  // has read its own, and until the polls above have landed.
  __syncthreads();
  // asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0");
  //__amd_builtin_amdgcn_s_wait_storecnt(0);
  ddaLL128AgEpochEnd(epochDev, flatBlockId, total, epochLen, flag32);
}

} // namespace meta::comms
