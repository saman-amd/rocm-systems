/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-gather device kernel for the DDA fabric path (gfx1250).
 * A warp owns one 2 KiB slice: eight uint64 registers per lane. 
 * No GPU barrier; staging uses comm->ddaScratch reached via comm->ddaPeerPtrsDev.
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
#include "algorithms/CollCommon_ll128.h"

namespace meta::comms {

// LL128 all-gather. 2D grid: grid.x places one column per group of peersPerBlock
// remote peers; grid.y splits that group's slices across blocks, one warp per slice.
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
                                          size_t slotWords, // per-rank slot stride, in 8B words
                                          int peersPerBlock) { // remote peers this block serves

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int nPeers = nRanks - 1;

  // Peers [peerLo, peerLo + blockPeers) of the rotation belong to this column.
  // Only the last column can be short, and it always holds at least one peer.
  const int peerLo = (int)blockIdx.x * peersPerBlock;
  const int blockPeers = peerLo + peersPerBlock > nPeers ? nPeers - peerLo : peersPerBlock;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)nRanks * (uint64_t)slotWords;

  // Slices stride by warp within this group's column only.
  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  // Bases that hold for every peer of the group, so only the peer term varies.
  const uint64_t scatterBase = bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
  const uint64_t* gatherBank = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords;
  int8_t* dstBase = reinterpret_cast<int8_t*>(recvbuff);

  // Phase 1: pack a slice once, then push it to every peer this block serves.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = perRankBytes - dataByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread];
    ddaLL128LoadRegs<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + (size_t)(2 * lane);
    for (int j = 0; j < blockPeers; ++j) {
      const int peer = (selfRank + peerLo + j + 1) % nRanks;
      ddaLL128StoreWire(
        reinterpret_cast<uint64_t*>(peerScratch[peer]) + scatterBase + wireOff, regs, flag, flagLane);
    }
  }

  // Local copy sendbuff -> recvbuff[selfRank].
  {
    v4u_gptr s4 = (v4u_gptr)sendbuff;
    v4u_gptr d4 = (v4u_gptr)(reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes);
    const size_t nVec = perRankBytes >> 4; // 16B chunks
    const size_t gtid = (size_t)flatBlockId * (size_t)nthreads + (size_t)tid;
    const size_t stride = (size_t)total * (size_t)nthreads;
    for (size_t i = gtid; i < nVec; i += stride) {
      d4[i] = s4[i];
    }
  }

  // Phase 2: poll this block's peers for the same slices and unpack.
  const int startPeer = (int)(gwarp % (size_t)blockPeers);
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    const size_t rem = perRankBytes - dataByte;
    const int eltInSlice =
      rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
    const size_t wireOff = s * (size_t)kDdaLL128WireWordsPerSlice + (size_t)(2 * lane);
    for (int j = 0; j < blockPeers; ++j) {
      const int k = (startPeer + j) % blockPeers;
      const int peer = (selfRank + peerLo + k + 1) % nRanks;
      uint64_t vr[kDdaLL128WordsPerThread];
      ddaLL128PollWire(gatherBank + (uint64_t)peer * (uint64_t)slotWords + wireOff, vr, flag, lane);
      ddaLL128StoreRegs<int8_t>(
        dstBase + (size_t)peer * perRankBytes + dataByte, vr, eltInSlice, lane, flagLane);
    }
  }

#if defined(__gfx1250__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace meta::comms
