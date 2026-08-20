/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric all-gather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_gather.h"

#include "algorithms/all_gather/all_gather_dda_fabric_ll128.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h" // DDA_FABRIC_MAXBLOCKS
#include "debug.h"
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

// Runtime-adjustable LL128 AllGather grid shape. Threads must be a multiple of
// the wave size in [wave, 1024]; invalid values fall back to the tuned default.
// Env: RCCL_DDA_LL128_AG_THREADS, RCCL_DDA_LL128_AG_MAXBLOCKS.
RCCL_PARAM(DdaLL128AgThreads, "DDA_LL128_AG_THREADS", 512);
RCCL_PARAM(DdaLL128AgMaxBlocks, "DDA_LL128_AG_MAXBLOCKS", DDA_FABRIC_MAXBLOCKS);

namespace {

using meta::comms::ddaLL128AgMaxPerRankBytes;
using meta::comms::ddaLL128AgSlices;
using meta::comms::ddaLL128AgSlotWords;
namespace ll128 = meta::comms::ll128;

// Validated block size from the runtime flag; falls back to `dflt` if the
// configured value is not a whole number of waves in [wave, 1024].
unsigned ddaLL128AgThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128AgThreads();
  if (v >= ll128::kWarp && v <= 1024 && (v % ll128::kWarp) == 0) {
    return (unsigned)v;
  }
  return dflt;
}

// Total blocks the grid may use.
size_t ddaLL128AgBlockCap() {
  int64_t cap = rcclParamDdaLL128AgMaxBlocks();
  cap = std::min<int64_t>(std::max<int64_t>(cap, 1), DDA_FABRIC_MAXBLOCKS);
  return (size_t)cap;
}

// Blocks in one peer column: a warp per slice, capped by the column's share of
// the grid budget. Warps past the slice count own no slice. nCols is nRanks - 1,
// since LL128 has no column for the local copy.
unsigned ddaLL128AgBlocksPerPeer(size_t slices, size_t warpsPerBlock, int nCols, size_t totalBlockCap) {
  const size_t warps = warpsPerBlock < 1 ? 1 : warpsPerBlock;
  const size_t cap = totalBlockCap / (size_t)(nCols < 1 ? 1 : nCols);
  size_t bpp = (slices + warps - 1) / warps;
  if (bpp > cap) {
    bpp = cap;
  }
  return bpp < 1 ? 1u : (unsigned)bpp;
}

// Single source of the launch geometry: grid.x = one column per remote peer
// (nRanks - 1, since LL128 has no column for the local copy), grid.y = the
// per-peer slice split. The kernel indexes epochDev[flatBlockId] with
// flatBlockId up to nCols*bpp-1, so the grid is clamped to the device epoch
// array (sized for nRanks*kDdaLLAgMaxBlocksPerPeer cells) whatever the block
// cap says; that sizing always leaves room for at least one block per column.
static inline std::pair<dim3, dim3> ddaAllGatherFabricLL128Geom(ncclComm* comm, size_t perRankBytes) {
  const unsigned threads = ddaLL128AgThreads(512);
  const size_t warps = threads / (unsigned)ll128::kWarp;
  const int nCols = comm->nRanks > 1 ? comm->nRanks - 1 : 1;
  unsigned blocksPerPeer =
    ddaLL128AgBlocksPerPeer(ddaLL128AgSlices(perRankBytes), warps, nCols, ddaLL128AgBlockCap());
  if ((size_t)nCols * blocksPerPeer > (size_t)comm->ddaLLEpochLen) {
    blocksPerPeer = (unsigned)std::max(comm->ddaLLEpochLen / nCols, 1);
  }
  return std::make_pair(dim3((unsigned)nCols, blocksPerPeer), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricLL128Typed(
  const void* sendbuff, void* recvbuff,
  size_t sendcount, // per-rank element count of T (== bytes when T == int8_t)
  ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perRankBytes = sendcount * sizeof(T);
  const size_t slices = ddaLL128AgSlices(perRankBytes);
  const size_t slotWords = ddaLL128AgSlotWords(nRanks, comm->ddaScratchBytes);

  auto gridBlock = ddaAllGatherFabricLL128Geom(comm, perRankBytes);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;
  const unsigned blocksPerPeer = grid.y;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA fabric AllGather LL128: nRanks=%d perRankBytes=%zu slices=%zu grid=%ux%u block=%u "
       "(warp-per-slice, bpp=%u, slotWords=%zu)",
       nRanks, perRankBytes, slices, grid.x, grid.y, block.x, blocksPerPeer, slotWords);

  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllGatherFabricLL128<T, 4>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen, slices, slotWords);
    break;
  case 8:
    meta::comms::ddaAllGatherFabricLL128<T, 8>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen, slices, slotWords);
    break;
  default:
    meta::comms::ddaAllGatherFabricLL128<T, 0>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen, slices, slotWords);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                         ncclDataType_t datatype) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  // The pack/unpack path moves 16B chunks with no chunk straddling a line, so
  // require a 16B multiple and 16B-aligned user buffers rather than assuming it.
  const size_t perRankBytes = sendcount * ncclTypeSize(datatype);
  if (perRankBytes % 16 != 0) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(sendbuff) % 16) != 0 || (reinterpret_cast<uintptr_t>(recvbuff) % 16) != 0) {
    return false;
  }
  // Derived from the scratch allocation, so this also covers the case of a
  // buffer too small to hold a single slice per slot.
  if (perRankBytes > ddaLL128AgMaxPerRankBytes(comm->nRanks, comm->ddaScratchBytes)) {
    return false;
  }

  return true;
}

uint32_t ncclAllGatherDdaFabricLL128Blocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype) {
  const auto grid = ddaAllGatherFabricLL128Geom(comm, sendcount * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllGatherDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t sendcount,
                                         ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  // AllGather is a pure copy, so the payload moves as raw bytes: instantiate the
  // kernel once for int8_t and scale the count, like ncclAllGatherDdaFabricLL.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLL128Typed<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
