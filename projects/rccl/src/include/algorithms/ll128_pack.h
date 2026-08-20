/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128 pack / poll / unpack helpers for the DDA path.
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
 
 #include "nccl_device/rccl_ptr.h"

// Which of the two wire forms the LL128 kernels use; see storeLine below. Set
// it to 0 to time the per-lane path on hardware that has the cooperative one.
#ifndef RCCL_LL128_COOP_WIRE
#define RCCL_LL128_COOP_WIRE RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
#endif

namespace meta::comms {
namespace ll128 {
 
// Lane and line geometry, mirroring device.h's WARP_SIZE and NCCL_LL128_LINESIZE
constexpr int kWarp = 32;
constexpr int kLineBytes = 128; // NCCL_LL128_LINESIZE
constexpr int kLineElems = kLineBytes / 8; // 16 u64 per line
constexpr int kLineSkip = 2 * kWarp / kLineElems; // 4 
constexpr int kWordsPerThread = 8;
constexpr int kPairs = kWordsPerThread / 2; // 4 register pairs per thread

// Wire geometry of one slice
constexpr int kWireWordsPerSlice = kWarp * kWordsPerThread; // 256 u64
constexpr int kWireBytesPerSlice = kWireWordsPerSlice * 8; // 2 KiB, 16 lines
constexpr int kFlagWordsPerSlice = kWireWordsPerSlice / kLineElems; // 16, one per line
constexpr int kDataBytesPerSlice = (kWireWordsPerSlice - kFlagWordsPerSlice) * 8; // 1920
static_assert(kWireBytesPerSlice % kLineBytes == 0, "a slice must be a whole number of lines");

 // The last lane of each line's lane group owns that line's flag word: lanes
 // 7,15,23,31 for a 128-byte line
 __device__ __forceinline__ bool isFlagLane(int wid) {
   return (wid % (kLineElems / 2)) == (kLineElems / 2 - 1);
 }
 
 // Dense 16-byte-chunk index for register-pair g of lane wid (== the prims_ll128
 // `ix` formula). Compensates for the flag holes so the packed payload in the
 // user buffer stays gap-free and coalesced.
 __device__ __forceinline__ int chunkIx(int g, int wid) {
   return g * kWarp - kLineSkip * (g / 2) + wid - (g % 2) * (wid / (kLineElems / 2));
 }
 
 __device__ __forceinline__ void store128(uint64_t* dst, uint64_t lo, uint64_t hi) {
   union {
     v4u v;
     uint64_t w[2];
   } u;
   u.w[0] = lo;
   u.w[1] = hi;
 #if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
   __builtin_amdgcn_global_store_b128((v4u_gptr)dst, u.v, RCCL_SYSTEM_SYNCSCOPE);
 #else
   __builtin_nontemporal_store(u.v, (v4u_gptr)dst);
 #endif
   asm volatile("" ::: "memory");
 }
 
 __device__ __forceinline__ void load128(const uint64_t* src, uint64_t& lo, uint64_t& hi) {
   asm volatile("" ::: "memory");
   union {
     v4u v;
     uint64_t w[2];
   } u;
 #if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
   u.v = __builtin_amdgcn_global_load_b128((v4u_gptr)src, RCCL_SYSTEM_SYNCSCOPE);
 #else
   u.v = __builtin_nontemporal_load((v4u_gptr)src);
 #endif
   lo = u.w[0];
   hi = u.w[1];
 }

__device__ __forceinline__ void storeLine(uint64_t* dst, uint64_t lo, uint64_t hi) {
#if RCCL_LL128_COOP_WIRE
  union {
    v4i v;
    uint64_t w[2];
  } u;
  u.w[0] = lo;
  u.w[1] = hi;
  __builtin_amdgcn_cooperative_atomic_store_8x16B(
    (v4i_gptr)dst, u.v, __ATOMIC_RELAXED, RCCL_SYSTEM_SYNCSCOPE);
  asm volatile("" ::: "memory");
#else
  store128(dst, lo, hi);
#endif
}

__device__ __forceinline__ void loadLine(const uint64_t* src, uint64_t& lo, uint64_t& hi) {
#if RCCL_LL128_COOP_WIRE
  asm volatile("" ::: "memory");
  union {
    v4i v;
    uint64_t w[2];
  } u;
  u.v = __builtin_amdgcn_cooperative_atomic_load_8x16B(
    (v4i_gptr)src, __ATOMIC_RELAXED, RCCL_SYSTEM_SYNCSCOPE);
  lo = u.w[0];
  hi = u.w[1];
#else
  load128(src, lo, hi);
#endif
}
 
 // load of slice's payload from `src` into registers
 template <typename T>
 __device__ __forceinline__ void loadRegs(
     uint64_t (&regs)[kWordsPerThread], const T* src, int eltN, int wid, bool flag) {
   constexpr int EltPer16B = 16 / sizeof(T);
 #pragma unroll
   for (int g = 0; g < kPairs; g++) {
     if (!flag || g % 2 == 0) {
       int ix = chunkIx(g, wid);
       if (ix * EltPer16B < eltN)
         load128(reinterpret_cast<const uint64_t*>(src + ix * EltPer16B),
                 regs[2 * g], regs[2 * g + 1]);
     }
   }
 #pragma unroll
   for (int g = 1; g < kPairs; g += 2)  // move flag-lane data out of odd regs
     if (flag) regs[2 * g] = regs[2 * g - 1];
 }
 
 // Store one slice to the wire with the flag word embedded on the flag lane
__device__ __forceinline__ void storeWire(
  uint64_t* wire, const uint64_t (&regs)[kWordsPerThread], uint64_t flag, bool flagLane) {
#pragma unroll
  for (int u = 0; u < kWordsPerThread; u += 2)
    storeLine(wire + u * kWarp, regs[u], flagLane ? flag : regs[u + 1]);
}
 
 // Poll until every line this lane reads has landed, then read the payload once.
__device__ __forceinline__ void pollWire(
  const uint64_t* wire, uint64_t (&vr)[kWordsPerThread], uint64_t flag, int wid) {
  const bool flagLane = isFlagLane(wid);
  bool needReload;
  do {
    needReload = false;
#pragma unroll
    for (int u = 0; u < kWordsPerThread; u += 2) {
      loadLine(wire + u * kWarp, vr[u], vr[u + 1]);
      needReload |= flagLane && (vr[u + 1] != flag);
    }
  } while (__any(needReload));
#pragma unroll
  for (int u = 0; u < kWordsPerThread; u += 2)
    loadLine(wire + u * kWarp, vr[u], vr[u + 1]);
}
 
 // Flag-lane un-shuffle then store of registers into `dst`
 template <typename T>
 __device__ __forceinline__ void storeRegs(
     T* dst, uint64_t (&regs)[kWordsPerThread], int eltN, int wid, bool flag) {
   constexpr int EltPer16B = 16 / sizeof(T);
 #pragma unroll
   for (int g = 1; g < kPairs; g += 2)  // reverse the load shuffle
     if (flag) regs[2 * g - 1] = regs[2 * g];
 #pragma unroll
   for (int g = 0; g < kPairs; g++) {
     if (!flag || g % 2 == 0) {
       int ix = chunkIx(g, wid);
       if (ix * EltPer16B < eltN)
         store128(reinterpret_cast<uint64_t*>(dst + ix * EltPer16B),
                  regs[2 * g], regs[2 * g + 1]);
     }
   }
 }
 
}  // namespace ll128

// Slices needed to carry perRankBytes of payload.
constexpr size_t ddaLL128AgSlices(size_t perRankBytes) {
  return (perRankBytes + ll128::kDataBytesPerSlice - 1) / ll128::kDataBytesPerSlice;
}

}  // namespace meta::comms