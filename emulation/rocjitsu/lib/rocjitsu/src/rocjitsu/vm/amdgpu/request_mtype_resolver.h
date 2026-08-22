// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_REQUEST_MTYPE_RESOLVER_H_
#define ROCJITSU_VM_AMDGPU_REQUEST_MTYPE_RESOLVER_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstdint>
#include <limits>

namespace rocjitsu {
namespace amdgpu {

/// Caches the effective MTYPE for the current page within one memory request.
/// MTYPE is treated as stable until the request advances to another page.
class RequestMtypeResolver {
public:
  RequestMtypeResolver(GpuMemory *memory, uint32_t vmid)
      : memory_(memory), vmid_(vmid), fallback_(Mtype::RW), combine_(false),
        request_guard_(memory ? memory->acquire_page_table_request(vmid)
                              : GpuMemory::PageTableRequestGuard{}) {}

  RequestMtypeResolver(GpuMemory *memory, uint32_t vmid, Mtype instruction_mtype)
      : memory_(memory), vmid_(vmid), fallback_(instruction_mtype), combine_(true),
        request_guard_(memory ? memory->acquire_page_table_request(vmid)
                              : GpuMemory::PageTableRequestGuard{}) {}

  Mtype fallback() const { return fallback_; }

  Mtype at(uint64_t addr) {
    if (!memory_)
      return fallback_;

    const uint64_t page = addr >> GpuMemory::PAGE_SHIFT;
    // Registrations without a request lease retain the generation-checked
    // lookup on every chunk instead of caching across a possible mutation.
    if (page != page_ || (vmid_ != 0 && !request_guard_.owns_lock())) {
      page_ = page;
      const Mtype pte_mtype = memory_->pte_mtype(addr, vmid_);
      page_mtype_ = combine_ ? effective_mtype(fallback_, pte_mtype) : pte_mtype;
    }
    return page_mtype_;
  }

private:
  GpuMemory *memory_;
  uint32_t vmid_;
  Mtype fallback_;
  bool combine_;
  GpuMemory::PageTableRequestGuard request_guard_;
  uint64_t page_ = std::numeric_limits<uint64_t>::max();
  Mtype page_mtype_ = Mtype::RW;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_REQUEST_MTYPE_RESOLVER_H_
