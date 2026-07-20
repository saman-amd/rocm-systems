// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file guest_kfd.h
/// @brief KFD discovery driver that appends one synthetic DBT guest GPU.

#ifndef ROCJITSU_KMD_LINUX_GUEST_KFD_H_
#define ROCJITSU_KMD_LINUX_GUEST_KFD_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/sysfs.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

namespace rocjitsu {

class GuestKfdTestAccess;

/// @brief KFD driver that exposes a guest GPU for DBT while forwarding host GPU work.
///
/// @details This class appends one guest GPU to KFD topology and process
/// apertures, but does not execute guest queues itself. With the hardware
/// backend, host-GPU operations are forwarded to real /dev/kfd. With the
/// simulator backend, they are delegated to the supplied simulated KFD. If an
/// execution ioctl still targets the guest GPU, the driver returns an error so
/// the missing HSA forwarding path is visible.
class GuestKfd : public LinuxKfd {
public:
  /// @brief Construct a guest discovery driver from parsed DBT configuration.
  /// @param execution_driver Optional simulated host KFD. When null, host
  ///        execution is forwarded to the real `/dev/kfd`. When provided,
  ///        GuestKfd adopts its bootstrap open reference (or opens it lazily)
  ///        and releases that reference on the last application close.
  explicit GuestKfd(config::DbtGuestConfig config, LinuxKfd *execution_driver = nullptr);

  /// @brief Close the execution KFD and remove generated overlay state.
  ~GuestKfd() override;

  /// @brief Open the execution KFD and lazily prepare guest discovery.
  int open() override;

  /// @brief Handle close for the /dev/kfd fd represented by this driver.
  int close() override;

  /// @brief Route guest discovery ioctls locally and host ioctls to the execution KFD.
  int ioctl(unsigned long request, void *arg) override;

  /// @brief Map host-backed KFD offsets and reject unsupported guest doorbells.
  void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) override;

  /// @brief Forward unmaps for mappings created through this driver.
  int munmap(void *addr, size_t length) override;

  /// @brief Return the underlying hardware or simulated KFD fd.
  [[nodiscard]] int fd() const override;

  /// @brief Return true when @p fd is an internal rocjitsu-owned fd.
  [[nodiscard]] bool owns_fd(int fd) const override;

  /// @brief Redirect KFD topology and guest DRM sysfs paths into the overlay.
  [[nodiscard]] std::string redirect_sysfs_path(const char *path) const override;

  /// @brief Return true if a mapping range overlaps a protected doorbell.
  [[nodiscard]] bool is_doorbell_range(const void *addr, size_t length) const override;

  /// @brief Forward fixed-map replacement to the simulated execution driver when present.
  void *mmap_replacing_client_doorbell_views(void *addr, size_t length, int prot, int flags, int fd,
                                             off_t offset) override;

  /// @brief Return true when @p minor is the configured guest render node.
  [[nodiscard]] bool handles_drm_render_minor(uint32_t minor) const override;

  /// @brief Return synthetic AMDGPU metadata for the guest render node.
  [[nodiscard]] const Sysfs::GpuInfo *gpu_info_for_render_minor(uint32_t minor) const override;

  /// @brief Return the generated KFD topology root.
  [[nodiscard]] std::string topology_path() const override;

  /// @brief Return the simulated host DRM root, or empty for hardware execution.
  [[nodiscard]] std::string drm_path() const override;

  /// @brief Prepare guest discovery without retaining an application open fd.
  bool prepare_for_discovery();

  /// @brief Add one open reference for a duplicated KFD fd.
  /// @retval true A reference was added.
  /// @retval false No live guest process to retain.
  [[nodiscard]] bool retain_local_open() override;

  /// @brief Number of app-facing KFD descriptor references still live.
  /// @details Only application-visible dup fds are counted — the internal real
  /// /dev/kfd fd is not — so this driver's count is zero at publication and its
  /// teardown baseline is zero.
  [[nodiscard]] uint32_t local_open_ref_count() const override;

  /// @brief Release blocking calls before teardown so their driver snapshot drops.
  /// @details Two distinct backends, one contract — in both cases a parked
  /// WAIT_EVENTS returns a benign KFD_IOC_WAIT_RESULT_TIMEOUT and nothing else is
  /// mutated:
  /// - Simulator execution: WAIT_EVENTS is served by the SimulatedKfd execution
  ///   driver, so this DELEGATES to it and its local process's waiters are
  ///   released.
  /// - Hardware execution: WAIT_EVENTS is forwarded to the real kernel, which this
  ///   process cannot interrupt. Instead this sets hw_closing_, which
  ///   forward_wait_events_bounded()'s poll loop observes between its short kernel
  ///   polls and returns on — CANCELLING the wait rather than waking it.
  /// Idempotent.
  void begin_local_shutdown() override;

  /// @brief Stop classifying the hidden real /dev/kfd fd number as KFD after a
  /// dup2/dup3 overwrote it.
  /// @details GuestKfd hands applications ordinary dup fds and keeps the real
  /// /dev/kfd fd internal (real_kfd_fd_). If a dup2/dup3 target reuses that hidden
  /// fd number, fd()/owns_fd() must stop reporting it as KFD so later ioctl/mmap/
  /// close are not routed to whatever now occupies the number. The real fd is NOT
  /// counted in open_refs_ (only app-facing dups are), so a match returns
  /// kClearedKeepRefs — the interposer must clear the classification WITHOUT
  /// dropping an open reference. For simulator execution, the separately owned
  /// backend open also stays pinned until the final app-facing close; a later
  /// primary-fd re-mint is balanced so it does not add another backend reference.
  /// Returns kNotPrimary if @p fd is not the current hidden real fd.
  [[nodiscard]] PrimaryInvalidation invalidate_primary_fd(int fd) override;

private:
  friend class GuestKfdTestAccess;

  class TopologyOverlay;

  /// @brief Open real KFD, generate topology, and select the host GPU.
  bool ensure_ready();

  /// @brief Prepare guest discovery while mutex_ is already held.
  bool ensure_ready_locked();

  /// @brief Open the process's real /dev/kfd fd while mutex_ is held.
  bool ensure_real_kfd_locked();

  /// @brief Forward one ioctl to the real /dev/kfd fd.
  int forward_ioctl(unsigned long request, void *arg);

  /// @brief Forward WAIT_EVENTS to the real kernel as bounded, cancellable polls.
  /// @details A hardware-backed guest has no simulator wait to wake, and the real
  /// kernel WAIT_EVENTS can block indefinitely. The interposer dispatches this
  /// ioctl while holding a driver snapshot, so an indefinite kernel wait
  /// would keep that pin held and deadlock teardown (begin_local_shutdown() cannot
  /// wake a kernel syscall, and closing the fd does not cancel an in-flight wait).
  /// This breaks a long/indefinite wait into short kernel polls that re-check
  /// hw_closing_ between iterations, so begin_local_shutdown() can cancel it and the
  /// pin drains promptly. Mirrors the RemoteDriver client-side WAIT_EVENTS loop.
  int forward_wait_events_bounded(unsigned long request, void *arg);

  /// @brief The bounded-poll algorithm behind forward_wait_events_bounded().
  /// @details Split out and fully injected so its four outcomes — zero-timeout
  /// pass-through, cancellation (with the caller's timeout restored), indefinite
  /// polling until an event completes, and finite-deadline expiry — are unit
  /// testable without a real /dev/kfd. @p poll performs ONE short kernel poll,
  /// reading and writing @p args (the loop has already set args.timeout to the
  /// per-poll slice) and returning the ioctl result. @p cancelled is the flag
  /// begin_local_shutdown() sets; it is re-read between polls, never inside one.
  /// @returns The ioctl result to hand back to the caller; args.timeout is always
  /// restored to the caller's original value before returning.
  static int wait_events_poll_loop(kfd_ioctl_wait_events_args &args,
                                   const std::atomic<bool> &cancelled,
                                   const std::function<int()> &poll);

  /// @brief Return real process apertures plus one synthetic guest aperture.
  int get_process_apertures_ioctl(void *arg) override;

  /// @brief Return guest clock-counter values or forward host requests.
  int get_clock_counters_ioctl(void *arg) override;

  /// @brief Succeed guest VM acquisition without creating a guest execution VM.
  int acquire_vm_ioctl(void *arg) override;

  /// @brief Report the configured guest-visible local memory size.
  int get_available_memory_ioctl(void *arg) override;

  /// @brief Accept guest startup memory policy setup and forward host policy.
  int set_memory_policy_ioctl(void *arg) override;

  /// @brief Allocate a synthetic KFD memory handle for guest startup bookkeeping.
  int alloc_memory_ioctl(void *arg) override;

  /// @brief Release a synthetic KFD memory handle or forward a real handle.
  int free_memory_ioctl(void *arg) override;

  /// @brief Rewrite guest gpu_id entries to the selected host before mapping.
  int map_memory_ioctl(void *arg) override;

  /// @brief Mirror map_memory rewrites for unmap requests.
  int unmap_memory_ioctl(void *arg) override;

  /// @brief Shared guest-to-host device-id rewrite for map/unmap memory ioctls.
  template <typename Args> int map_or_unmap_memory_ioctl(Args *args, unsigned long request);

  /// @brief Fail unsupported guest execution ioctls visibly.
  int reject_guest_execution_ioctl(unsigned long request, void *arg) const;

  /// @brief Return true when an ioctl argument names the synthetic guest GPU.
  bool request_targets_guest(unsigned long request, void *arg) const;

  /// @brief Build the synthetic aperture record appended after real apertures.
  kfd_process_device_apertures guest_apertures() const;

  config::DbtGuestConfig config_;
  /// @brief Non-owning simulated execution driver owned by the local VM.
  LinuxKfd *execution_driver_ = nullptr;
  Sysfs::GpuInfo guest_{};
  std::unique_ptr<TopologyOverlay> overlay_;
  mutable std::mutex mutex_;
  std::atomic<int> real_kfd_fd_{-1};
  uint32_t open_refs_ = 0;
  bool owns_execution_driver_open_ = false;
  uint32_t host_gpu_id_ = 0;
  static constexpr uint64_t kSyntheticHandleBase = 1ULL << 63;
  uint64_t next_synthetic_handle_ = kSyntheticHandleBase;
  std::unordered_set<uint64_t> synthetic_handles_;
  std::unordered_set<uint64_t> synthetic_mmap_offsets_;
  std::atomic<bool> ready_{false};
  /// @brief Set by begin_local_shutdown() for a hardware-backed guest so an
  /// in-flight bounded WAIT_EVENTS poll loop returns and drops its driver snapshot
  /// before teardown.
  std::atomic<bool> hw_closing_{false};
  /// @brief The synthetic descriptor applications receive from open().
  /// @details A memfd, NEVER a duplicate of the real /dev/kfd, mirroring what
  /// SimulatedKfd hands out. This is a security boundary, not a convenience: a
  /// forked child inherits the parent's descriptor table, and under the
  /// fork-then-exec contract the interposer passes a child's ioctl/dup/dup2/fcntl
  /// straight to libc. Had the app-facing fd been a real KFD duplicate, that child
  /// could drive real hardware through it and dup2() it to clear FD_CLOEXEC and
  /// carry it across the exec that was supposed to sanitize the process. A memfd
  /// carries no such authority: inherited, it is inert.
  std::atomic<int> app_fd_{-1};

  /// @brief Create the synthetic app-facing descriptor once. Caller holds mutex_.
  bool ensure_app_fd_locked();

  /// @brief The PRIVATE real /dev/kfd descriptor used for host forwarding.
  /// @details Never returned to an application; see app_fd_.
  [[nodiscard]] int host_fd() const;
};

/// @brief Test-only handle to GuestKfd's internal bounded-wait algorithm.
/// @details wait_events_poll_loop() is the cancellation contract the interposer's
/// teardown depends on for a hardware-backed guest, but reaching it through
/// GuestKfd::ioctl() needs a real /dev/kfd. It is a pure function of its injected
/// poll and cancellation flag, so exposing it here lets the contract be tested on
/// any host. Nothing outside tests may use this.
class GuestKfdTestAccess {
public:
  static int wait_events_poll_loop(kfd_ioctl_wait_events_args &args,
                                   const std::atomic<bool> &cancelled,
                                   const std::function<int()> &poll) {
    return GuestKfd::wait_events_poll_loop(args, cancelled, poll);
  }
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_GUEST_KFD_H_
