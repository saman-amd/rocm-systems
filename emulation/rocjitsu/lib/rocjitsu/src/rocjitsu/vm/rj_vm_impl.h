// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm_impl.h
/// @brief Private definition of rj_vm_t. Internal to the library.

#ifndef ROCJITSU_VM_RJ_VM_IMPL_H_
#define ROCJITSU_VM_RJ_VM_IMPL_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/refcount.h"
#include "rocjitsu/vm/timing/timing_config.h"
#include "rocjitsu/vm/timing/timing_loader.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "simdojo/sim/simulation.h"

#include <atomic>
#include <memory>

struct rj_vm_t : rocjitsu::RefCounted {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  simdojo::SimulationEngine::Config engine_config{};
  rocjitsu::config::LoadedConfig loaded;
  rocjitsu::SoC *soc = nullptr;
  rocjitsu::VirtualMachine *vm = nullptr;
  std::atomic<bool> plugin_group_active{false};

  /// @brief The loaded timing model and the config block that configured it.
  ///
  /// @details Held here rather than in the plugin group because the observer
  /// the group owns holds references to both, so both have to outlive it. The
  /// order is enforced explicitly in shutdown_plugin_group() rather than left
  /// to member destruction order, which would put the group (reached through
  /// `loaded`) and these on two unrelated teardown paths.
  ///
  /// `config` is also the model's TimingHost, which is why it is a pointer that
  /// stays put rather than a value that could be moved out from under a model
  /// already holding the reference.
  std::unique_ptr<rocjitsu::timing::TimingConfig> timing_config;
  rocjitsu::timing::OwnedTimingModel timing_model;
};

#endif // ROCJITSU_VM_RJ_VM_IMPL_H_
