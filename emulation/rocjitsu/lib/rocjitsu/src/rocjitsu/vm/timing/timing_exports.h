// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_exports.h
/// @brief Loader boundary for timing-model shared objects.
///
/// @details A timing model ships as `librocjitsu_timing_<name>.so` and is found
/// through the standard dynamic-linker search path, next to the interposer.
/// It exports three `extern "C"` functions and nothing else; a linker version
/// script enforces that.
///
/// The boundary is C-shaped: no ownership crosses it except through the
/// matching create/destroy pair, so a model always allocates and frees with its
/// own allocator. What does cross it is the TimingModel vtable, called inbound
/// from rocjitsu, and the TimingHost vtable, called inbound from the model.
/// Both work without either side resolving the other's symbols, because a
/// vtable travels with its object.
///
/// Models are built in-tree and shipped with rocjitsu. This boundary carries no
/// version compatibility promise for independently built models: host and
/// models are rebuilt and distributed together.

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/timing/timing_host.h"
#include "rocjitsu/vm/timing/timing_model.h"

#include "util/log.h"

#include <cstdint>
#include <exception>
#include <string>

namespace rocjitsu::timing {

/// @brief Version of the loader boundary a shared object was built against.
///
/// @details Bumped whenever anything in this header, timing_model.h, event.h or
/// timing_host.h changes shape. Host and models are built and shipped together,
/// so this does not buy compatibility — it buys a clear diagnostic instead of
/// undefined behaviour when a stale `.so` is left in the library path, which is
/// otherwise a crash with no explanation.
inline constexpr std::uint32_t kTimingAbiVersion = 1;

/// @brief Static description of a model, returned by the metadata export.
///
/// @details Both strings point at storage owned by the shared object with
/// static lifetime, valid for as long as the library is loaded.
struct TimingModelMetadata {
  /// @brief Must be kTimingAbiVersion. The loader refuses anything else.
  std::uint32_t abi_version;
  /// @brief Model name. Must match the `<name>` in
  ///        `librocjitsu_timing_<name>.so` and the `timing.model` config value.
  const char *name;
  /// @brief JSON schema for `timing.model_config`, in the same form the
  ///        execution-plugin loader uses: each key maps to
  ///        `{ "type": ..., "description": ..., "default": ... }`. An entry with
  ///        no default is required. `"{}"` when the model takes no private
  ///        configuration.
  const char *config_schema;
};

using TimingModelMetadataFn = const TimingModelMetadata *(*)();
using TimingModelCreateFn = TimingModel *(*)(const TimingHost *host);
using TimingModelDestroyFn = void (*)(TimingModel *model);

inline constexpr const char *kTimingModelMetadataSymbol = "rocjitsu_timing_model_metadata";
inline constexpr const char *kTimingModelCreateSymbol = "rocjitsu_timing_model_create";
inline constexpr const char *kTimingModelDestroySymbol = "rocjitsu_timing_model_destroy";

/// @brief Filename a model of the given name is loaded from.
inline std::string timing_model_library_name(std::string_view model_name) {
  return "librocjitsu_timing_" + std::string(model_name) + ".so";
}

} // namespace rocjitsu::timing

#define ROCJITSU_TIMING_EXPORT RJ_API_EXPORT

/// @brief Emit the loader exports for a timing model.
///
/// @param ModelClass Concrete ::rocjitsu::timing::TimingModel subclass,
///        constructible from a single `const TimingHost &`.
/// @param NAME Model name string literal, matching the `.so` suffix.
/// @param CONFIG_SCHEMA JSON schema for `timing.model_config` ("{}" if none).
///
/// @code
///   ROCJITSU_DEFINE_TIMING_MODEL(MyModel, "mymodel", "{}")
/// @endcode
#define ROCJITSU_DEFINE_TIMING_MODEL(ModelClass, NAME, CONFIG_SCHEMA)                              \
  extern "C" ROCJITSU_TIMING_EXPORT const ::rocjitsu::timing::TimingModelMetadata *                \
  rocjitsu_timing_model_metadata() {                                                               \
    static const ::rocjitsu::timing::TimingModelMetadata kMetadata{                                \
        ::rocjitsu::timing::kTimingAbiVersion, NAME, CONFIG_SCHEMA};                               \
    return &kMetadata;                                                                             \
  }                                                                                                \
  extern "C" ROCJITSU_TIMING_EXPORT ::rocjitsu::timing::TimingModel *rocjitsu_timing_model_create( \
      const ::rocjitsu::timing::TimingHost *host) {                                                \
    if (!host)                                                                                     \
      return nullptr;                                                                              \
    try {                                                                                          \
      return new ModelClass(*host);                                                                \
    } catch (const std::exception &e) {                                                            \
      ::util::Logger::warn("timing model '", NAME, "': create failed: ", e.what());                \
      return nullptr;                                                                              \
    } catch (...) {                                                                                \
      ::util::Logger::warn("timing model '", NAME, "': create failed with unknown exception");     \
      return nullptr;                                                                              \
    }                                                                                              \
  }                                                                                                \
  extern "C" ROCJITSU_TIMING_EXPORT void rocjitsu_timing_model_destroy(                            \
      ::rocjitsu::timing::TimingModel *model) {                                                    \
    delete model;                                                                                  \
  }
