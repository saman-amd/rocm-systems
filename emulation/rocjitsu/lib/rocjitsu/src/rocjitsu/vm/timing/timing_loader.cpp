// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/timing_loader.h"

#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "rocjitsu/vm/timing/timing_config.h"

#include "util/dynamic_loader.h"
#include "util/log.h"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace rocjitsu::timing {
namespace {

using plugin_detail::is_valid_plugin_name;
using plugin_detail::resolve_config_json;

/// @brief Library handles kept for the lifetime of the process.
///
/// @details The model object, its vtable and its destructor all live inside the
/// library, and the host holds the model until shutdown, so there is no point
/// at which closing it would be safe and useful.
std::vector<util::LibraryHandle> &open_handles() {
  static std::vector<util::LibraryHandle> handles;
  return handles;
}

/// @brief Turn a soname into something dlopen() can find.
///
/// @details Mirrors the probe in plugin_loader.cpp, which is file-local there.
/// Duplicated rather than hoisted because the two loaders are independently
/// versioned boundaries and sharing the search policy would make a change to
/// one silently retarget the other; the layouts probed are the same because the
/// build emits both kinds of module into the same directory.
std::string resolve_model_path(const std::string &soname, const std::string &model_dir) {
  if (!model_dir.empty())
    return model_dir + "/" + soname;

#if defined(__linux__)
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void *>(&resolve_model_path), &info) != 0 && info.dli_fname) {
    std::error_code error;
    const auto module_dir = std::filesystem::canonical(info.dli_fname, error).parent_path();
    if (!error) {
      // Shared-library hosts keep models beside librocjitsu. The statically
      // linked CLI lives in <build>/tools/rocjitsu/rocjitsu in a build tree and
      // <prefix>/bin/rocjitsu when installed, so probe those layouts too.
      const std::array candidates{
          module_dir / soname,
          module_dir / ".." / "lib" / soname,
          module_dir / ".." / "lib64" / soname,
          module_dir / ".." / ".." / soname,
      };
      for (const auto &candidate : candidates) {
        if (std::filesystem::is_regular_file(candidate, error) && !error)
          return candidate.lexically_normal().string();
        error.clear();
      }
    }
  }
#endif

  return soname;
}

} // namespace

OwnedTimingModel TimingModelLoader::load(TimingConfig &config, const std::string &model_dir) {
  const std::string &name = config.model_name();

  // Shared with the plugin loader because the hazard is identical: an
  // unrestricted name interpolated into a filename turns a config key into a
  // pathname ("../evil", "/abs/path") that dlopen would load directly.
  if (!is_valid_plugin_name(name)) {
    util::Logger::warn("timing model '", name,
                       "': invalid name (allowed: letters, digits, '_', '-'), not loading");
    return {};
  }

  const std::string soname = timing_model_library_name(name);
  const std::string libpath = resolve_model_path(soname, model_dir);
  util::LibraryHandle handle = util::open_library(libpath.c_str());
  if (!handle) {
    util::Logger::warn("timing model '", name, "': cannot load ", libpath, ": ",
                       util::last_library_error());
    return {};
  }

  auto meta_fn = util::lookup_symbol<TimingModelMetadataFn>(handle, kTimingModelMetadataSymbol);
  auto create_fn = util::lookup_symbol<TimingModelCreateFn>(handle, kTimingModelCreateSymbol);
  auto destroy_fn = util::lookup_symbol<TimingModelDestroyFn>(handle, kTimingModelDestroySymbol);
  if (!meta_fn || !create_fn || !destroy_fn) {
    util::Logger::warn("timing model '", name, "': ", soname, " is missing required exports");
    util::close_library(handle);
    return {};
  }

  const TimingModelMetadata *meta = meta_fn();
  if (!meta) {
    util::Logger::warn("timing model '", name, "': metadata export returned null");
    util::close_library(handle);
    return {};
  }

  // Host and models are built and shipped together, so a mismatch is never a
  // compatibility question — it means a stale .so was found earlier on the
  // search path than the one this build produced. Naming both numbers turns
  // what would otherwise be an unexplained crash inside the model into a line
  // that says which file to delete.
  if (meta->abi_version != kTimingAbiVersion) {
    util::Logger::warn("timing model '", name, "': ", libpath, " was built against timing ABI ",
                       std::to_string(meta->abi_version), " but this build speaks ABI ",
                       std::to_string(kTimingAbiVersion),
                       "; refusing to load (a stale copy is probably earlier on the library path)");
    util::close_library(handle);
    return {};
  }

  if (meta->name && name != meta->name)
    util::Logger::warn("timing model '", name, "': metadata name '", meta->name,
                       "' differs from library name");

  // The schema validator only understands flat "string"/"number"/"boolean"
  // entries, so a model whose private configuration is nested must declare
  // "{}" and validate the object itself; resolution here would reject or
  // flatten a shape it cannot describe.
  std::string resolved;
  if (!resolve_config_json(name, meta->config_schema, config.model_config_json(), resolved)) {
    util::close_library(handle);
    return {};
  }
  // The model reads its configuration back out through the host, so the
  // defaults the schema just filled in have to land there and not merely in a
  // local string.
  config.set_model_config_json(resolved);

  TimingModel *model = create_fn(&config);
  if (!model) {
    util::Logger::warn("timing model '", name, "': create returned null");
    util::close_library(handle);
    return {};
  }

  // Keep the handle open for at least as long as the returned model: its
  // deleter calls destroy_fn, which lives in this library.
  open_handles().push_back(handle);
  util::Logger::plugins("timing model '", name, "' loaded from ", libpath);
  return OwnedTimingModel(model, TimingModelDeleter{destroy_fn});
}

} // namespace rocjitsu::timing
