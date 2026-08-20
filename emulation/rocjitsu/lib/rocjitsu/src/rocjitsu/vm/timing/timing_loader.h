// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_loader.h
/// @brief Host-side loading of a timing model from its shared object.
///
/// @details A run has at most one timing model, named by `timing.model` in the
/// architecture config file:
///
/// @code{.json}
///   "timing": {
///     "model": "leaky",
///     "clock_mhz": 2200,
///     "machine": { "compute_units": 256, "vector_alu": { "issue_cycles": 4 } },
///     "model_config": { }
///   }
/// @endcode
///
/// The loader opens `librocjitsu_timing_<model>.so`, checks that it was built
/// against this loader boundary, resolves `timing.model_config` against the
/// model's declared schema, and instantiates the model against the supplied
/// TimingConfig as its TimingHost.
///
/// This is deliberately a near-copy of PluginLoader rather than a shared
/// generic: the two boundaries have different metadata, different ABI rules and
/// different failure policies — a plugin that fails to load is skipped and the
/// run continues, whereas a timing model that fails to load leaves the run with
/// no clock authority at all, which the caller has to notice.

#pragma once

#include "rocjitsu/vm/timing/timing_exports.h"

#include <memory>
#include <string>

namespace rocjitsu::timing {

class TimingConfig;

/// @brief Frees a model through its own library's destroy export.
///
/// @details Allocation and deallocation must stay on the same side of the
/// dynamic-library boundary: the model was built with its own allocator and its
/// destructor lives in its `.so`, so `delete` from the host is undefined even
/// when both sides happen to share a libstdc++.
struct TimingModelDeleter {
  TimingModelDestroyFn destroy = nullptr;

  void operator()(TimingModel *model) const {
    if (model && destroy)
      destroy(model);
  }
};

/// @brief A model instance owned by the host, freed by its library.
using OwnedTimingModel = std::unique_ptr<TimingModel, TimingModelDeleter>;

/// @brief Loads the model named by a parsed `timing` block.
class TimingModelLoader {
public:
  /// @brief Open the model's shared object and instantiate it.
  ///
  /// @param config The parsed `timing` block. It is also the TimingHost handed
  ///        to the model, so it must outlive the returned model; the loader
  ///        additionally writes the schema-resolved `model_config` back into it
  ///        (see TimingConfig::set_model_config_json), because only the loader
  ///        has seen the model's schema and the model reads its configuration
  ///        back out through the host.
  /// @param model_dir When non-empty, a trusted directory the shared object is
  ///        loaded from by explicit path. Required in daemon mode, where the
  ///        process is not re-`exec`'d and so cannot rely on a
  ///        launcher-populated `LD_LIBRARY_PATH`. When empty the library is
  ///        resolved beside the interposer and then by soname through the
  ///        standard dynamic-linker search path.
  ///
  /// @returns The model, or a null pointer when it could not be loaded. Every
  ///          failure is reported to the log first; a caller that gets null
  ///          must run without a timing model rather than substitute one, since
  ///          a substituted model would publish numbers attributed to the model
  ///          the config actually named.
  ///
  /// @details The library handle is kept open for the lifetime of the process:
  /// the model's code, vtable and destructor all live inside it.
  static OwnedTimingModel load(TimingConfig &config, const std::string &model_dir = {});
};

} // namespace rocjitsu::timing
