// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file model_export.cpp
/// @brief Loader exports for librocjitsu_timing_leaky.so.
///
/// Kept out of model.cpp so the model can also be linked into the unit-test
/// binary without colliding with another model's loader exports.

#include "rocjitsu/vm/timing/models/leaky/model.h"
#include "rocjitsu/vm/timing/timing_exports.h"

ROCJITSU_DEFINE_TIMING_MODEL(rocjitsu::timing::leaky::LeakyBucketModel, "leaky", "{}")
