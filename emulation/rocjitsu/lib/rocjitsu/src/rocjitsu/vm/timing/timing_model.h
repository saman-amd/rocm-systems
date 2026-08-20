// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_model.h
/// @brief The extension point: a model that turns observed execution into time.
///
/// @details rocjitsu executes functionally and has no opinion about how long
/// the modelled hardware would have taken. A timing model supplies that
/// opinion. When one is loaded it becomes the device clock the guest observes,
/// so a program that times itself measures the modelled machine rather than the
/// host the simulator happens to be running on.
///
/// The interface is at the level of observed events rather than simulator
/// hooks. rocjitsu owns the observation layer and there is exactly one of it:
/// turning hooks into a coherent per-wavefront event stream is fiddly, and a
/// bug there would otherwise be re-derived by every model that tried it.
///
/// Only four methods must be implemented. A model that costs one instruction at
/// a time and publishes a cycle count is a few dozen lines; see models/leaky.

#pragma once

#include "rocjitsu/vm/timing/event.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace rocjitsu::timing {

class TimingHost;

class TimingModel {
public:
  virtual ~TimingModel() = default;

  /// @brief Name of this model, echoed into every report so a result is always
  ///        attributable to the model that produced it.
  virtual std::string_view name() const = 0;

  /// @brief Optional payloads this model wants the host to build.
  ///
  /// @details Filling MemoryAccess::lane_addresses means writing 64 addresses
  /// per vector memory instruction, and populating StaticInstInfo::reads and
  /// ::writes means walking the operand list; together they dominate the
  /// observer's per-instruction cost. A throughput model needs neither, so it
  /// declares neither and the host does not build them.
  ///
  /// Sampled once, when the model is installed, so it must be a constant.
  /// Anything not requested arrives empty rather than absent: a model that
  /// reads a field it did not ask for sees no addresses and, per the fail-slow
  /// rule, must charge the access as if it missed everywhere.
  struct Interest {
    bool lane_addresses = false;
    bool register_ranges = false;
  };
  virtual Interest interest() const { return {}; }

  // -- Observation ----------------------------------------------------------
  //
  // Call order for one wavefront is on_wave_begin, then on_instruction once per
  // executed instruction in program order, then on_wave_end. The dispatch
  // callbacks bracket the wavefronts belonging to that dispatch.
  //
  // Wavefronts of several dispatches interleave, and calls naming different
  // compute units arrive concurrently. The host serializes calls that name the
  // same compute unit and nothing beyond that, because serializing further
  // would make the model the simulator's bottleneck. State shared across
  // compute units — a device-wide memory system, an aggregate report — is the
  // model's to protect.

  virtual void on_dispatch_begin(const DispatchInfo &) {}

  virtual void on_wave_begin(const WaveRef &) {}

  /// @brief One instruction executed on a wavefront.
  virtual void on_instruction(const WaveRef &wave, const InstructionEvent &event) = 0;

  /// @brief Every wavefront of a workgroup reached a barrier together.
  ///
  /// @details Delivered as a group because a barrier's cost is the spread
  /// between the wavefronts, which cannot be computed from any one of them.
  virtual void on_barrier(std::span<const WaveRef>) {}

  virtual void on_wave_end(const WaveRef &) {}

  /// @brief Every wavefront of a dispatch has finished.
  virtual void on_dispatch_end(const DispatchKey &) {}

  /// @brief The run is ending. Close out anything still open, so a run that
  ///        stops without a completion notification still produces a report.
  ///
  /// @details Called exactly once, however many teardown paths the host reaches.
  ///
  /// Usually after every other callback has stopped — but not guaranteed to be.
  /// A local run under the interposer never tears the simulator down, so this
  /// arrives from an exit handler while the engine thread may still be running,
  /// and a wavefront can be mid-instruction. Take whatever lock the observation
  /// path takes rather than assuming quiescence.
  virtual void on_finalize() {}

  // -- The clock ------------------------------------------------------------

  /// @brief The device's simulated "now", in shader-clock cycles.
  ///
  /// @details Read from guest timestamp paths — a guest thread inside an ioctl,
  /// the completion tracker writing a signal — on threads that have nothing to
  /// do with execution and may run concurrently with any other callback. It
  /// must therefore take none of the model's locks; publish it from a relaxed
  /// atomic that the observation path stores into.
  ///
  /// It must never move backwards. A model whose internal timeline can retreat,
  /// because it revises an estimate, must clamp here rather than expose the
  /// retreat: a guest that saw time go backwards computes a negative duration,
  /// and code that subtracts timestamps rarely defends against that.
  virtual std::uint64_t device_cycles() const = 0;

  /// @brief The shader clock rate, which turns those cycles into a time.
  ///
  /// @details Must not change during a run; a guest that has already read the
  /// clock frequency will not read it again.
  virtual double clock_ghz() const = 0;

  // -- Reporting ------------------------------------------------------------

  /// @brief Append this model's end-of-run report to @p out, as text or JSONL.
  ///
  /// @details Called after on_finalize(). The host writes it to the run's
  /// plugin sink alongside its own coverage report.
  virtual void write_report(std::string &out) const { (void)out; }
};

} // namespace rocjitsu::timing
