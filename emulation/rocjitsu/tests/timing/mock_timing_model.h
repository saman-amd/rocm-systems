// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mock_timing_model.h
/// @brief Test doubles for the timing plugin API's two vtables.
///
/// @details event.h names no rocjitsu type on purpose, so everything worth
/// asserting about a model can be asserted by handing it events built here,
/// with no simulator, no compiled kernel and no GPU. These doubles are the
/// other half of that: a TimingHost whose tuning is a map rather than a config
/// file, and three models that each isolate one property of the contract —
/// what the callbacks promise, what a cycle count is worth in nanoseconds, and
/// what happens when a model breaks the monotonicity rule.
///
/// MockTimingHost reimplements TimingConfig's fail-slow resolution rather than
/// stubbing it out. If an unnamed class resolved to something cheap here, every
/// model test would silently be exercising a policy the production host does
/// not have, and the tests that matter most — the ones about unclassified
/// instructions — would prove nothing.

#pragma once

#include "rocjitsu/vm/timing/event.h"
#include "rocjitsu/vm/timing/inst_class.h"
#include "rocjitsu/vm/timing/time_source.h"
#include "rocjitsu/vm/timing/timing_host.h"
#include "rocjitsu/vm/timing/timing_model.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::timing::test {

/// @brief A TimingHost backed by a settable parameter map.
///
/// @details Also a ledger, in the same shape as TimingConfig's: a test can ask
/// which keys a model requested and which of them fell back, which is how the
/// fail-slow rule is checked without parsing a report.
class MockTimingHost final : public TimingHost {
public:
  void set_int(std::string key, std::uint64_t value) { ints_[std::move(key)] = value; }
  void set_real(std::string key, double value) { reals_[std::move(key)] = value; }

  /// @brief Name a class's issue cost, i.e. write `<class>.issue_cycles`.
  void set_class_issue_cycles(InstClass cls, std::uint64_t cycles) {
    set_int(std::string(inst_class_name(cls)) + ".issue_cycles", cycles);
  }

  void set_model_config_json(std::string json) { model_config_json_ = std::move(json); }
  void set_clock_ghz(double ghz) { clock_ghz_ = ghz; }

  // -- TimingHost -----------------------------------------------------------

  std::uint64_t tune(std::string_view key, std::uint64_t pessimistic) const override {
    auto it = ints_.find(key);
    const bool named = it != ints_.end();
    const std::uint64_t value = named ? it->second : pessimistic;
    record(key, named);
    return value;
  }

  double tune_real(std::string_view key, double pessimistic) const override {
    auto it = reals_.find(key);
    const bool named = it != reals_.end();
    const double value = named ? it->second : pessimistic;
    record(key, named);
    return value;
  }

  std::uint64_t class_issue_cycles(InstClass cls) const override {
    const std::string key = std::string(inst_class_name(cls)) + ".issue_cycles";
    auto it = ints_.find(key);
    if (it != ints_.end()) {
      record(key, true);
      return it->second;
    }
    std::uint64_t worst = 1;
    for (std::size_t i = 0; i < kNumInstClasses; ++i) {
      auto named =
          ints_.find(std::string(inst_class_name(static_cast<InstClass>(i))) + ".issue_cycles");
      if (named != ints_.end())
        worst = std::max(worst, named->second);
    }
    record(key, false);
    return worst;
  }

  std::string model_config_json() const override { return model_config_json_; }
  double clock_ghz() const override { return clock_ghz_; }

  void note_unmodeled(std::string_view effect) const override {
    std::lock_guard lock(mutex_);
    ++unmodeled_[std::string(effect)];
  }

  void log(std::string_view message) const override {
    std::lock_guard lock(mutex_);
    logs_.emplace_back(message);
  }

  /// @brief The same two sections TimingConfig emits, in the same order.
  ///
  /// @details Written out rather than stubbed so a test that asserts on a
  /// report's shape is asserting on something a production run would also
  /// produce.
  void write_coverage_report(std::string &out) const override {
    std::lock_guard lock(mutex_);
    out += "parameters in effect (" + std::to_string(queried_.size()) + "):\n";
    for (const auto &[key, from_config] : queried_) {
      out += "  machine." + key;
      if (!from_config)
        out += "   [NOT IN CONFIG - pessimistic default]";
      out += "\n";
    }
    if (unmodeled_.empty()) {
      out += "unmodelled effects: none declared\n";
      return;
    }
    out += "unmodelled effects (" + std::to_string(unmodeled_.size()) + " kinds):\n";
    for (const auto &[effect, count] : unmodeled_)
      out += "  " + effect + "   x" + std::to_string(count) + "\n";
  }

  // -- Assertions -----------------------------------------------------------

  bool queried(std::string_view key) const {
    std::lock_guard lock(mutex_);
    return queried_.find(key) != queried_.end();
  }

  /// @returns True when @p key was asked for and the map did not name it.
  bool fell_back(std::string_view key) const {
    std::lock_guard lock(mutex_);
    auto it = queried_.find(key);
    return it != queried_.end() && !it->second;
  }

  std::uint64_t unmodeled_count(std::string_view effect) const {
    std::lock_guard lock(mutex_);
    auto it = unmodeled_.find(effect);
    return it == unmodeled_.end() ? 0 : it->second;
  }

  std::size_t unmodeled_kinds() const {
    std::lock_guard lock(mutex_);
    return unmodeled_.size();
  }

  /// @brief The whole ledger, for a caller matching on part of a label rather
  ///        than on the exact wording of one.
  std::map<std::string, std::uint64_t, std::less<>> unmodeled() const {
    std::lock_guard lock(mutex_);
    return unmodeled_;
  }

  std::vector<std::string> logs() const {
    std::lock_guard lock(mutex_);
    return logs_;
  }

private:
  void record(std::string_view key, bool from_config) const {
    std::lock_guard lock(mutex_);
    queried_.insert_or_assign(std::string(key), from_config);
  }

  std::map<std::string, std::uint64_t, std::less<>> ints_;
  std::map<std::string, double, std::less<>> reals_;
  std::string model_config_json_ = "{}";
  double clock_ghz_ = 1.0;

  mutable std::mutex mutex_;
  mutable std::map<std::string, bool, std::less<>> queried_;
  mutable std::map<std::string, std::uint64_t, std::less<>> unmodeled_;
  mutable std::vector<std::string> logs_;
};

/// @brief A model that costs nothing and remembers everything.
///
/// @details For asserting the observation contract itself — call order, the
/// identity of what was passed, one on_wave_begin per wave, dispatch callbacks
/// bracketing their wavefronts. Arguments are deep-copied, so a test can
/// compare against them after the driving loop has moved on; the exception is
/// InstructionEvent::info, which is a borrowed pointer into the caller's
/// StaticInstInfo storage exactly as it is in production.
class RecordingModel final : public TimingModel {
public:
  enum class Callback : std::uint8_t {
    DispatchBegin,
    WaveBegin,
    Instruction,
    Barrier,
    WaveEnd,
    DispatchEnd,
    Finalize,
  };

  struct Call {
    Callback kind = Callback::Finalize;
    DispatchInfo dispatch;              ///< DispatchBegin.
    DispatchKey key;                    ///< DispatchEnd.
    WaveRef wave;                       ///< WaveBegin, Instruction, WaveEnd.
    InstructionEvent event;             ///< Instruction.
    std::vector<WaveRef> barrier_waves; ///< Barrier.
  };

  std::string_view name() const override { return "recording"; }

  Interest interest() const override { return interest_; }
  void set_interest(Interest interest) { interest_ = interest; }

  void on_dispatch_begin(const DispatchInfo &info) override {
    Call call;
    call.kind = Callback::DispatchBegin;
    call.dispatch = info;
    call.key = info.key;
    push(std::move(call));
  }

  void on_wave_begin(const WaveRef &wave) override {
    Call call;
    call.kind = Callback::WaveBegin;
    call.wave = wave;
    push(std::move(call));
  }

  void on_instruction(const WaveRef &wave, const InstructionEvent &event) override {
    Call call;
    call.kind = Callback::Instruction;
    call.wave = wave;
    call.event = event;
    push(std::move(call));
  }

  void on_barrier(std::span<const WaveRef> waves) override {
    Call call;
    call.kind = Callback::Barrier;
    call.barrier_waves.assign(waves.begin(), waves.end());
    push(std::move(call));
  }

  void on_wave_end(const WaveRef &wave) override {
    Call call;
    call.kind = Callback::WaveEnd;
    call.wave = wave;
    push(std::move(call));
  }

  void on_dispatch_end(const DispatchKey &key) override {
    Call call;
    call.kind = Callback::DispatchEnd;
    call.key = key;
    push(std::move(call));
  }

  void on_finalize() override {
    Call call;
    call.kind = Callback::Finalize;
    push(std::move(call));
  }

  std::uint64_t device_cycles() const override { return 0; }
  double clock_ghz() const override { return clock_ghz_; }
  void set_clock_ghz(double ghz) { clock_ghz_ = ghz; }

  void write_report(std::string &out) const override {
    out += "recording: " + std::to_string(calls().size()) + " calls\n";
  }

  std::vector<Call> calls() const {
    std::lock_guard lock(mutex_);
    return calls_;
  }

  /// @brief Just the kinds, in order, which is what an ordering assertion wants.
  std::vector<Callback> sequence() const {
    std::lock_guard lock(mutex_);
    std::vector<Callback> kinds;
    kinds.reserve(calls_.size());
    for (const Call &call : calls_)
      kinds.push_back(call.kind);
    return kinds;
  }

  std::size_t count(Callback kind) const {
    std::lock_guard lock(mutex_);
    return static_cast<std::size_t>(std::count_if(
        calls_.begin(), calls_.end(), [kind](const Call &call) { return call.kind == kind; }));
  }

private:
  void push(Call call) {
    std::lock_guard lock(mutex_);
    calls_.push_back(std::move(call));
  }

  mutable std::mutex mutex_;
  std::vector<Call> calls_;
  Interest interest_;
  double clock_ghz_ = 1.0;
};

/// @brief Charges every instruction the same constant, and nothing else.
///
/// @details The point of it is that guest-visible time becomes exactly
/// predictable: N instructions at C cycles is N*C cycles, and
/// nanoseconds_for() says what that is worth at the configured clock. A test
/// can therefore assert an exact nanosecond figure out of SimulatedClock rather
/// than a bound, which is the only way to catch a conversion that is off by the
/// clock ratio — an error that leaves the numbers plausibly shaped.
class FixedCostModel final : public TimingModel {
public:
  explicit FixedCostModel(std::uint64_t cycles_per_instruction = 1, double clock_ghz = 1.0)
      : cycles_per_instruction_(cycles_per_instruction), clock_ghz_(clock_ghz) {}

  std::string_view name() const override { return "fixed"; }

  void on_instruction(const WaveRef &, const InstructionEvent &) override {
    device_cycles_.fetch_add(cycles_per_instruction_, std::memory_order_relaxed);
    instructions_.fetch_add(1, std::memory_order_relaxed);
  }

  std::uint64_t device_cycles() const override {
    return device_cycles_.load(std::memory_order_relaxed);
  }
  double clock_ghz() const override { return clock_ghz_; }

  std::uint64_t instructions() const { return instructions_.load(std::memory_order_relaxed); }

  /// @brief What @p instructions of this model's work is worth, in nanoseconds.
  double nanoseconds_for(std::uint64_t instructions) const {
    return static_cast<double>(instructions * cycles_per_instruction_) / clock_ghz_;
  }

private:
  std::uint64_t cycles_per_instruction_;
  double clock_ghz_;
  std::atomic<std::uint64_t> device_cycles_{0};
  std::atomic<std::uint64_t> instructions_{0};
};

/// @brief A model that breaks the never-retreat rule on purpose.
///
/// @details TimingModel::device_cycles() is documented as never moving
/// backwards, and a model that revises an estimate is told to clamp rather than
/// expose the retreat. This one does not clamp, so a test can prove the clock
/// downstream of it does — the guarantee a guest actually depends on has to
/// hold even when a model is wrong, because a negative duration is a defect
/// nobody's timing code defends against.
class RetreatingModel final : public TimingModel {
public:
  explicit RetreatingModel(std::uint64_t cycles = 0, double clock_ghz = 1.0)
      : cycles_(cycles), clock_ghz_(clock_ghz) {}

  std::string_view name() const override { return "retreating"; }

  void on_instruction(const WaveRef &, const InstructionEvent &) override {}

  /// @brief Set the raw value the next read reports, in either direction.
  void set_cycles(std::uint64_t cycles) { cycles_.store(cycles, std::memory_order_relaxed); }

  std::uint64_t device_cycles() const override { return cycles_.load(std::memory_order_relaxed); }
  double clock_ghz() const override { return clock_ghz_; }

private:
  std::atomic<std::uint64_t> cycles_;
  double clock_ghz_;
};

/// @brief The narrow TimeSource view of a model, as the observer installs it.
///
/// @details Holds a reference rather than owning: SimulatedClock never frees a
/// binding, so anything it has been handed must outlive the last read, and
/// making that the caller's problem keeps the lifetime visible in the test.
class ModelTimeSource final : public TimeSource {
public:
  explicit ModelTimeSource(const TimingModel &model) : model_(model) {}

  std::uint64_t current_cycles() const override { return model_.device_cycles(); }
  double clock_ghz() const override { return model_.clock_ghz(); }

private:
  const TimingModel &model_;
};

} // namespace rocjitsu::timing::test
