// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/observer.h"

#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/timing/timing_host.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

namespace rocjitsu::timing {
namespace {

// -- Coverage labels ---------------------------------------------------------
//
// Written once here so the ledger's vocabulary is one readable block rather
// than string literals scattered through the file. Each names the *cause* the
// observer knows and a model cannot, so an observer gap and a model gap never
// merge into one count.

constexpr std::string_view kGapNoMemoryState =
    "observer: memory instruction whose addresses the simulator did not expose";
constexpr std::string_view kGapTensorAddresses =
    "observer: tensor data mover addresses (no pipeline state on this target)";
constexpr std::string_view kGapExport = "observer: export traffic";
constexpr std::string_view kGapWaitXcnt = "observer: s_wait_xcnt charged as a full drain";
constexpr std::string_view kGapWaitEvent = "observer: s_wait_event charged as a full drain";
constexpr std::string_view kGapDispatchQueue =
    "observer: completion for a dispatch that was never announced (not forwarded)";
constexpr std::string_view kGapPlacement =
    "observer: dispatch placement (timing.machine.compute_units is 1, so every workgroup is "
    "modelled on one compute unit)";

bool has_prefix(std::string_view text, std::string_view prefix) { return text.starts_with(prefix); }

bool contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

// -- Classification ----------------------------------------------------------

/// @brief Whether a vector-memory mnemonic names an atomic.
///
/// @details Tested before load and store because an atomic mnemonic usually
/// contains neither word, and the ones that do — a returning atomic — are still
/// atomics for timing: they occupy the memory path for the whole round trip
/// whether or not the result is written back.
bool is_atomic_name(std::string_view mnemonic) { return contains(mnemonic, "atomic"); }

InstClass vector_memory_class(std::string_view mnemonic) {
  if (is_atomic_name(mnemonic))
    return InstClass::VectorMemoryAtomic;
  if (contains(mnemonic, "store"))
    return InstClass::VectorMemoryWrite;
  // Everything else in these encodings moves data toward the wavefront: loads,
  // sample and gather operations, and the prefetch-shaped ones that still
  // occupy the return path.
  return InstClass::VectorMemoryRead;
}

/// @brief Split a local-data-share mnemonic into read or write.
///
/// @details A returning atomic reads as well as writes and is classified as a
/// read, because the wavefront waits for its result and waiting is what a model
/// has to get right. A non-returning one is a pure write nothing waits on.
InstClass lds_class(std::string_view mnemonic) {
  if (contains(mnemonic, "read") || contains(mnemonic, "load") || contains(mnemonic, "_rtn"))
    return InstClass::LdsRead;
  // The lane-crossing permutes name neither a read nor a load, but they return
  // a value to a vector register and the wavefront waits for it, so they cost
  // what a read costs. ds_swizzle has the same shape. Calling them writes would
  // drop them off the counter their consumer waits on.
  if (contains(mnemonic, "permute") || contains(mnemonic, "swizzle"))
    return InstClass::LdsRead;
  if (contains(mnemonic, "write") || contains(mnemonic, "store"))
    return InstClass::LdsWrite;
  // Bare atomics, appends and consumes update memory without returning.
  return InstClass::LdsWrite;
}

/// @brief Transcendental operation stems, matched after the `v_` prefix and an
///        optional packed infix.
///
/// @details These run on the separate low-throughput pipe rather than the main
/// vector unit. Listed by operation name because that is what is stable across
/// targets; the encoding that carries them is not.
constexpr std::array<std::string_view, 10> kTranscendentalStems = {
    "rcp", "rsq", "sqrt", "log", "exp", "sin", "cos", "rcp_iflag", "tanh", "exp2",
};

/// @brief Whether a vector mnemonic names a transcendental.
///
/// @details The stem is looked for at a bounded position rather than anywhere
/// in the string, so `v_rcp_f32`, `v_pk_rcp_f16` and `v_rcp_iflag_f32` all
/// match while a longer name that merely contains the letters does not.
bool is_transcendental_name(std::string_view mnemonic) {
  std::string_view body = mnemonic.substr(2); // Past the leading "v_".
  if (has_prefix(body, "pk_"))
    body = body.substr(3);
  for (std::string_view stem : kTranscendentalStems) {
    if (!has_prefix(body, stem))
      continue;
    std::string_view rest = body.substr(stem.size());
    if (rest.empty() || rest.front() == '_')
      return true;
  }
  return false;
}

/// @brief Classify a scalar (`s_`) mnemonic.
///
/// @details Falls through to ScalarAlu rather than to Unknown, and that is a
/// classification rather than a guess: the `s_` prefix is itself evidence of
/// which unit the instruction occupies, and everything on the scalar unit that
/// is *not* scalar arithmetic is named above.
InstClass scalar_class(std::string_view mnemonic, const Instruction &inst) {
  if (has_prefix(mnemonic, "s_endpgm") || (inst.flags() & PROGRAM_TERMINATOR) != 0)
    return InstClass::Terminate;
  if (has_prefix(mnemonic, "s_nop"))
    return InstClass::Nop;
  // Before the wait family: s_wait_alu and s_delay_alu carry the waitcnt flag
  // but describe an ALU dependency, not outstanding memory.
  if (has_prefix(mnemonic, "s_delay_alu") || has_prefix(mnemonic, "s_wait_alu"))
    return InstClass::DelayAlu;
  if (has_prefix(mnemonic, "s_waitcnt") || has_prefix(mnemonic, "s_wait_"))
    return InstClass::WaitCounter;
  if (has_prefix(mnemonic, "s_barrier"))
    return InstClass::Barrier;
  if (has_prefix(mnemonic, "s_sendmsg") || has_prefix(mnemonic, "s_ttracedata") ||
      has_prefix(mnemonic, "s_incperflevel") || has_prefix(mnemonic, "s_decperflevel"))
    return InstClass::Message;
  if (has_prefix(mnemonic, "s_load") || has_prefix(mnemonic, "s_store") ||
      has_prefix(mnemonic, "s_buffer_") || has_prefix(mnemonic, "s_dcache") ||
      has_prefix(mnemonic, "s_atomic") || has_prefix(mnemonic, "s_prefetch") ||
      has_prefix(mnemonic, "s_atc_probe"))
    return InstClass::ScalarMemory;
  if (has_prefix(mnemonic, "s_branch") || has_prefix(mnemonic, "s_cbranch") ||
      has_prefix(mnemonic, "s_setpc") || has_prefix(mnemonic, "s_swappc") ||
      has_prefix(mnemonic, "s_call") || has_prefix(mnemonic, "s_getpc"))
    return InstClass::Branch;
  return InstClass::ScalarAlu;
}

/// @brief Classify a vector (`v_`) mnemonic.
InstClass vector_class(std::string_view mnemonic, const Instruction &inst) {
  if (inst.is_mfma() || has_prefix(mnemonic, "v_mfma") || has_prefix(mnemonic, "v_smfmac") ||
      has_prefix(mnemonic, "v_wmma") || has_prefix(mnemonic, "v_swmmac"))
    return InstClass::MatrixMultiply;
  if (is_transcendental_name(mnemonic))
    return InstClass::Transcendental;
  return InstClass::VectorAlu;
}

/// @brief Classify by mnemonic family, or nothing when no family matches.
std::optional<InstClass> class_by_family(std::string_view mnemonic) {
  if (has_prefix(mnemonic, "ds_"))
    return lds_class(mnemonic);
  if (has_prefix(mnemonic, "global_") || has_prefix(mnemonic, "flat_") ||
      has_prefix(mnemonic, "scratch_") || has_prefix(mnemonic, "buffer_") ||
      has_prefix(mnemonic, "tbuffer_") || has_prefix(mnemonic, "image_"))
    return vector_memory_class(mnemonic);
  if (has_prefix(mnemonic, "tensor_"))
    return InstClass::TensorMemory;
  if (has_prefix(mnemonic, "exp") || has_prefix(mnemonic, "export"))
    return InstClass::Export;
  return std::nullopt;
}

/// @brief The class of @p inst, or InstClass::Unknown.
///
/// @details Unknown is the only fallback, and it is not a cheap one: it maps to
/// the vector pipe and the config resolves it to the most expensive issue cost
/// the part names. That is the whole difference from a classifier with an
/// `Other` bucket — such a bucket occupies no unit and costs nothing, so an
/// opcode nobody has ever classified makes a run read *fast*, and a coverage
/// gap becomes indistinguishable from a fast kernel.
InstClass classify(const Instruction &inst) {
  std::string_view mnemonic = inst.mnemonic();

  // The scalar and vector prefixes cover the overwhelming majority of a kernel
  // and need the instruction's flags to resolve, so they come first and
  // separately from the name-only families.
  if (has_prefix(mnemonic, "s_"))
    return scalar_class(mnemonic, inst);
  if (has_prefix(mnemonic, "v_"))
    return vector_class(mnemonic, inst);
  if (std::optional<InstClass> by_family = class_by_family(mnemonic))
    return *by_family;

  // Fall back to the flags rocjitsu does set, so an instruction whose name has
  // never been seen is still placed on the right unit when its metadata says
  // enough. Branch precedes the memory flag because an indirect branch carries
  // neither a recognizable name nor a memory role.
  if (inst.is_barrier())
    return InstClass::Barrier;
  if (inst.is_waitcnt())
    return InstClass::WaitCounter;
  if (inst.is_branch() || (inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0)
    return InstClass::Branch;
  if ((inst.flags() & PROGRAM_TERMINATOR) != 0)
    return InstClass::Terminate;
  // A memory op of unknown direction is called a read: a read is the direction
  // the wavefront stalls on, and mis-costing a store as a load overstates the
  // dependency rather than dropping it.
  if (inst.is_memory_op())
    return InstClass::VectorMemoryRead;

  return InstClass::Unknown;
}

/// @brief Re-decide a memory class now that the addresses are known.
///
/// @details A FLAT access is named identically whether it reaches memory or the
/// local data share; only the aperture its addresses fall in decides.
InstClass refine_with_memory_space(InstClass initial, bool is_local, bool is_load) {
  switch (initial) {
  case InstClass::VectorMemoryRead:
  case InstClass::VectorMemoryWrite:
  case InstClass::LdsRead:
  case InstClass::LdsWrite:
    if (is_local)
      return is_load ? InstClass::LdsRead : InstClass::LdsWrite;
    return is_load ? InstClass::VectorMemoryRead : InstClass::VectorMemoryWrite;
  default:
    // An atomic, a scalar access and a tensor transfer are already on the only
    // path they can take.
    return initial;
  }
}

// -- Register ranges ---------------------------------------------------------

/// @brief Translate a decoded register reference into a dependency range.
/// @returns Whether the reference names a file a model tracks.
bool range_from(const RegisterRef &ref, RegisterRange &out) {
  switch (ref.cls) {
  case RegClass::VGPR:
    out.file = RegisterFile::Vector;
    break;
  case RegClass::ACC_VGPR:
    out.file = RegisterFile::Accumulator;
    break;
  case RegClass::SGPR:
    out.file = RegisterFile::Scalar;
    break;
  default:
    // The condition code, mode and program counter carry dependencies the
    // scalar unit forwards in a single cycle. Tracking them would add
    // bookkeeping on the hottest path for a stall the hardware does not have.
    return false;
  }
  out.index = ref.index;
  out.count = ref.width == 0 ? 1u : ref.width;
  return true;
}

void append_operand(const Operand *operand, std::vector<RegisterRange> &ranges) {
  if (operand == nullptr)
    return;
  std::optional<RegisterRef> ref = operand->to_register_ref();
  if (!ref.has_value())
    return;
  RegisterRange range;
  if (range_from(*ref, range))
    ranges.push_back(range);
}

// -- Waits -------------------------------------------------------------------

/// @brief Bring every counter to zero: the wavefront drains completely.
void set_full_drain(WaitThresholds &out) {
  for (std::size_t index = 0; index < kNumWaitCounters; ++index)
    out.set(static_cast<WaitCounter>(index), 0);
}

/// @brief Fill @p out with the thresholds *this* instruction established.
///
/// @param mnemonic Decides which fields the instruction wrote.
/// @param target The wavefront's decoded wait target, read after execution.
/// @returns A coverage label when the decode was a pessimistic guess, else
///          empty.
///
/// @details The wavefront's wait target is sticky: an instruction naming one
/// counter leaves every other field holding whatever an earlier wait put there.
/// Copying the whole target would therefore attribute stale thresholds to this
/// instruction, and a model that believed them would insert stalls the hardware
/// never had. The mnemonic says exactly which fields were written, and rocjitsu
/// has already decoded the immediate into them — so the decoded value is used
/// rather than a per-family immediate layout duplicated here, which would have
/// to be kept in step with the ISA generator forever.
std::string_view fill_wait_event(std::string_view mnemonic, const amdgpu::WaitTarget &target,
                                 WaitThresholds &out) {
  if (mnemonic == "s_waitcnt") {
    out.set(WaitCounter::VectorLoad, target.vmcnt);
    out.set(WaitCounter::LgkmCombined, target.lgkmcnt);
    out.set(WaitCounter::Export, target.expcnt);
    return {};
  }
  // The GFX10 single-counter spellings. They are checked before the GFX11 `
  // s_wait_<name>` family because they share no prefix with it, and leaving
  // them to fall through would have produced a wait that constrains nothing —
  // the exact shape of a stall silently costing zero.
  if (mnemonic == "s_waitcnt_vmcnt") {
    out.set(WaitCounter::VectorLoad, target.vmcnt);
    return {};
  }
  if (mnemonic == "s_waitcnt_lgkmcnt") {
    out.set(WaitCounter::LgkmCombined, target.lgkmcnt);
    return {};
  }
  if (mnemonic == "s_waitcnt_expcnt") {
    out.set(WaitCounter::Export, target.expcnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_waitcnt_vscnt") || has_prefix(mnemonic, "s_wait_storecnt")) {
    out.set(WaitCounter::VectorStore, target.vscnt);
    if (mnemonic.ends_with("_dscnt"))
      out.set(WaitCounter::LdsAndGds, target.dscnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_wait_loadcnt")) {
    out.set(WaitCounter::VectorLoad, target.vmcnt);
    if (mnemonic.ends_with("_dscnt"))
      out.set(WaitCounter::LdsAndGds, target.dscnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_wait_dscnt")) {
    out.set(WaitCounter::LdsAndGds, target.dscnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_wait_kmcnt")) {
    out.set(WaitCounter::ScalarMemory, target.kmcnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_wait_tensorcnt")) {
    out.set(WaitCounter::Tensor, target.tensorcnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_wait_asynccnt")) {
    out.set(WaitCounter::Async, target.asynccnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_wait_expcnt")) {
    out.set(WaitCounter::Export, target.expcnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_wait_samplecnt") || has_prefix(mnemonic, "s_wait_bvhcnt")) {
    // rocjitsu folds both onto the vector load counter.
    out.set(WaitCounter::VectorLoad, target.vmcnt);
    return {};
  }
  if (has_prefix(mnemonic, "s_wait_idle") || mnemonic == "s_waitcnt_depctr") {
    // A full drain. Naming every counter is equivalent and saves a model a
    // separate "wait for everything" path.
    set_full_drain(out);
    return {};
  }
  // s_wait_xcnt and s_wait_event name things this event vocabulary has no
  // counter for and the simulator keeps no threshold for. Charging a full drain
  // is the slowest reasonable reading and is therefore the one taken: it can
  // only over-serialize, whereas the alternative — leaving the thresholds
  // unconstrained — makes the instruction free, and a wait that costs nothing
  // is precisely the unbounded optimistic bias this whole design exists to
  // refuse. Both are declared, so the ledger says how much of the run depends
  // on the guess.
  if (has_prefix(mnemonic, "s_wait_xcnt")) {
    set_full_drain(out);
    return kGapWaitXcnt;
  }
  if (has_prefix(mnemonic, "s_wait_event")) {
    set_full_drain(out);
    return kGapWaitEvent;
  }
  // Anything else that reached here was classified as a wait by an instruction
  // flag rather than by name, so its counters are unknown: drain.
  set_full_drain(out);
  return kGapWaitEvent;
}

/// @brief Translate rocjitsu's wait-counter type into the event vocabulary.
///
/// @details Taken from the simulator rather than derived from the class,
/// because which counter an operation posts to is a per-target ISA decision.
/// Vector stores are the case that bites: the compute targets have no separate
/// store counter and post stores to the load counter, so a model told the store
/// counter would park those completions where no wait instruction on that
/// target can name them, and the wait would cost nothing.
WaitCounter wait_counter_for(amdgpu::WaitCounterType type) {
  switch (type) {
  case amdgpu::WaitCounterType::VMCNT:
  case amdgpu::WaitCounterType::LOADCNT:
    return WaitCounter::VectorLoad;
  case amdgpu::WaitCounterType::VSCNT:
  case amdgpu::WaitCounterType::STORECNT:
    return WaitCounter::VectorStore;
  case amdgpu::WaitCounterType::LGKMCNT:
    return WaitCounter::LgkmCombined;
  case amdgpu::WaitCounterType::DSCNT:
    return WaitCounter::LdsAndGds;
  case amdgpu::WaitCounterType::KMCNT:
    return WaitCounter::ScalarMemory;
  case amdgpu::WaitCounterType::EXPCNT:
    return WaitCounter::Export;
  case amdgpu::WaitCounterType::TENSORCNT:
    return WaitCounter::Tensor;
  case amdgpu::WaitCounterType::ASYNCCNT:
    return WaitCounter::Async;
  }
  return WaitCounter::Count;
}

// -- Memory ------------------------------------------------------------------

/// @brief Whether @p address falls in the wavefront's shared aperture.
///
/// @details Written to agree exactly with the test compute_unit.cpp uses to
/// route the same access, including the base-is-zero guard and the inclusive
/// limit. The two must not drift: the functional path decides where the data
/// went and this decides what it costs, and a disagreement charges LDS latency
/// for a trip to memory or the reverse.
bool in_shared_aperture(const amdgpu::Wavefront &wf, std::uint64_t address) {
  return wf.shared_aperture_base() != 0 && address >= wf.shared_aperture_base() &&
         address <= wf.shared_aperture_limit();
}

/// @brief Build a memory access from the state rocjitsu attached while
///        executing the instruction.
///
/// @param want_lane_addresses Whether the model asked for per-lane addresses.
/// @returns Whether the instruction carried memory state at all.
bool fill_memory_event(const Instruction &inst, const amdgpu::Wavefront &wf,
                       bool want_lane_addresses, MemoryAccess &out) {
  const DynamicInstState *state = inst.data();
  if (state == nullptr)
    return false;

  if (state->tag() == amdgpu::SCALAR_MEM) {
    const auto *scalar = inst.data_as<amdgpu::ScalarMemState>();
    out.space = MemorySpace::Scalar;
    out.is_load = scalar->is_load;
    out.scalar_address = scalar->addr;
    out.scalar_bytes = scalar->num_dwords * (scalar->elem_size == 0 ? 4u : scalar->elem_size);
    if (out.scalar_bytes == 0)
      out.scalar_bytes = 4;
    out.wait_counter = wait_counter_for(scalar->wait_counter_type);
    return true;
  }

  if (state->tag() != amdgpu::GLOBAL_MEM && state->tag() != amdgpu::LOCAL_MEM)
    return false;

  const auto *vector = inst.data_as<amdgpu::VectorMemState>();
  out.is_load = vector->is_load;
  out.non_temporal = vector->non_temporal;
  out.wait_counter = wait_counter_for(vector->wait_counter_type);
  // A decoder that left the width unset gets four bytes, the same assumption
  // the compute unit's own bounds check makes for the same fields.
  const std::uint32_t element = vector->elem_size == 0 ? 4u : vector->elem_size;
  const std::uint32_t elements = vector->num_elems == 0 ? 1u : vector->num_elems;
  out.bytes_per_lane = element * elements;

  const std::uint32_t lanes = std::min<std::uint32_t>(vector->wf_size, 64);
  if (want_lane_addresses)
    out.lane_addresses.reserve(lanes);
  // The aperture scan is not optional even when the addresses are not wanted:
  // it is what tells a FLAT access that reached the LDS from one that reached
  // memory, and that decides which unit and which latency the access is costed
  // on. Only the copying is skipped, which is the part that dominates — sixty
  // four stores per vector memory instruction, on every wavefront.
  bool any_shared = false;
  bool any_global = false;
  for (std::uint32_t lane = 0; lane < lanes; ++lane) {
    if ((vector->lane_mask & (1ull << lane)) == 0)
      continue;
    const std::uint64_t address = vector->per_lane_addr[lane];
    if (want_lane_addresses)
      out.lane_addresses.push_back(address);
    if (in_shared_aperture(wf, address))
      any_shared = true;
    else
      any_global = true;
    if (!want_lane_addresses && any_global)
      break; // The space is already decided; see the mixed-access rule below.
  }
  // A mixed access — some lanes in the aperture, some outside — is costed as
  // global. It is the pessimistic reading of the two and also the honest one:
  // the lanes that left the compute unit are the ones that set the latency.
  const bool local = state->tag() == amdgpu::LOCAL_MEM || (any_shared && !any_global);
  out.space = local ? MemorySpace::LocalDataShare : MemorySpace::Global;
  // The addresses are known whether or not the model wanted the list: the space
  // and the byte counts are exact either way, and those are what a model that
  // declined the list costs the access from. addresses_known is about the
  // observer having lost track of an access, not about the caller having
  // declined detail it does not use -- conflating the two would charge every
  // throughput model a full-width miss on every access it ever sees.
  return true;
}

} // namespace

// -- Construction ------------------------------------------------------------

namespace {

/// @brief Makes sure a run's timing report is emitted even when nothing ever
///        tears the simulator down.
///
/// @details A local run under the interposer destroys the VM only on
/// construction-failure paths, so neither onShutdown() nor ~TimingObserver()
/// fires on a successful run, and a report written from either would simply
/// never appear. Process exit is the one event left. This is host code rather
/// than something a model has to arrange for itself, which is the point: a
/// third-party model should not have to discover that its report never printed.
///
/// The registered observer is cleared by ~TimingObserver, so the handler can
/// never reach one that has already been destroyed — which is reachable in
/// daemon mode, where a second rj_vm_load_plugins() replaces the first
/// observer while the process keeps running.
class ExitReporter {
public:
  static void arm(TimingObserver &observer) {
    std::lock_guard lock(mutex());
    if (!std::exchange(registered(), true))
      std::atexit(&ExitReporter::run);
    current() = &observer;
  }

  static void disarm(const TimingObserver &observer) {
    std::lock_guard lock(mutex());
    if (current() == &observer)
      current() = nullptr;
  }

private:
  static void run() {
    std::lock_guard lock(mutex());
    if (TimingObserver *observer = current())
      observer->finalize_once();
  }

  // Function-local statics rather than namespace-scope ones: the handler runs
  // during static destruction, where the order against a namespace-scope object
  // in this translation unit is not something to rely on.
  static std::mutex &mutex() {
    static std::mutex m;
    return m;
  }
  static TimingObserver *&current() {
    static TimingObserver *observer = nullptr;
    return observer;
  }
  static bool &registered() {
    static bool value = false;
    return value;
  }
};

} // namespace

TimingObserver::TimingObserver(TimingModel &model, const TimingHost &host)
    : ExecutionPlugin("timing"), model_(model), host_(host), interest_(model.interest()),
      time_source_(model) {
  // Registered here rather than by the caller so that every path that builds an
  // observer gets a report, including a test that builds one directly.
  ExitReporter::arm(*this);

  // The same pessimistic default every other reader of this key uses, so the
  // coverage report cannot name a value nothing used. Zero is not a legal
  // answer here: the API forbids sentinel values, and treating it as "let the
  // simulator place them" would silently hand the model the emulator's own
  // placement, which crams a whole grid onto the compute units of one die.
  const std::uint64_t declared = host_.tune("compute_units", kPessimisticComputeUnits);
  declared_compute_units_ = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, declared));
  // A part with one compute unit is not a part anyone is modelling, so this
  // value means the config either omitted the key or described something that
  // cannot spread a grid. Both deserve the same note, and the coverage report
  // separately says which of the two it was.
  if (declared_compute_units_ <= kPessimisticComputeUnits)
    host_.note_unmodeled(kGapPlacement);
}

void TimingObserver::finalize_once() {
  if (finalized_.exchange(true))
    return;
  model_.on_finalize();

  // The model's report and the run's coverage record go out together, and the
  // coverage record goes out second. A reader who stops at the numbers has
  // still had to scroll past what produced them: which parameters the config
  // actually named, and which effects the model told us it was not modelling.
  // Emitting the numbers alone would make a report look like evidence of
  // something it never measured.
  std::string report;
  model_.write_report(report);
  host_.write_coverage_report(report);
  if (!report.empty())
    sink().write(report);
}

TimingObserver::~TimingObserver() {
  ExitReporter::disarm(*this);

  // The terminal call has to happen on a path that always runs. A local run
  // under the interposer never destroys the VM, so onShutdown() may never fire
  // at all, and a model that only closed its report there would produce
  // nothing. Destruction is the one event that cannot be skipped.
  finalize_once();
}

void TimingObserver::onShutdown() { finalize_once(); }

// -- Static instruction properties -------------------------------------------

const TimingObserver::CacheEntry &TimingObserver::static_info(std::uint64_t pc,
                                                              const Instruction &inst) {
  {
    std::shared_lock<std::shared_mutex> guard(info_mutex_);
    auto found = info_cache_.find(pc);
    if (found != info_cache_.end() && found->second->info.mnemonic == inst.mnemonic())
      return *found->second;
  }

  auto entry = std::make_unique<CacheEntry>();
  StaticInstInfo &info = entry->info;
  info.mnemonic = std::string(inst.mnemonic());
  info.inst_class = classify(inst);
  info.size_bytes = static_cast<std::uint32_t>(inst.size() <= 0 ? 4 : inst.size());

  if (interest_.register_ranges) {
    for (int index = 0; index < inst.num_src_operands(); ++index)
      append_operand(inst.src_operand(index), info.reads);
    for (int index = 0; index < inst.num_dst_operands(); ++index)
      append_operand(inst.dst_operand(index), info.writes);
  }

  if (info.inst_class == InstClass::DelayAlu && inst.num_src_operands() > 0) {
    if (const Operand *operand = inst.src_operand(0))
      info.delay_immediate = static_cast<std::uint32_t>(operand->encoding_value());
  }

  if (info.inst_class == InstClass::Unknown) {
    // Per opcode rather than a single bucket, because the list of names is what
    // somebody fixing the classifier needs, and it is bounded by the number of
    // distinct opcodes in the run. Built here, once, so that charging it on
    // every execution — which is what makes the count a coverage measure — is
    // free of allocation on the hot path.
    entry->gap_label = "observer: opcode not classified: " + info.mnemonic;
  }

  std::unique_lock<std::shared_mutex> guard(info_mutex_);
  std::unique_ptr<CacheEntry> &slot = info_cache_[pc];
  // Another wavefront may have derived the same entry concurrently. Keeping
  // whichever landed first is fine — they describe the same instruction — but
  // an entry left by translated code that used to live here must be replaced.
  if (!slot || slot->info.mnemonic != info.mnemonic) {
    // The superseded entry is retired, not freed. Other wavefronts are holding
    // it right now as a pending instruction, and a model may be holding it as
    // InstructionEvent::info, both of which this class promises will stay valid
    // (see event.h). Freeing it here is a use-after-free reachable only when
    // translated code reuses an address, which is exactly the case nothing
    // routinely exercises. The list is bounded by how often that reuse happens,
    // not by instruction count.
    if (slot)
      retired_info_.push_back(std::move(slot));
    slot = std::move(entry);
  }
  return *slot;
}

// -- Dispatches --------------------------------------------------------------

void TimingObserver::onAmdgpuDispatchPacketProcessed(const rocjitsu::KernelDispatchInfo &info) {
  DispatchInfo out;
  out.key.dispatch_id = info.dispatch_id;
  out.key.queue_id = info.queue_id;
  out.kernel_name = info.kernelNameOrUnknown();
  out.grid_size[0] = info.grid_size_x;
  out.grid_size[1] = info.grid_size_y;
  out.grid_size[2] = info.grid_size_z;
  out.workgroup_size[0] = info.workgroup_size_x;
  out.workgroup_size[1] = info.workgroup_size_y;
  out.workgroup_size[2] = info.workgroup_size_z;
  out.workgroup_count = info.workgroup_count;
  out.waves_per_workgroup = info.wfs_per_workgroup;
  out.vector_registers_per_wave = info.vgprs_per_wf;
  out.scalar_registers_per_wave = info.sgprs_per_wf;
  out.lds_bytes_per_workgroup = info.lds_bytes_per_workgroup;
  out.wave_size = info.wave_size;

  {
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    // onAmdgpuDispatchExecutionEnd carries only the id, so the queue half of
    // the key has to be remembered from here, where it is still available.
    dispatch_queue_[info.dispatch_id] = info.queue_id;
    if (!announced_.insert(packed(out.key)).second)
      return;
  }
  model_.on_dispatch_begin(out);
}

void TimingObserver::ensure_dispatch_announced(const DispatchKey &key,
                                               const amdgpu::Wavefront &wf) {
  {
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    dispatch_queue_[key.dispatch_id] = key.queue_id;
    if (!announced_.insert(packed(key)).second)
      return;
  }
  // Everything recoverable from a wavefront, and nothing invented. A shape with
  // an unknown name still lands in the right report row; a wavefront with no
  // dispatch at all does not.
  DispatchInfo out;
  out.key = key;
  out.kernel_name = rocjitsu::kUnknownKernelIdentity;
  out.wave_size = wf.wf_size();
  out.vector_registers_per_wave = wf.num_vgprs();
  out.scalar_registers_per_wave = wf.num_sgprs();
  model_.on_dispatch_begin(out);
}

void TimingObserver::onAmdgpuDispatchExecutionEnd(std::uint32_t dispatch_id) {
  DispatchKey key;
  key.dispatch_id = dispatch_id;
  bool queue_known = false;
  {
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    auto found = dispatch_queue_.find(dispatch_id);
    if (found != dispatch_queue_.end()) {
      key.queue_id = found->second;
      dispatch_queue_.erase(found);
      queue_known = true;
    }
    announced_.erase(packed(key));
  }
  // The completion tracker retires every queue entry it holds, and not all of
  // them are kernel dispatches this observer announced. Delivering an end for
  // one of those would be worse than dropping it: with no queue recovered the
  // key would be (0, id), which on a multi-XCD part is a real dispatch some
  // other command processor announced, and the model would close that one
  // instead. Nothing is open for a dispatch that was never announced, so
  // staying quiet leaves nothing dangling — but it is still counted, because a
  // rising count here would mean real kernel dispatches were going unannounced.
  if (!queue_known) {
    host_.note_unmodeled(kGapDispatchQueue);
    return;
  }
  model_.on_dispatch_end(key);
}

// -- Wavefronts --------------------------------------------------------------

std::uint32_t TimingObserver::placed_compute_unit(const amdgpu::Wavefront &wf) const {
  // Round-robin by workgroup, which is how a shader-processor input spreads
  // them. Every wavefront of a workgroup lands on the same unit, so the shard
  // lock a barrier takes stays the shard lock its members' instructions take.
  return wf.wg_id() % declared_compute_units_;
}

void TimingObserver::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
  auto state = std::make_unique<ObservedWave>();
  WaveRef &ref = state->ref;
  ref.dispatch.dispatch_id = wf.dispatch_id();
  ref.dispatch.queue_id = wf.queue_id();
  ref.workgroup_id = wf.wg_id();
  ref.wave_slot = wf.wf_id();
  ref.compute_unit_id = placed_compute_unit(wf);
  ref.wave_lanes = wf.wf_size();
  // The wavefront's own allocation rather than the file's size, so a model
  // sizing a scoreboard from it pays for what the kernel asked for.
  ref.vector_registers = std::max<std::uint32_t>(wf.num_vgprs(), 1);
  ref.scalar_registers = std::max<std::uint32_t>(wf.num_sgprs(), 1);

  const WaveRef announced = ref;
  // Overwrites whatever an earlier dispatch left in this slot: plugin state
  // survives Wavefront::reset(), so a recycled slot still holds the previous
  // wavefront's identity until it is replaced here.
  wf.set_plugin_state(slot_index(), std::move(state));

  ensure_dispatch_announced(announced.dispatch, wf);
  std::lock_guard<std::mutex> guard(shard_for(announced.compute_unit_id));
  model_.on_wave_begin(announced);
}

void TimingObserver::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  ObservedWave *state = wave_state(wf);
  if (state == nullptr)
    return;
  const WaveRef ref = state->ref;

  // The instruction that terminated the wavefront never reaches the after hook:
  // an s_endpgm with no outstanding waits halts inside execute_instruction, and
  // the compute unit returns on `is_halted()` before firing it. Emitting the
  // still-pending instruction here is what stops the model losing exactly one
  // instruction — the terminator — from every wavefront in the run.
  //
  // branch_taken is false: an instruction reaching this path ended the program
  // rather than transferring control within it, and there is no post-execute
  // program counter left to compare against anyway.
  //
  // Emitted before the shard lock is taken rather than under it, because
  // emit_pending() takes that same lock itself. Ordering is not at risk: this
  // hook and the instruction hooks for one wavefront run on the wavefront's own
  // thread, so the model still sees the terminator before the wave ends.
  if (state->has_pending)
    emit_pending(*state, wf, /*branch_taken=*/false, /*inst=*/nullptr);

  std::lock_guard<std::mutex> guard(shard_for(ref.compute_unit_id));
  model_.on_wave_end(ref);
}

void TimingObserver::onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
  std::vector<WaveRef> refs;
  refs.reserve(wavefronts.size());
  for (amdgpu::Wavefront *wf : wavefronts) {
    if (wf == nullptr)
      continue;
    if (const ObservedWave *state = wave_state(*wf))
      refs.push_back(state->ref);
  }
  if (refs.empty())
    return;

  // Usually one shard: every wavefront of a workgroup shares a compute unit. A
  // cluster barrier spans several, so the distinct shards are taken in
  // ascending order, which is the whole of the deadlock argument — every path
  // in this file acquires shard locks in that order and holds no other lock
  // while doing it.
  std::vector<std::size_t> shards;
  shards.reserve(refs.size());
  for (const WaveRef &ref : refs)
    shards.push_back(ref.compute_unit_id & (kNumShards - 1));
  std::sort(shards.begin(), shards.end());
  shards.erase(std::unique(shards.begin(), shards.end()), shards.end());

  for (std::size_t shard : shards)
    shards_[shard].lock();
  model_.on_barrier(refs);
  for (auto shard = shards.rbegin(); shard != shards.rend(); ++shard)
    shards_[*shard].unlock();
}

// -- Instructions ------------------------------------------------------------

void TimingObserver::onAmdgpuBeforeExecuteInstruction(std::uint64_t pc, const Instruction &inst,
                                                      amdgpu::Wavefront &wf) {
  ObservedWave *state = wave_state(wf);
  if (state == nullptr)
    return;
  // A handful of paths retire an instruction without reaching the after hook:
  // an s_trap that enters the trap handler, and an s_trap with no configured
  // handler, both return early. Their instruction is still pending here, and
  // overwriting it would silently drop one instruction per occurrence — the
  // same class of loss as the terminator below, and just as invisible. The
  // program counter has already moved by now, so the fall-through address is
  // what says whether control was transferred.
  if (state->has_pending && state->pending_info != nullptr) {
    const bool taken = pc != state->pending_pc + state->pending_info->size_bytes;
    emit_pending(*state, wf, taken, /*inst=*/nullptr);
  }
  const CacheEntry &entry = static_info(pc, inst);
  state->pending_pc = pc;
  state->pending_info = &entry.info;
  state->pending_gap = entry.gap_label.empty() ? nullptr : &entry.gap_label;
  state->pending_active_lanes = static_cast<std::uint32_t>(std::popcount(wf.exec()));
  state->has_pending = true;
}

void TimingObserver::onAmdgpuAfterExecuteInstruction(std::uint64_t pc, const Instruction &inst,
                                                     amdgpu::Wavefront &wf) {
  ObservedWave *state = wave_state(wf);
  if (state == nullptr || !state->has_pending)
    return;
  // The compute unit fires this hook *before* applying the common `pc += size`
  // step, so for straight-line code the reported program counter is still the
  // one the instruction issued at, and only an instruction that wrote the
  // program counter itself reads back different. Comparing against the
  // fall-through address instead would be true of every instruction and would
  // charge the whole kernel a taken-branch penalty per instruction.
  emit_pending(*state, wf, /*branch_taken=*/pc != state->pending_pc, &inst);
}

void TimingObserver::emit_pending(ObservedWave &state, const amdgpu::Wavefront &wf,
                                  bool branch_taken, const Instruction *inst) {
  state.has_pending = false;
  const StaticInstInfo *info = state.pending_info;
  if (info == nullptr)
    return;

  InstructionEvent event;
  event.pc = state.pending_pc;
  event.info = info;
  event.effective_class = info->inst_class;
  event.active_lanes = state.pending_active_lanes;
  event.wave_lanes = wf.wf_size();
  event.branch_taken = branch_taken;

  if (state.pending_gap != nullptr)
    host_.note_unmodeled(*state.pending_gap);

  if (info->inst_class == InstClass::WaitCounter) {
    const std::string_view gap = fill_wait_event(info->mnemonic, wf.wait_target(), event.wait);
    if (!gap.empty())
      host_.note_unmodeled(gap);
  }

  if (inst != nullptr && fill_memory_event(*inst, wf, interest_.lane_addresses, event.memory)) {
    event.effective_class = refine_with_memory_space(
        event.effective_class, event.memory.space == MemorySpace::LocalDataShare,
        event.memory.is_load);
  } else if (class_is_memory(event.effective_class)) {
    // The class says traffic was issued and no state came back to describe it —
    // a tensor transfer, whose pipeline state this target does not attach; an
    // instruction whose decoder never filled one; or the terminal path, where
    // the instruction object is already gone. The access is still declared,
    // with addresses_known false, so a model charges it as an uncoalesced miss
    // to the farthest level it models. Dropping it instead is the single most
    // tempting and most wrong thing available here: it costs nothing, it
    // removes real traffic, and it leaves no trace in the numbers.
    switch (event.effective_class) {
    case InstClass::LdsRead:
      event.memory.space = MemorySpace::LocalDataShare;
      break;
    case InstClass::LdsWrite:
      event.memory.space = MemorySpace::LocalDataShare;
      event.memory.is_load = false;
      break;
    case InstClass::ScalarMemory:
      event.memory.space = MemorySpace::Scalar;
      break;
    case InstClass::TensorMemory:
      event.memory.space = MemorySpace::Tensor;
      break;
    case InstClass::VectorMemoryWrite:
      event.memory.space = MemorySpace::Global;
      event.memory.is_load = false;
      break;
    case InstClass::Export:
      // Exports have no memory space in this vocabulary and no per-lane
      // addresses; a model costs them from the class and the export unit
      // alone, which leaves the bandwidth they consume unmodelled.
      host_.note_unmodeled(kGapExport);
      break;
    default:
      event.memory.space = MemorySpace::Global;
      break;
    }
    if (event.memory.valid()) {
      event.memory.addresses_known = false;
      host_.note_unmodeled(event.effective_class == InstClass::TensorMemory ? kGapTensorAddresses
                                                                            : kGapNoMemoryState);
    }
  }

  std::lock_guard<std::mutex> guard(shard_for(state.ref.compute_unit_id));
  model_.on_instruction(state.ref, event);
}

} // namespace rocjitsu::timing
