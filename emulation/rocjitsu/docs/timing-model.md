# Timing Models

rocjitsu executes GPU code functionally: it computes the right answer, as fast as
the host can. It has no opinion about how long the modelled hardware would have
taken. A **timing model** supplies that opinion, and it ships as its own shared
object, loaded through its own `timing` config block. It is deliberately not one
of the execution `plugins`: a model drives the clock every guest program reads,
and an observer selected by accident must never end up doing that.

When one is loaded the simulated device clock becomes that model's clock, and a
guest program that times itself measures the modelled machine instead of the
host it happens to be running on. `hipEventElapsedTime`, the KFD clock counters,
the SDMA timestamp packet and `s_memtime` inside a kernel all resolve to the
same modelled timeline.

The extension point is deliberately narrow. A model receives observed execution
as a stream of plain structs and answers one question — what time is it. This is
enough to build a cycle-accurate microarchitectural model, and small enough that
a useful "leaky bucket" throughput model is about a hundred lines.

## Shape

```
    guest program
         │ hipEventElapsedTime, KFD clock counters, s_memtime
         ▼
    SimulatedClock ◄──────────── TimingModel::device_cycles()
         ▲                                    ▲
         │ installed for the run              │ events
    rocjitsu core ──hooks──► TimingObserver ──┘
                                   ▲
                                   │ tuning, coverage ledger
                             TimingHost ◄── architecture config file
```

rocjitsu owns the observation layer, and there is exactly one of it. Turning
execution hooks into a coherent per-wavefront event stream is fiddly in ways
that are invisible when you get them wrong — a wavefront's last instruction is
never reported by the after-execute hook, the wait-counter target register is
sticky across instructions, a FLAT access is named identically whether it
reaches memory or the LDS — and a bug there would otherwise be re-derived by
every model that tried it. A model never sees a rocjitsu type.

## The interface

A model implements `rocjitsu::timing::TimingModel` (`vm/timing/timing_model.h`).
Four methods are pure virtual; everything else defaults to doing nothing.

```cpp
class TimingModel {
public:
  virtual std::string_view name() const = 0;

  // Observation
  virtual Interest interest() const { return {}; }   // opt in to costly payloads
  virtual void on_dispatch_begin(const DispatchInfo &) {}
  virtual void on_wave_begin(const WaveRef &) {}
  virtual void on_instruction(const WaveRef &, const InstructionEvent &) = 0;
  virtual void on_barrier(std::span<const WaveRef>) {}
  virtual void on_wave_end(const WaveRef &) {}
  virtual void on_dispatch_end(const DispatchKey &) {}
  virtual void on_finalize() {}   // once, at the end of the run

  // The clock
  virtual std::uint64_t device_cycles() const = 0;
  virtual double clock_ghz() const = 0;

  // Reporting
  virtual void write_report(std::string &out) const {}
};
```

Call order for one wavefront is `on_wave_begin`, then `on_instruction` once per
executed instruction in program order, then `on_wave_end`. Dispatch callbacks
bracket the wavefronts belonging to that dispatch. Wavefronts of several
dispatches interleave, and calls naming different compute units arrive
concurrently: the host serialises calls that name the same `compute_unit_id` and
nothing more, because serialising further would make the model the simulator's
bottleneck. Anything a model shares across compute units is the model's to
protect.

`device_cycles()` is the exception to all of that. It is read from threads with
nothing to do with execution — a guest thread inside an ioctl, the completion
tracker writing a signal — so it must take none of the model's locks and must
never move backwards. Publish it from a relaxed atomic.

`interest()` exists because two payloads dominate the observer's per-instruction
cost: filling 64 lane addresses per vector memory access, and walking the operand
list to build register ranges. A throughput model needs neither and does not pay
for them. Declining a payload is not the same as the observer losing track of
something: byte counts and classes stay exact, and `memory.addresses_known`
stays true. It goes false only when the observer issued an access it genuinely
could not place, which a model must then charge as a miss to the farthest level
it models.

## Events

`InstructionEvent` is what a model costs. The static half is derived once per
program counter and shared between every execution of it; only what varies per
execution travels in the event.

| Field | Meaning |
|---|---|
| `pc`, `info->mnemonic` | identity, for the model's own reporting |
| `info->inst_class` | what kind of instruction this is |
| `info->reads`, `info->writes` | register ranges, when requested |
| `effective_class` | the class to cost *this* execution as |
| `active_lanes`, `wave_lanes` | per-lane work, and passes for a wave wider than the SIMD |
| `branch_taken` | whether control flow left the fall-through path |
| `wait` | the thresholds this `s_waitcnt` waits down to |
| `memory` | space, byte counts, addresses, and the counter the wave will wait on |

`effective_class` is separate from `info->inst_class` because a FLAT access can
only be told apart from an LDS access by the addresses it produced. The observer
sharpens the class per execution rather than writing back into the shared static
entry, which other wavefronts are reading concurrently.

`memory.lane_addresses` carries real per-lane byte addresses, taken after
address calculation. This is the reason to model timing inside a functional
simulator rather than over a static instruction list: coalescing, cache
behaviour and bank conflicts all depend on values the kernel actually computed,
and no static analysis recovers them. `memory.wait_counter` is likewise reported
by the simulator rather than derived from the class, because which counter a
store posts to is a per-target ISA decision — the older compute targets have no
separate store counter and post stores to the load counter.

Every key is the `(dispatch_id, queue_id)` pair, not `dispatch_id` alone.
Dispatch ids are allocated per command processor and a multi-XCD part has one
per XCD, so ids collide across dies.

## Tuning lives in the architecture config

A model contains no numbers. Every latency, rate and capacity comes from the
`timing` block of the architecture config file, so retargeting a model to a
different part is a config edit rather than a rebuild.

```json
{
  "vm": { "gpu": { "device": { "...": "..." } } },
  "timing": {
    "model": "leaky",
    "clock_mhz": 2100,
    "machine": {
      "compute_units": 256,
      "simd_lanes": 16,
      "vector_alu":      { "issue_cycles": 1, "ports": 4 },
      "matrix_multiply": { "issue_cycles": 4, "ports": 4 },
      "scalar_alu":      { "issue_cycles": 1, "ports": 1 },
      "none":            { "ports": 1 },
      "lds_read":        { "issue_cycles": 1 },
      "global":          { "bytes_per_cycle": 3200 },
      "lds":             { "bytes_per_cycle": 256 },
      "dispatch_latency_cycles": 2000
    },
    "model_config": { }
  }
}
```

An entry can carry `issue_cycles` (what one instruction of that *class* costs)
or `ports` (how many of that *functional unit* one compute unit has), and the
two namespaces overlap where a class shares a unit's name. `none.ports` is the
front-end issue slot every instruction occupies whatever unit it goes to, which
is what keeps a wait or an `s_nop` from being free.

`machine` is the shared vocabulary — anything describing the part itself, which
any model may read. `model_config` is private to the named model and is passed
through untouched, so one model's invented knob never looks like a property of
the hardware.

The block is read in a second, schema-free pass over the config file, the same
way `plugins` and `sinks` are. It has to be: the typed load runs with unexpected
fields skipped, so a `timing` block on the schema-typed path would be dropped in
silence and the model would run entirely on fallbacks with nothing to say it had.

Two numbers in that block are worth naming, because both are easy to get
plausibly wrong. `compute_units` is the count the device *advertises* to the
guest, which on the CDNA configs is not the count the topology instantiates —
picking the wrong one is a silent 5-12% error. And a bandwidth is better taken
as a measured floor than derived from `mem_width x mem_clk_max`: a peak rate
rewrites every bandwidth-heavy kernel, and a config carrying one is
indistinguishable from a config that was correlated against hardware.

Values reach a model through `TimingHost`:

```cpp
std::uint64_t units = host.tune("compute_units", /*pessimistic=*/1);
double        rate  = host.tune_real("global.bytes_per_cycle", /*pessimistic=*/1.0);
std::uint64_t alu   = host.class_issue_cycles(InstClass::VectorAlu);
```

`tune()` is for counts and cycles, `tune_real()` for rates — reading a
bytes-per-cycle through `tune()` truncates it.

There are no sentinel values. A parameter is either named in the config or it is
not; `0` means zero, not "unbounded" or "unmodelled" or "let the simulator
decide". Overloading zero is how a config ends up describing a machine part that
is free, infinitely wide, or invisible depending on which field was touched. A
parameter the config omits is not zero either — it is that parameter's
pessimistic value, and the run says so.

Note that `clock_mhz` never changes a cycle count. It converts cycles into the
time the guest reads, and it is the engine clock the driver advertises. Changing
it rescales every reported duration while leaving every modelled cycle
identical, which reads like a model change and is not one.

## Fail slow

A timing model's most dangerous failure is not being wrong. It is being wrong in
both directions, so that per-benchmark errors cancel and the aggregate looks
healthy. An effect a model does not know about is charged **zero**, which always
makes the model read fast, and on a benchmark dominated by something else looks
exactly like accuracy.

This is not a hypothetical. In the GPU simulator literature, treating one
address computation as free — kernarg base addresses "provided at no cost" — is
a 1.85x runtime error on a single benchmark; the same work is 1.6x *too slow* on
another benchmark at the same time, and the authors state plainly that the two
cannot be reconciled with a fudge factor. Zero is not a neutral default. It is
an unbounded optimistic bias that scales with how often the event occurs.

The API therefore has no silent defaults. Three rules, enforced rather than
recommended:

**Unknown is a class, and it is expensive.** When the observer cannot classify
an opcode, `effective_class` is `InstClass::Unknown` — not a guess, and not a
catch-all that happens to occupy no functional unit. `Unknown` is the *zero
value* of the enum, so a default-constructed event is the expensive case rather
than a free one, and it contends for the vector pipe like everything else. A
model charges it `class_issue_cycles(InstClass::Unknown)`, which resolves to the
largest issue cost the config gives *any* class. An opcode nobody has classified
makes a run read slow and look suspicious, never fast.

**A missing config value resolves pessimistically.** The second argument to
`tune()` is the slowest reasonable value for that parameter, not the typical
one. The host records every parameter a model asks for, together with whether
the config actually named it, and prints the lot at shutdown. A model that asks
for something the config forgot runs slow and says so, in the run rather than
only in the numbers.

**Anything not modelled is declared.** `host.note_unmodeled("lds bank
conflicts")` adds to a ledger the host prints at shutdown. Declare a structural
gap — a whole subsystem the model does not have — once at construction; declare
a data-dependent gap every time it happens, so the count measures how much of
the run it touched. A run whose ledger is non-empty is not a validated run,
which is the point: silence has to mean coverage, or a report is not evidence.

The sample model's constructor is worth reading as the intended register. It
declares five structural gaps in five lines, and charges every one of them the
expensive way: no cache, so all traffic is billed at the DRAM rate; no coalescer,
so every lane is billed its own bytes; an access whose addresses could not be
recovered is billed as a full-width global transfer.

## Two kinds of model

The same event stream serves both ends of the range.

A **leaky bucket** model ignores ordering entirely. It pours work into
per-resource buckets — issue cycles per functional unit, global bytes, LDS bytes
— and at `on_dispatch_end` takes the fullest bucket divided by its drain rate,
floored by a dispatch latency. It reads `effective_class`, `wave_lanes` and the
memory byte counts, and `addresses_known`. `vm/timing/models/leaky/` is exactly
this, and is the reference for the smallest useful model.

One subtlety that bites every bucket model: the drain width is bounded by the
*dispatch*, not by the part. One workgroup does not run on two compute units, so
a grid of four workgroups drains at four compute units' width however many the
config declares. Divide by the whole machine regardless and a small grid comes
out faster than a single workgroup's work can possibly be retired — which reads
as a very fast kernel rather than as a modelling error. The corollary is worth
stating too, because it looks like a bug the first time you see it: widening a
grid that does not yet fill the machine should *not* change the reported time.

A **cycle-accurate** model uses the rest: register ranges to build a scoreboard,
lane addresses to model coalescing and cache state, wait thresholds to model
`s_waitcnt` stalls, and `on_barrier` for the spread between wavefronts. It keeps
a per-compute-unit cycle timeline and issues into modelled pipelines.

The difference shows up in what a guest can read *during* a kernel. A model with
a timeline advances `device_cycles()` as instructions are observed, so
`s_memtime` either side of a loop returns a useful delta. A bucket model has no
timeline: it publishes the whole cost at `on_dispatch_end`, so two in-kernel
reads legitimately return the same value and only the host-side event pair sees
anything. Neither is wrong, but a model author should decide which one they are
building, because a kernel that spins until the counter changes will hang under
the second.

Two things constrain both. First, a model here **observes**; it does not decide.
It sees each decision after the functional simulator made it, so it cannot model
the effect of its own timing on execution — a contended atomic whose winner
would differ under the modelled clock resolves the way the functional run
resolved it. That bounds achievable accuracy on workloads whose behaviour
depends on timing. It is a property of the extension point, not of any model,
and it is why nothing built on this interface should be called cycle-accurate
without saying what was validated. Everything else — throughput, latency
exposure, occupancy, memory-system behaviour over a fixed access stream — is
reachable.

Second, note what a heavily hand-validated cycle-level model of a *fully
documented, small* SoC achieves: roughly 13-17% mean absolute error on runtime
and about 20% on microarchitectural counters. That is the ceiling the literature
reports for the well-understood case. A model that claims materially better on a
part with no public microarchitecture description is reporting a fit, not an
accuracy.

## Writing one

A model ships as `librocjitsu_timing_<name>.so` and exports three loader
entry points:

```cpp
class MyModel final : public rocjitsu::timing::TimingModel { /* ... */ };

ROCJITSU_DEFINE_TIMING_MODEL(MyModel, "mymodel", /*model_config schema=*/"{}")
```

The macro emits `rocjitsu_timing_model_metadata`, `_create` and `_destroy`, and
stamps `kTimingAbiVersion` into the metadata; a linker version script makes
those three the only exported symbols. The loader refuses a model whose ABI
version does not match, which turns a stale `.so` left in the library path from
an unexplained crash into a diagnostic.

Select the model with `"timing": { "model": "mymodel" }` in the architecture
config.

Models are built in-tree and shipped with rocjitsu. The boundary is C-shaped and
narrow, but it carries no compatibility promise: host and models are rebuilt and
distributed together, exactly as for execution plugins.

Note the config-schema limit inherited from the plugin loader: the validator
understands only flat `string`, `number` and `boolean` entries. A model with
nested private configuration declares `"{}"` and validates the passed-through
object itself — and a model that does not validate will accept typos in silence.

## What the guest still cannot see

A model drives the device timeline, not the world. These remain host time and
will not agree with it:

- `HSA_SYSTEM_INFO_TIMESTAMP`, which ROCR serves inside the guest process from
  `clock_gettime` and never asks the driver for.
- Any CPU-side measurement the program makes itself.
- In daemon mode, one process serves several guests through one process-wide
  clock, so concurrent guests share a single timeline and each other's
  advancement. There is no per-client clock.
- Blocking *durations* — a signal wait timeout, a syncobj deadline — are still
  measured in host time. They do not corrupt a measured interval, because they
  are not timestamps a guest reads, but a guest that asks for a 10 ms timeout
  waits 10 ms of host time no matter how much simulated time has passed, so
  timeout-driven control flow can diverge from the modelled timeline.
- A passthrough or multi-GPU node still answers the clock-counters ioctl from
  the real driver for GPUs that are not emulated. rocjitsu warns once per such
  device rather than silently mixing the two.

A profiler correlating a host timestamp against a dispatch timestamp will
therefore see two unrelated timelines once a model is driving the clock.

## Testing a model

The event structs name no rocjitsu type, so a model can be driven directly from
a unit test with hand-built events — no simulator, no compiled kernel, no GPU.
Most of what is worth testing about a timing model is how stalls compose, and
constructing the exact event sequence tests that far more precisely than hoping
a kernel reaches the case. See `tests/timing/`, which includes reusable mock
models and a mock `TimingHost`.

For end-to-end validation, `tests/corpus/rocm-meter.py` runs a suite of torch and
Triton kernels and reports per-kernel device time. Run it under the emulator and
compare against the same suite recorded on the real part.

Score it on the **spread** of `log2(modelled / measured)`, not on bias and not on
correlation. Bias is a median and hides a model whose errors are large in both
directions; correlation is nearly free and says almost nothing, with published
GPU models reaching 0.97 correlation at 42-75% mean absolute error. Absolute
time is what the guest reads off the clock, so absolute time is what has to be
right.

Report the error distribution per kernel category rather than one aggregate, and
report what fraction of cases the model had no opinion about at all — a suite
where memory-only and multi-stream cases are quietly dropped flatters whatever
is left.
