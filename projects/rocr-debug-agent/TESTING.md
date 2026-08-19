<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Testing rocr-debug-agent

**rocr-debug-agent (the ROCdebug-agent) is the HSA tool library that reports on
faulting AMD GPU wavefronts** - dumping registers, LDS, and disassembly with
source correlation when a HIP kernel traps or faults. It is loaded via
`HSA_TOOLS_LIB` in front of a running application and built on top of
**amd-dbgapi** and the ROCr runtime (`hsa-runtime64`). The goal is to keep its
wave-dump output correct across supported GPU architectures as amd-dbgapi,
ROCr, and HIP evolve underneath it.

**Testing today is a single, GPU-bound layer.** Every current test builds and
runs a real HIP kernel with the debug agent attached, then inspects its
output; there is no cheaper, GPU-free layer to catch a regression before
paying for a full build-and-run cycle - see
[Use layered validation](#use-layered-validation).

**Testing should be accessible to contributors.** The same `ctest` / `make
test` invocation a contributor runs on their own GPU is what CI runs (via a
thin retry wrapper), so behavior can be validated locally first, with CI
providing GPU architectures a contributor may not have on hand.

> **There's no test-writing guide beyond this document and `test/run-test.py`
> itself.** Read the comment header in `run-test.py` before adding a test.

## Testing principles

### Make tests accessible during development

Every fix or feature should come with a test that verifies the specific
behavior being changed. There is no in-process, GPU-free tier to iterate on
quickly: every test requires a GPU-enabled build and a real device, so a
contributor without local GPU access cannot fully validate a change before CI
does. `make test` at least builds and runs the full suite from a locally
built tree, so no CI-only setup is required to reproduce it.

### Use layered validation

There are three tiers of validation to consider for a change: unit tests,
functional tests, and performance tests. Today, only the functional tier
exists for rocr-debug-agent:

- **Unit tests** - **do not exist.** `src/code_object.cpp`'s ELF/DWARF
  handling and the option/`%`-token parsing in `src/debug_agent.cpp` are
  self-contained enough to unit test without a GPU, but no such tests or
  harness exists yet.
- **Functional tests** - the `test/` suite: the `rocm-debug-agent-test` binary
  (8 HIP scenarios, selected by CLI argument `0`-`7`) driven by
  `test/run-test.py` (15 logical checks × 3 `HIP_ENABLE_DEFERRED_LOADING`
  settings = 45 runs per invocation). This is where all current testing
  happens.
- **Performance tests** - **do not exist.** No perf harness, baseline, or
  regression threshold exists today.

| ID | Workload | Purpose |
|----|----------|---------|
| 0 | `vector_add_normal.cpp` | Fault-free baseline |
| 1 | `vector_add_assert_trap.cpp` | `s_trap` / `ASSERT_TRAP` wave dump |
| 2 | `vector_add_memory_fault.cpp` | Null-pointer write -> `MEMORY_VIOLATION` |
| 3 | `snapshot_objfile_on_load.cpp` | Code object snapshotted at load survives the host corrupting the original buffer |
| 4 | `save_code_objects.cpp` | Two module loads, for `--save-code-objects` variants |
| 5 | `print_all_waves.cpp` | Multi-wave fault, for `--all` |
| 6 | `sigquit.cpp` | Long-running kernel + external `SIGQUIT` |
| 7 | `vector_add_assert_trap_no_debug_info.cpp` | Same trap, built without `-ggdb`, for debug-info comparison |

### Scale coverage to available resources

The suite runs on whichever GPU architectures CI provisions for a given run,
with a periodic broader sweep across additional architectures to catch
regressions a single run's coverage might miss. There is no host-OS or
toolchain matrix to consider - the suite currently only targets Linux HIP
builds.

### Use static analysis for mechanical checks

Formatting and mechanical checks run as pre-commit hooks. rocr-debug-agent
has no project-local pre-commit configuration of its own; it relies entirely
on the repo-wide `.pre-commit-config.yaml` (`clang-format`, `gersemi`,
`black`, whitespace/YAML checks) plus the project's own `.clang-format`.

### Add reliable tests to required CI

CI invokes the suite through a retry wrapper -
`.github/scripts/test_rocr-debug-agent.py` in this project's source tree -
which gets installed alongside the test binaries into `tests/rocm-debug-agent/`
once rocr-debug-agent is built as part of a full TheRock ROCm stack; it isn't
something you'll find by building this project in isolation. It retries the
whole `run-test.py` invocation up to 3 times with backoff before failing the
job. This tolerates transient flakiness, but there is no mechanism today to
track a specific known-flaky or known-broken test explicitly: a blanket retry
can mask a real intermittent regression as readily as a real flake, and a
deterministically failing test still burns all 3 attempts before the job
reports failure.

## Testing changes to rocr-debug-agent

### Host-side / option-parsing changes

There is no host-only test tier to cover this cleanly (see
[Use layered validation](#use-layered-validation)): a change to option
parsing, `%`-token filename expansion, or ELF/DWARF handling in `src/` must
currently be exercised indirectly through a full GPU scenario in `test/`,
even though the logic itself doesn't need a GPU.

### AMDGPU and ROCm-specific changes

Add or extend a scenario under `test/`:

1. Add (or extend) a `.cpp` HIP workload in `test/`, following an existing
   file (e.g. `vector_add_memory_fault.cpp`) as a template, and add new files
   to `MAIN_SOURCES` in `test/CMakeLists.txt`.
2. Wire it into the `switch` statement in `test/debug_agent_test.cpp`'s
   `main()` with a new scenario ID and a `Run<Name>Test()` wrapper that
   iterates `hipGetDeviceCount()` and calls `hipDeviceReset()` between
   devices.
3. Add a corresponding entry to `TEST_DEFINITIONS` in `test/run-test.py`:
   regex `patterns` matched against stdout/stderr for simple checks, or a
   custom `function` handler for cases needing filesystem checks or
   timing-sensitive signal handling (see `test_save_code_objects` or
   `test_sigquit` for examples).

This is the only test tier that exists today, so essentially every functional
change goes through it. New tests run automatically under all three
`HIP_ENABLE_DEFERRED_LOADING` settings via the outer harness loop; if the
scenario is sensitive to DWARF source-file paths, compile it with the same
`-fdebug-prefix-map` / `-fdebug-compilation-dir` flags already applied to
`rocm-debug-agent-test`.

### Performance-sensitive changes

There is no perf test tier to add to (see
[Use layered validation](#use-layered-validation)). Validate
performance-sensitive changes - for example wave-dump latency at high wave
counts - manually, on the same hardware, before and after, until a harness
exists.

## Testing rocr-debug-agent against ROCm

### Build through a full ROCm stack

rocr-debug-agent links against amd-dbgapi and the ROCr runtime, and its test
suite additionally needs a HIP compiler to precompile two on-device `.hipfb`
blobs ahead of time. In practice, this means building and testing
rocr-debug-agent as part of a full ROCm build via TheRock rather than
compiling it in isolation.

### Keep test environments reproducible

The harness sets `HSA_TOOLS_LIB` and extends `LD_LIBRARY_PATH` itself (see
`test/run-test.py`), and sweeps `HIP_ENABLE_DEFERRED_LOADING` across three
values itself, rather than depending on the ambient environment already
being set correctly. A correct ROCm install plus the built library alongside
the test binaries is generally all that's needed.

### Run the same tests locally and in CI

For local iteration from a rocr-debug-agent build tree, `make test` (`ctest`)
is the entry point - it invokes `test/run-test.py` against the build's
`rocm-debug-agent-test` binary and runs the full suite:

```bash
make test
```

To reproduce what CI runs against a fully assembled ROCm/TheRock stack, use
the retry wrapper rather than invoking `rocm-debug-agent-test` directly -
manually setting `HSA_TOOLS_LIB` and `LD_LIBRARY_PATH` for a single scenario
is error-prone and easy to get subtly wrong:

```bash
OUTPUT_ARTIFACTS_DIR=<path-to-built-rocm-tree> tests/rocm-debug-agent/test_rocr-debug-agent.py
# or, if ROCM_PATH already points at that tree:
tests/rocm-debug-agent/test_rocr-debug-agent.py --try-rocm-path
```

This locates `rocm-debug-agent-test` and `run-test.py` under
`tests/rocm-debug-agent/` in that tree, disables core dumps, and retries the
suite the same way CI does.

## Validating rocr-debug-agent on hardware and in CI

### Test on supported GPU hardware

The suite iterates every device `hipGetDeviceCount()` reports, calling
`hipDeviceReset()` between devices; it does not gate on whether a given
device is a supported architecture. On mixed or partially unsupported
hardware, expect a real test failure on the unsupported device rather than a
clean skip.

### When tests run

Tests run automatically whenever a pull request touches rocr-debug-agent or
one of its runtime dependencies, against whichever GPU architectures CI has
provisioned for that run. A separate, periodic run widens coverage to
additional architectures to catch regressions a single PR wouldn't exercise.
Treat the CI configuration itself as the source of truth for current
triggers and hardware coverage - both change independently of this document.

### Read results and triage failures

`run-test.py` prints a `PASS` / `FAIL` / `UNSUPPORTED` verdict per check to
stdout, plus a final tally, and writes a `run-test.log` diagnostic log with
captured stdout/stderr for each subprocess run in the working directory.
There is no machine-readable summary format and no concept of an
expected/known failure for a specific check - a failing check is always a
hard `FAIL`, after CI's blanket 3x retry is exhausted.
