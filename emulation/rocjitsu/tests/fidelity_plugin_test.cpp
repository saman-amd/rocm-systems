// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file Tests for the semantic-fidelity plugin.
///
/// The scenarios below run real kernels through the emulator rather than
/// calling the classifier directly, so they exercise the path a CI consumer
/// actually depends on: fidelity decided per opcode, propagated through
/// registers, and surfaced in the emitted report.

#include "rocjitsu/vm/plugins/fidelity/plugin.h"

#include "aql_queue.h"
#include "embedded_schema.h"
#include "hsa/AMDHSAKernelDescriptor.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/plugins/throughput/plugin.h"
#include "rocjitsu/vm/soc.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <format>
#include <string>
#include <string_view>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::plugins::fidelity;

constexpr uint32_t sopp(uint32_t op, uint16_t simm16 = 0) {
  return 0xBF800000u | (op << 16) | simm16;
}
constexpr uint32_t S_NOP = sopp(0);
constexpr uint32_t S_ENDPGM = sopp(1);
constexpr uint32_t S_BARRIER = sopp(10);

constexpr uint32_t vop1_encode(uint32_t opcode, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((opcode & 0xFF) << 9) | (src0 & 0x1FF);
}

constexpr uint32_t kVMovB32 = 1;
constexpr uint32_t kVRcpF32 = 34;
constexpr uint32_t kVSqrtF32 = 39;
/// Inline constant 1.0f, so the approximations get a defined input.
constexpr uint32_t kSrcOne = 242;

/// Minimal single-CU SoC that can run a kernel with a fidelity plugin attached.
struct FidelityFixture {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *mem = nullptr;
  std::shared_ptr<ExecutionPluginGroup> plugin_group;
  StringSink *sink = nullptr;

  FidelityFixture() {
    std::string json = std::format(R"({{
      "max_ticks":10000,"num_threads":1,"exec_mode":"functional",
      "vm":{{"arch":"{}","gpu":{{"device":{{"wave_front_size":{}}}}}}},
      "topology":{{"root":{{"name":"soc","type":"soc","children":[
        {{"name":"vram","type":"gpu_memory"}},
        {{"name":"xcd0","type":"xcd","children":[
          {{"name":"l2","type":"l2_cache"}},
          {{"name":"cp","type":"command_processor"}},
          {{"name":"se0","type":"shader_engine","children":[
            {{"name":"cu[0:1]","type":"compute_unit","config":[
              {{"key":"num_wf_slots","value":"1"}},
              {{"key":"sgprs_per_wf","value":"104"}},
              {{"key":"vgprs_per_wf","value":"256"}},
              {{"key":"lds_size_kb","value":"64"}}
            ]}}
          ]}}
        ]}}
      ]}},"links":[
        {{"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2}},
        {{"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}}
      ]}}}}
    )",
                                   "cdna4", 64);
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    soc = loaded.soc();
    mem = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->create();

    PluginSinkConfig sink_config;
    sink = &sink_config.emplace<StringSink>();
    plugin_group = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  }

  /// Attach the fidelity plugin, optionally alongside the throughput plugin so
  /// the two reports can be compared on the same run.
  void start(bool with_throughput = false) {
    EXPECT_TRUE(plugin_group->add(std::make_unique<FidelityPlugin>()));
    if (with_throughput) {
      EXPECT_TRUE(plugin_group->add(std::make_unique<plugins::throughput::ThroughputPlugin>()));
    }
    soc->set_plugin_group(plugin_group);
    plugin_group->onInit();
  }

  uint64_t write_kernel(uint64_t addr, const uint32_t *code, size_t num_words) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    mem->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    mem->load_image(reinterpret_cast<const uint8_t *>(code), num_words * 4,
                    addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  void run_kernel(const uint32_t *code, size_t num_words) {
    uint64_t ko = write_kernel(0x1000, code, num_words);
    test::AqlQueue queue(mem, cp());
    queue.dispatch(ko, 64, 64);
    for (uint32_t i = 0; i < 100000 && engine->step(); ++i) {
    }
  }

  amdgpu::CommandProcessor *cp() { return soc->xcd(0)->command_processor(); }

  std::string report() {
    plugin_group->onShutdown();
    return sink->str();
  }
};

/// Extract an integer field from a JSONL record.
uint64_t field(std::string_view record, std::string_view key) {
  const std::string needle = std::format("\"{}\":", key);
  const size_t pos = record.find(needle);
  EXPECT_NE(pos, std::string_view::npos) << "missing field " << key;
  if (pos == std::string_view::npos)
    return 0;
  return std::stoull(std::string(record.substr(pos + needle.size(), 20)));
}

/// Return the line of `text` emitted by `schema` with the given record type.
///
/// Selecting on schema matters because several plugins may be attached and
/// they all emit "dispatch" and "summary" records.
std::string line_for(std::string_view text, std::string_view record,
                     std::string_view schema = "rocjitsu.fidelity.v1") {
  const std::string schema_needle = std::format("\"schema\":\"{}\"", schema);
  const std::string record_needle = std::format("\"record\":\"{}\"", record);
  size_t start = 0;
  while (start < text.size()) {
    const size_t end = text.find('\n', start);
    const std::string_view line = text.substr(start, end - start);
    if (line.find(schema_needle) != std::string_view::npos &&
        line.find(record_needle) != std::string_view::npos)
      return std::string(line);
    if (end == std::string_view::npos)
      break;
    start = end + 1;
  }
  return {};
}

} // namespace

/// Classification is keyed on semantics, not on the throughput family that
/// already exists, so opcodes in the same family can differ in fidelity.
TEST(SemanticFidelityTest, ClassifiesOpcodesBySemanticsNotFamily) {
  EXPECT_EQ(classify("v_mov_b32"), Fidelity::kExact);
  EXPECT_EQ(classify("v_add_f32"), Fidelity::kExact);
  EXPECT_EQ(classify("s_nop"), Fidelity::kExact);
  EXPECT_EQ(classify("s_barrier"), Fidelity::kExact);

  EXPECT_EQ(classify("v_rcp_f32"), Fidelity::kApproximate);
  EXPECT_EQ(classify("v_sqrt_f32"), Fidelity::kApproximate);
  EXPECT_EQ(classify("v_sin_f32"), Fidelity::kApproximate);
  EXPECT_EQ(inexactness_of("v_rcp_f32"), Inexactness::kNumericApproximation);

  EXPECT_EQ(classify("s_denorm_mode"), Fidelity::kApproximate);
  EXPECT_EQ(inexactness_of("s_denorm_mode"), Inexactness::kElidedSemantics);
  EXPECT_EQ(inexactness_of("v_mov_b32"), Inexactness::kNone);
}

/// Encoding suffixes select an operand form, not a different execute body.
TEST(SemanticFidelityTest, IgnoresEncodingSuffixes) {
  EXPECT_EQ(classify("v_rcp_f32_e32"), Fidelity::kApproximate);
  EXPECT_EQ(classify("v_rcp_f32_e64"), Fidelity::kApproximate);
  EXPECT_EQ(classify("v_sqrt_f32_dpp"), Fidelity::kApproximate);
  EXPECT_EQ(classify("v_mov_b32_e32"), Fidelity::kExact);
}

/// A kernel of only exact opcodes must be reported as trustworthy, otherwise
/// the report would be useless noise on well-behaved code.
TEST(FidelityPluginTest, ExactKernelIsReportedNumericallyValidated) {
  FidelityFixture f;
  f.start();
  const uint32_t code[] = {vop1_encode(kVMovB32, 0, kSrcOne), S_NOP, S_ENDPGM};
  f.run_kernel(code, 3);

  const std::string dispatch = line_for(f.report(), "dispatch");
  ASSERT_FALSE(dispatch.empty());
  EXPECT_NE(dispatch.find("\"numerically_validated\":true"), std::string::npos) << dispatch;
  EXPECT_EQ(field(dispatch, "approximate"), 0u);
  EXPECT_EQ(field(dispatch, "unsupported"), 0u);
  EXPECT_GT(field(dispatch, "exact"), 0u);
}

/// The reported gap: approximate opcodes must be distinguishable from exact
/// ones even though both are vector-ALU instructions.
TEST(FidelityPluginTest, SeparatesApproximateFromExactWithinOneFamily) {
  FidelityFixture f;
  f.start(/*with_throughput=*/true);
  const uint32_t code[] = {
      vop1_encode(kVMovB32, 0, kSrcOne),
      vop1_encode(kVRcpF32, 1, kSrcOne),
      vop1_encode(kVSqrtF32, 2, kSrcOne),
      S_BARRIER,
      S_NOP,
      S_ENDPGM,
  };
  f.run_kernel(code, 6);
  const std::string text = f.report();
  const std::string dispatch = line_for(text, "dispatch");
  ASSERT_FALSE(dispatch.empty());

  // Throughput counts all three vector-ALU ops the same because family is a
  // microarchitectural axis; fidelity is the axis that separates them.
  const std::string throughput = line_for(text, "dispatch", "rocjitsu.throughput.v2");
  ASSERT_FALSE(throughput.empty());
  EXPECT_EQ(field(throughput, "instructions"), 0u) << "expected scalar family first";
  EXPECT_NE(throughput.find("\"vector\":{\"instructions\":3"), std::string::npos) << throughput;

  EXPECT_EQ(field(dispatch, "approximate"), 2u) << dispatch;
  EXPECT_NE(dispatch.find("\"numerically_validated\":false"), std::string::npos);
  EXPECT_NE(dispatch.find("v_rcp_f32"), std::string::npos) << dispatch;
  EXPECT_NE(dispatch.find("v_sqrt_f32"), std::string::npos) << dispatch;
  EXPECT_NE(dispatch.find("\"reason\":\"numeric_approximation\""), std::string::npos);
}

/// An approximate value that is only produced, never consumed by a branch,
/// address or barrier, must not be reported as having reached a sink.
TEST(FidelityPluginTest, ApproximateValueWithoutConsumerReachesNoSink) {
  FidelityFixture f;
  f.start();
  const uint32_t code[] = {vop1_encode(kVRcpF32, 1, kSrcOne), S_ENDPGM};
  f.run_kernel(code, 2);

  const std::string dispatch = line_for(f.report(), "dispatch");
  ASSERT_FALSE(dispatch.empty());
  EXPECT_EQ(field(dispatch, "approximate"), 1u);
  EXPECT_EQ(field(dispatch, "control_flow"), 0u);
  EXPECT_EQ(field(dispatch, "addressing"), 0u);
  EXPECT_EQ(field(dispatch, "synchronization"), 0u);
}

/// Taint must survive an intermediate exact instruction: consuming an
/// approximate value makes the consumer's own result approximate too.
TEST(FidelityPluginTest, TaintPropagatesThroughExactInstructions) {
  FidelityFixture f;
  f.start();
  const uint32_t code[] = {
      vop1_encode(kVRcpF32, 1, kSrcOne),
      // v1 is approximate; copying it must carry that to v2.
      vop1_encode(kVMovB32, 2, /*src0=*/256 + 1),
      S_ENDPGM,
  };
  f.run_kernel(code, 3);

  const std::string dispatch = line_for(f.report(), "dispatch");
  ASSERT_FALSE(dispatch.empty());
  EXPECT_NE(dispatch.find("\"numerically_validated\":false"), std::string::npos) << dispatch;
}

/// Writing an exact result over a tainted register must clear the taint,
/// otherwise a long-running kernel would drift to reporting everything as
/// suspect and the report would lose its value.
TEST(FidelityPluginTest, ExactOverwriteClearsTaint) {
  FidelityFixture f;
  f.start();
  const uint32_t code[] = {
      vop1_encode(kVRcpF32, 1, kSrcOne),
      vop1_encode(kVMovB32, 1, kSrcOne),
      S_ENDPGM,
  };
  f.run_kernel(code, 3);

  const std::string dispatch = line_for(f.report(), "dispatch");
  ASSERT_FALSE(dispatch.empty());
  EXPECT_EQ(field(dispatch, "control_flow"), 0u);
  EXPECT_EQ(field(dispatch, "addressing"), 0u);
  EXPECT_EQ(field(dispatch, "synchronization"), 0u);
}

/// The question the report exists to answer: an approximate value that forms a
/// memory address is not merely imprecise, it may have made the emulated run
/// touch different memory than hardware would.
TEST(FidelityPluginTest, ApproximateValueReachingAnAddressIsReported) {
  FidelityFixture f;
  f.start();
  // global_load_dword v5, v[2:3], off -- the address operand is v[2:3].
  constexpr uint32_t kGlobalLoadLo = 0xDC508000u;
  constexpr uint32_t kGlobalLoadHi = 0x057F0002u;
  const uint32_t code[] = {
      vop1_encode(kVRcpF32, /*vdst=*/2, kSrcOne),
      vop1_encode(kVRcpF32, /*vdst=*/3, kSrcOne),
      kGlobalLoadLo,
      kGlobalLoadHi,
      S_ENDPGM,
  };
  f.run_kernel(code, 5);

  const std::string dispatch = line_for(f.report(), "dispatch");
  ASSERT_FALSE(dispatch.empty());
  EXPECT_EQ(field(dispatch, "addressing"), 1u) << dispatch;
  EXPECT_EQ(field(dispatch, "control_flow"), 0u) << dispatch;
  EXPECT_NE(dispatch.find("\"numerically_validated\":false"), std::string::npos);
}

/// The end-of-run summary must aggregate every dispatch, since CI consumes one
/// verdict per run rather than per dispatch.
TEST(FidelityPluginTest, SummaryAggregatesAcrossDispatches) {
  FidelityFixture f;
  f.start();
  const uint32_t code[] = {vop1_encode(kVRcpF32, 1, kSrcOne), S_ENDPGM};
  f.run_kernel(code, 2);

  const std::string text = f.report();
  const std::string summary = line_for(text, "summary");
  ASSERT_FALSE(summary.empty()) << text;
  EXPECT_EQ(field(summary, "dispatches"), 1u);
  EXPECT_EQ(field(summary, "approximate"), 1u);
  EXPECT_NE(summary.find("\"numerically_validated\":false"), std::string::npos);
}