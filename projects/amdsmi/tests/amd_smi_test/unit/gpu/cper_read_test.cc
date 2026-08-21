/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Regression tests for the CPER read path via
// amdsmi_get_gpu_cper_entries_by_path(); no GPU required.
// ROCM-25398: a zero-byte CPER node must not abort the process.
// ROCM-25954: an empty ring returns SUCCESS with zero entries, not an error.
//
// Also covers the structural bounds of the parser: a crafted record whose
// sec_cnt, sec_offset, or reg_arr_size points past the buffer must be rejected
// per section rather than read or written through (CWE-125).

#include <gtest/gtest.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_cper.h"
#include "amd_smi/impl/amd_smi_cper_testing.h"

namespace {

// 4 MiB ring + 12 B header, matching the st_size reported in the ROCM-25954
// field report.
constexpr off_t kRingCapacity = 4194316;

// Runs the CPER read path against a file. Out-params report the final
// entry_count and buf_size.
amdsmi_status_t CallCperByPath(const char* path, uint64_t* out_entry_count = nullptr,
                               uint64_t* out_buf_size = nullptr) {
  std::vector<char> cper_data(4096, 0);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(8, nullptr);
  uint64_t buf_size = cper_data.size();
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      path, 0xFFFFFFFF, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count, &cursor,
      /*product_serial=*/0);

  if (out_entry_count) *out_entry_count = entry_count;
  if (out_buf_size) *out_buf_size = buf_size;
  return status;
}

// Restores the production read() seam however a test exits.
struct CperReadFnGuard {
  ~CperReadFnGuard() { cper_set_read_fn_for_testing(nullptr); }
};

ssize_t FakeReadZero(int, void*, size_t) { return 0; }  // empty ring

ssize_t FakeReadPartial(int, void* buf, size_t) {  // short read of non-record bytes
  std::memset(buf, 0, 100);
  return 100;
}

ssize_t FakeReadError(int, void*, size_t) {  // I/O failure
  errno = EIO;
  return -1;
}

// Sparse regular file: advertises `size` via st_size at no disk cost and passes
// the S_ISREG guard, matching the debugfs node's "capacity in st_size" shape.
// Fatal-asserts on setup failure, returning the path via out_path.
void MakeSparseFile(off_t size, std::string* out_path) {
  std::string tmpl = "/tmp/amdsmi_cper_cap_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  int rc = ftruncate(fd, size);
  close(fd);
  if (rc != 0) {
    unlink(tmpl.c_str());
    FAIL() << "failed to size temp file";
  }
  *out_path = tmpl;
}

// Minimal single-record CPER blob the parser accepts: "CPER" signature,
// 0xFFFFFFFF terminator, record_length == header size, severity 0 (matched by a
// full mask).
std::vector<char> MakeOneRecordBlob() {
  amdsmi_cper_hdr_t hdr{};
  std::memcpy(hdr.signature, "CPER", 4);
  hdr.signature_end = 0xFFFFFFFF;
  hdr.error_severity = AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED;
  hdr.record_length = sizeof(hdr);
  std::vector<char> blob(sizeof(hdr));
  std::memcpy(blob.data(), &hdr, sizeof(hdr));
  return blob;
}

// `count` identical single-record blobs concatenated.
std::vector<char> MakeRecordsBlob(size_t count) {
  std::vector<char> one = MakeOneRecordBlob();
  std::vector<char> blob;
  for (size_t i = 0; i < count; ++i) {
    blob.insert(blob.end(), one.begin(), one.end());
  }
  return blob;
}

// Writes bytes to a fresh temp file, returning its path via out_path.
// Fatal-asserts on setup failure.
void WriteTempFile(const std::vector<char>& bytes, std::string* out_path) {
  std::string tmpl = "/tmp/amdsmi_cper_rec_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  ssize_t written = write(fd, bytes.data(), bytes.size());
  close(fd);
  if (written != static_cast<ssize_t>(bytes.size())) {
    unlink(tmpl.c_str());
    FAIL() << "failed to write temp file";
  }
  *out_path = tmpl;
}

// Like CallCperByPath but with a caller-controlled buffer byte size and header
// slot count, to exercise the buffer-exhaustion return paths. out_cursor, when
// provided, reports the returned cursor.
amdsmi_status_t CallCperSized(const char* path, uint64_t buf_bytes, uint64_t slots,
                              uint64_t* out_entry_count, uint64_t* out_buf_size,
                              uint64_t* out_cursor = nullptr) {
  std::vector<char> cper_data(buf_bytes, 0);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(slots, nullptr);
  uint64_t buf_size = buf_bytes;
  uint64_t entry_count = slots;
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      path, 0xFFFFFFFF, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count, &cursor,
      /*product_serial=*/0);

  if (out_entry_count) *out_entry_count = entry_count;
  if (out_buf_size) *out_buf_size = buf_size;
  if (out_cursor) *out_cursor = cursor;
  return status;
}

}  // namespace

// Zero-size regular file: st_size == 0, so read(fd, buf, 0) returns 0 trivially.
// Hits the same empty-ring success branch as the field case (large st_size,
// read() returns 0) and must not abort the process.
TEST(GpuUnit, CperReadZeroSizeFile) {
  std::string tmpl = "/tmp/amdsmi_cper_zero_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  close(fd);  // leave it empty -> st_size == 0

  uint64_t entry_count = 99;  // sentinel; the call must overwrite both to 0
  uint64_t buf_size = 99;
  amdsmi_status_t status = CallCperByPath(tmpl.c_str(), &entry_count, &buf_size);
  unlink(tmpl.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Characterizes how a regular/CI filesystem differs from the debugfs target.
// ftruncate to a non-zero size with no payload gives st_size == 4096; a regular
// filesystem then returns 4096 zero bytes on read() (a full read of the hole),
// not 0 like the empty debugfs ring. Those zero bytes hold no CPER signature, so
// the result is still SUCCESS with no records. This is why the empty-ring shape
// (read() == 0 while st_size advertises ring capacity) needs the injectable read
// seam in CperEmptyRingAdvertisedCapacityShortRead to reproduce faithfully, and
// why CperReadZeroSizeFile pins the st_size == 0 corner with a real read().
TEST(GpuUnit, CperNonZeroFileRealReadHasNoRecords) {
  std::string path;
  MakeSparseFile(4096, &path);  // st_size == 4096, no payload written
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 99;  // sentinels; the call must overwrite both to 0
  uint64_t buf_size = 99;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Missing path -> NOT_SUPPORTED (stat() fails), no crash. Create then remove a
// temp file so the path is guaranteed absent (no hardcoded path another process
// might have created).
TEST(GpuUnit, CperReadMissingFile) {
  std::string tmpl = "/tmp/amdsmi_cper_missing_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  close(fd);
  unlink(tmpl.c_str());

  amdsmi_status_t status = CallCperByPath(tmpl.c_str());
  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
}

// Happy path: a well-formed single-record file parses to one entry. Guards the
// read path against regressions in the empty/error handling around it.
TEST(GpuUnit, CperParsesSingleRecord) {
  std::string path;
  WriteTempFile(MakeOneRecordBlob(), &path);
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 1u);
  EXPECT_GT(buf_size, 0u);
}

// Faithful ROCM-25954 repro: st_size advertises the 4 MiB ring capacity while
// read() returns 0 on an empty ring. Must be SUCCESS with zero entries.
TEST(GpuUnit, CperEmptyRingAdvertisedCapacityShortRead) {
  CperReadFnGuard guard;
  cper_set_read_fn_for_testing(&FakeReadZero);

  std::string path;
  MakeSparseFile(kRingCapacity, &path);
  ASSERT_FALSE(path.empty());
  uint64_t entry_count = 99;
  uint64_t buf_size = 99;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Partial read (0 < bytes_read < st_size) of non-record bytes: accepted as
// success; pin that no records are parsed and the out-params are zeroed.
TEST(GpuUnit, CperPartialReadNoRecords) {
  CperReadFnGuard guard;
  cper_set_read_fn_for_testing(&FakeReadPartial);

  std::string path;
  MakeSparseFile(kRingCapacity, &path);
  ASSERT_FALSE(path.empty());
  uint64_t entry_count = 99;
  uint64_t buf_size = 99;
  amdsmi_status_t status = CallCperByPath(path.c_str(), &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// A real read() failure (returns -1) must still surface FILE_ERROR.
TEST(GpuUnit, CperReadErrorIsFileError) {
  CperReadFnGuard guard;
  cper_set_read_fn_for_testing(&FakeReadError);

  std::string path;
  MakeSparseFile(kRingCapacity, &path);
  ASSERT_FALSE(path.empty());
  amdsmi_status_t status = CallCperByPath(path.c_str());
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_FILE_ERROR);
}

// A record larger than the caller's buffer yields OUT_OF_RESOURCES with nothing
// copied and the out-params zeroed.
TEST(GpuUnit, CperFirstRecordExceedsBufferOutOfResources) {
  std::string path;
  WriteTempFile(MakeOneRecordBlob(), &path);
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  // Non-zero buffer, but smaller than one record (sizeof(amdsmi_cper_hdr_t)).
  static_assert(sizeof(amdsmi_cper_hdr_t) > 64,
                "test assumes a 64-byte buffer is smaller than one CPER record");
  amdsmi_status_t status =
      CallCperSized(path.c_str(), /*buf_bytes=*/64, /*slots=*/8, &entry_count, &buf_size);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_OUT_OF_RESOURCES);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Two records with a buffer that fits only one: the first is copied and
// MORE_DATA is returned with the partial entry_count/buf_size.
TEST(GpuUnit, CperSecondRecordOverflowsBufferMoreData) {
  std::string path;
  WriteTempFile(MakeRecordsBlob(2), &path);
  ASSERT_FALSE(path.empty());

  const uint64_t one_record = sizeof(amdsmi_cper_hdr_t);
  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  uint64_t cursor = 0;
  // Room for one record plus a partial second, with ample header slots so the
  // byte-buffer limit (not the slot count) is what trips.
  amdsmi_status_t status =
      CallCperSized(path.c_str(), one_record + 16, /*slots=*/8, &entry_count, &buf_size, &cursor);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_MORE_DATA);
  EXPECT_EQ(entry_count, 1u);
  EXPECT_EQ(buf_size, one_record);
  EXPECT_EQ(cursor, 1u);
}

// Two records with ample byte budget but only one header slot: the slot count,
// not the byte buffer, is what trips. The first is copied and MORE_DATA is
// returned with one entry and a non-zero buf_size.
TEST(GpuUnit, CperSlotExhaustionMoreData) {
  std::string path;
  WriteTempFile(MakeRecordsBlob(2), &path);
  ASSERT_FALSE(path.empty());

  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  uint64_t cursor = 0;
  amdsmi_status_t status = CallCperSized(path.c_str(), /*buf_bytes=*/8192, /*slots=*/1,
                                         &entry_count, &buf_size, &cursor);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_MORE_DATA);
  EXPECT_EQ(entry_count, 1u);
  EXPECT_EQ(buf_size, sizeof(amdsmi_cper_hdr_t));
  EXPECT_EQ(cursor, 1u);
}

// Invalid arguments are rejected with OUT_OF_RESOURCES before any file read.
TEST(GpuUnit, CperByPathRejectsInvalidArgs) {
  std::string path;
  WriteTempFile(MakeOneRecordBlob(), &path);
  ASSERT_FALSE(path.empty());

  std::vector<char> data(256, 0);
  std::vector<amdsmi_cper_hdr_t*> hdrs(4, nullptr);
  const uint32_t mask = 0xFFFFFFFF;
  uint64_t bs = 0;
  uint64_t ec = 0;
  uint64_t cursor = 0;

  // null path: the guard zeroes the valid out-params before returning.
  bs = data.size();
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(nullptr, mask, data.data(), &bs, hdrs.data(), &ec,
                                                &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  EXPECT_EQ(bs, 0u);
  EXPECT_EQ(ec, 0u);
  // null cper_data
  bs = data.size();
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, nullptr, &bs, hdrs.data(), &ec,
                                                &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // null buf_size
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), nullptr,
                                                hdrs.data(), &ec, &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // null entry_count
  bs = data.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, hdrs.data(),
                                                nullptr, &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // zero buf_size
  bs = 0;
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, hdrs.data(),
                                                &ec, &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // zero entry_count
  bs = data.size();
  ec = 0;
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, hdrs.data(),
                                                &ec, &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // null cper_hdrs
  bs = data.size();
  ec = hdrs.size();
  cursor = 0;
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, nullptr, &ec,
                                                &cursor, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);
  // null cursor
  bs = data.size();
  ec = hdrs.size();
  EXPECT_EQ(amdsmi_get_gpu_cper_entries_by_path(path.c_str(), mask, data.data(), &bs, hdrs.data(),
                                                &ec, nullptr, 0),
            AMDSMI_STATUS_OUT_OF_RESOURCES);

  unlink(path.c_str());
}

namespace {

// Mirrors the GUID_INIT macro in amd_smi_cper.cc (mixed-endian EFI encoding) so
// the literals below diff argument-for-argument against the production ones.
constexpr amdsmi_cper_guid_t MakeGuid(uint32_t a, uint16_t b, uint16_t c, unsigned char d0,
                                      unsigned char d1, unsigned char d2, unsigned char d3,
                                      unsigned char d4, unsigned char d5, unsigned char d6,
                                      unsigned char d7) {
  return {{static_cast<unsigned char>(a), static_cast<unsigned char>(a >> 8),
           static_cast<unsigned char>(a >> 16), static_cast<unsigned char>(a >> 24),
           static_cast<unsigned char>(b), static_cast<unsigned char>(b >> 8),
           static_cast<unsigned char>(c), static_cast<unsigned char>(c >> 8), d0, d1, d2, d3, d4,
           d5, d6, d7}};
}

constexpr amdsmi_cper_guid_t kCrashdumpGuid =  // AMD_OOB_CRASHDUMP
    MakeGuid(0x32AC0C78, 0x2623, 0x48F6, 0xB0, 0xD0, 0x73, 0x65, 0x72, 0x5F, 0xD6, 0xAE);
constexpr amdsmi_cper_guid_t kNonStandardGuid =  // AMD_GPU_NONSTANDARD_ERROR
    MakeGuid(0x32AC0C78, 0x2623, 0x48F6, 0x81, 0xA2, 0xAC, 0x69, 0x17, 0x80, 0x55, 0x1D);

// A zeroed register array under ACA context (reg_ctx_type 1) decodes to this
// AFID, so any section wired up that way contributes exactly one entry.
constexpr uint16_t kAcaRegisterContext = 1;

constexpr size_t kDescTableOffset = sizeof(amdsmi_cper_hdr_t);

struct cper_sec_desc* DescAt(std::vector<char>* buf, size_t idx) {
  return reinterpret_cast<struct cper_sec_desc*>((buf->data() + kDescTableOffset) +
                                                 (idx * sizeof(struct cper_sec_desc)));
}

// Fills the header fields the parser requires. record_length and sec_cnt stay
// caller-controlled because they are what these tests vary.
void InitHeader(std::vector<char>* buf, uint16_t sec_cnt, uint32_t record_length) {
  auto* hdr = reinterpret_cast<amdsmi_cper_hdr_t*>(buf->data());
  std::memcpy(hdr->signature, "CPER", 4);
  hdr->signature_end = 0xFFFFFFFF;
  hdr->error_severity = AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED;
  hdr->sec_cnt = sec_cnt;
  hdr->record_length = record_length;
}

constexpr size_t kCrashdumpSecOffset = (kDescTableOffset + (3 * sizeof(struct cper_sec_desc)));
constexpr size_t kCrashdumpRecordSize = (kCrashdumpSecOffset + sizeof(struct cper_sec_crashdump));

// [header][desc0][desc1][desc2][crashdump]. desc0 keeps an all-zero (unknown)
// section type and yields nothing; desc1 and desc2 both decode the crashdump, so
// the AFID count reports how far the descriptor walk got.
std::vector<char> MakeCrashdumpRecord(uint32_t record_length) {
  std::vector<char> buf(kCrashdumpRecordSize, 0);
  InitHeader(&buf, /*sec_cnt=*/3, record_length);
  for (size_t i = 1; i < 3; ++i) {
    struct cper_sec_desc* desc = DescAt(&buf, i);
    desc->sec_type = kCrashdumpGuid;
    desc->sec_offset = static_cast<uint32_t>(kCrashdumpSecOffset);
  }
  auto* crashdump = reinterpret_cast<struct cper_sec_crashdump*>(buf.data() + kCrashdumpSecOffset);
  crashdump->data.reg_ctx_type = kAcaRegisterContext;
  return buf;
}

// [header][desc0][non-standard section]. reg_arr_size is caller-controlled so a
// test can claim more registers than reg_dump holds.
std::vector<char> MakeNonStandardRecord(uint16_t reg_arr_size) {
  constexpr size_t kSecOffset = (kDescTableOffset + sizeof(struct cper_sec_desc));
  constexpr size_t kSecSize =
      (sizeof(struct cper_sec_nonstd_err_hdr) + sizeof(struct cper_sec_nonstd_err_body));
  std::vector<char> buf(kSecOffset + kSecSize, 0);
  InitHeader(&buf, /*sec_cnt=*/1, static_cast<uint32_t>(buf.size()));

  struct cper_sec_desc* desc = DescAt(&buf, 0);
  desc->sec_type = kNonStandardGuid;
  desc->sec_offset = static_cast<uint32_t>(kSecOffset);

  auto* body = reinterpret_cast<struct cper_sec_nonstd_err_body*>(
      buf.data() + kSecOffset + sizeof(struct cper_sec_nonstd_err_hdr));
  body->err_ctx.reg_ctx_type = kAcaRegisterContext;
  body->err_ctx.reg_arr_size = reg_arr_size;
  return buf;
}

}  // namespace

// Positive control for the truncation test below: with record_length covering
// the whole descriptor table, both crashdump descriptors decode.
TEST(GpuUnit, CperDecodeDecodesEveryDescriptorInsideTheRecord) {
  std::vector<char> buf = MakeCrashdumpRecord(static_cast<uint32_t>(kCrashdumpRecordSize));
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_EQ(afids.size(), 2u);
}

// The same bytes with record_length claiming only one descriptor. sec_cnt still
// says three, so an unbounded walk decodes desc1 and desc2 from past the record
// end (2 AFIDs, as the control above shows); the walk must stop at desc1.
TEST(GpuUnit, CperDecodeStopsAtTruncatedDescriptorTable) {
  std::vector<char> buf =
      MakeCrashdumpRecord(static_cast<uint32_t>(kDescTableOffset + sizeof(struct cper_sec_desc)));
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_TRUE(afids.empty());
}

// A descriptor whose sec_offset points far past the buffer must be skipped: the
// section pointer is out of bounds and must never be dereferenced (CWE-125).
TEST(GpuUnit, CperDecodeSkipsSectionWithOutOfBoundsOffset) {
  std::vector<char> buf(kDescTableOffset + sizeof(struct cper_sec_desc), 0);
  InitHeader(&buf, /*sec_cnt=*/1, static_cast<uint32_t>(buf.size()));
  struct cper_sec_desc* desc = DescAt(&buf, 0);
  desc->sec_type = kCrashdumpGuid;
  desc->sec_offset = 0x7FFFFFFF;  // wildly out of range

  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());
  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_TRUE(afids.empty());
}

// Positive control for the clamp test below: the largest register array the
// decoder accepts yields one AFID.
TEST(GpuUnit, CperDecodeDecodesNonStandardSectionWithFullRegisterArray) {
  std::vector<char> buf = MakeNonStandardRecord(/*reg_arr_size=*/16 * sizeof(uint64_t));
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_EQ(afids.size(), 1u);
}

// reg_arr_size claims 8191 registers against a 16-register reg_dump. Clamped, it
// decodes exactly like the full array above; unclamped, the decoder reads ~64 KB
// past reg_dump and rejects the length outright, losing the AFID.
TEST(GpuUnit, CperDecodeClampsOversizedRegisterArraySize) {
  std::vector<char> buf = MakeNonStandardRecord(/*reg_arr_size=*/0xFFFF);
  const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(buf.data());

  std::vector<int> afids = cper_decode(hdr, buf.size());
  EXPECT_EQ(afids.size(), 1u);
}

// A header-only record claiming 64 descriptors: the serial injection walks the
// descriptor table of the copy it just made, so it must stop at record_length
// instead of writing fru_id past the copied record.
TEST(GpuUnit, CperInjectSerialStopsAtRecordEnd) {
  std::vector<char> blob = MakeOneRecordBlob();
  reinterpret_cast<amdsmi_cper_hdr_t*>(blob.data())->sec_cnt = 64;
  std::string path;
  WriteTempFile(blob, &path);
  ASSERT_FALSE(path.empty());

  constexpr char kGuard = static_cast<char>(0xAB);
  std::vector<char> cper_data(8192, kGuard);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(4, nullptr);
  uint64_t buf_size = cper_data.size();
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      path.c_str(), 0xFFFFFFFF, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count,
      &cursor, /*product_serial=*/1234);
  unlink(path.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(entry_count, 1u);
  ASSERT_EQ(buf_size, sizeof(amdsmi_cper_hdr_t));
  for (size_t i = buf_size; i < cper_data.size(); ++i) {
    ASSERT_EQ(cper_data[i], kGuard) << "byte " << i << " written past the copied record";
  }
}
