#include "pinned_host_atan2.cuh"

#include <cuda_runtime.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <utility>
#include <vector>

namespace {
struct Input { double y; double x; int row; int col; };
struct Result { uint64_t angle; int row; int col; uint32_t ordinal; };
using HostAtan2 = double (*)(double, double);

[[noreturn]] void fail(const char* message) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
void check(cudaError_t status, const char* operation) { if (status != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status)); std::exit(1); } }
uint64_t raw(double value) { return std::bit_cast<uint64_t>(value); }
constexpr char kPinnedLibmDigest[] = "1b87a1a50b496cfead2b0ad134c2ff536705c82608db240c7e8aa48d6c0e4217";
constexpr char kS2CorpusDigest[] = "4bee588f444e6fd596893fb78037ffda4ff67345fdcfd921e3209a7fd145dccb";
constexpr uint64_t kCanonicalNan = 0x7ff8000000000000ULL;

__global__ void angles(const Input* inputs, Result* results, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    results[i] = {static_cast<uint64_t>(__double_as_longlong(pinned_host_atan2(inputs[i].y, inputs[i].x))), inputs[i].row, inputs[i].col, static_cast<uint32_t>(i)};
  }
}

std::array<char, SHA256_DIGEST_LENGTH * 2 + 1> hex_digest(const std::array<unsigned char, SHA256_DIGEST_LENGTH>& digest) {
  std::array<char, SHA256_DIGEST_LENGTH * 2 + 1> text{};
  constexpr char hex[] = "0123456789abcdef";
  for (std::size_t i = 0; i < digest.size(); ++i) {
    text[2 * i] = hex[digest[i] >> 4];
    text[2 * i + 1] = hex[digest[i] & 0xf];
  }
  return text;
}

std::array<char, SHA256_DIGEST_LENGTH * 2 + 1> file_digest(const char* path) {
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr) fail("host atan2 provider open failed");
  if (std::fseek(file, 0, SEEK_END) != 0) fail("host atan2 provider seek failed");
  const long length = std::ftell(file);
  if (length < 0 || std::fseek(file, 0, SEEK_SET) != 0) fail("host atan2 provider size failed");
  std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
  if ((!bytes.empty() && std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size()) ||
      std::fclose(file) != 0) {
    fail("host atan2 provider read failed");
  }
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  if (SHA256(bytes.data(), bytes.size(), digest.data()) == nullptr) fail("host atan2 provider SHA-256 failed");
  return hex_digest(digest);
}

HostAtan2 verified_host_atan2(std::array<char, SHA256_DIGEST_LENGTH * 2 + 1>* provider_digest) {
  const HostAtan2 host_atan2 = static_cast<HostAtan2>(&std::atan2);
  Dl_info provider{};
  if (dladdr(reinterpret_cast<const void*>(host_atan2), &provider) == 0 ||
      provider.dli_fname == nullptr) {
    fail("host atan2 provider resolution failed");
  }
  *provider_digest = file_digest(provider.dli_fname);
  if (std::strcmp(provider_digest->data(), kPinnedLibmDigest) != 0) {
    fail("host atan2 provider SHA-256 mismatch");
  }
  return host_atan2;
}

std::array<char, SHA256_DIGEST_LENGTH * 2 + 1> table_payload_digest() {
  uint64_t table[241][7]{};
  check(pinned_host_atan2_copy_table(table), "cudaMemcpyFromSymbol C");
  std::array<unsigned char, 241 * 7 * sizeof(uint64_t)> bytes{};
  for (std::size_t i = 0; i < 241; ++i) {
    for (std::size_t j = 0; j < 7; ++j) {
      const uint64_t word = table[i][j];
      for (unsigned int byte = 0; byte < sizeof(word); ++byte) {
        bytes[(i * 7 + j) * sizeof(word) + byte] =
            static_cast<unsigned char>(word >> (byte * 8));
      }
    }
  }
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  if (SHA256(bytes.data(), bytes.size(), digest.data()) == nullptr) fail("SHA-256 table payload failed");
  return hex_digest(digest);
}

bool less_result(const Result& left, const Result& right) {
  const double a = std::bit_cast<double>(left.angle);
  const double b = std::bit_cast<double>(right.angle);
  if (a != b) return a < b;
  if (left.row != right.row) return left.row < right.row;
  if (left.col != right.col) return left.col < right.col;
  return left.ordinal < right.ordinal;
}

void compare(const std::vector<Input>& inputs, HostAtan2 host_atan2, int* angle_cases, int* order_cases) {
  Input* device_inputs = nullptr; Result* device_results = nullptr;
  check(cudaMalloc(&device_inputs, inputs.size() * sizeof(*device_inputs)), "cudaMalloc inputs");
  check(cudaMalloc(&device_results, inputs.size() * sizeof(*device_results)), "cudaMalloc results");
  check(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(*device_inputs), cudaMemcpyHostToDevice), "cudaMemcpy inputs");
  angles<<<static_cast<unsigned>((inputs.size() + 127) / 128), 128>>>(device_inputs, device_results, static_cast<int>(inputs.size()));
  check(cudaGetLastError(), "atan2 kernel launch"); check(cudaDeviceSynchronize(), "atan2 kernel synchronize");
  std::vector<Result> actual(inputs.size()), expected;
  check(cudaMemcpy(actual.data(), device_results, actual.size() * sizeof(*device_results), cudaMemcpyDeviceToHost), "cudaMemcpy results");
  check(cudaFree(device_results), "cudaFree results"); check(cudaFree(device_inputs), "cudaFree inputs");
  expected.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const Input& input = inputs[index];
    const uint64_t host = raw(host_atan2(input.y, input.x));
    const Result& device = actual[index];
    if (device.row != input.row || device.col != input.col || device.ordinal != index || device.angle != host) {
      std::fprintf(stderr, "raw-bit mismatch y=%a x=%a host=%016llx device=%016llx\n", input.y, input.x, static_cast<unsigned long long>(host), static_cast<unsigned long long>(device.angle));
      std::exit(1);
    }
    expected.push_back({host, input.row, input.col, static_cast<uint32_t>(index)}); ++*angle_cases;
  }
  std::stable_sort(actual.begin(), actual.end(), less_result);
  std::stable_sort(expected.begin(), expected.end(), less_result);
  if (!std::equal(actual.begin(), actual.end(), expected.begin(),
                  [](const Result& left, const Result& right) {
                    return left.angle == right.angle && left.row == right.row &&
                           left.col == right.col && left.ordinal == right.ordinal;
                  })) {
    fail("stable (angle,y,x,ordinal) order mismatch");
  }
  ++*order_cases;
}

std::vector<Input> active_inputs(const std::vector<std::pair<int, int>>& points) {
  uint64_t sum_y = 0, sum_x = 0;
  for (const auto [y, x] : points) { sum_y += y; sum_x += x; }
  const double cy = static_cast<double>(sum_y) / static_cast<double>(points.size());
  const double cx = static_cast<double>(sum_x) / static_cast<double>(points.size());
  std::vector<Input> values; values.reserve(points.size());
  for (const auto [y, x] : points) values.push_back({static_cast<double>(y) - cy, static_cast<double>(x) - cx, y, x});
  return values;
}

void append_le_u32(std::vector<unsigned char>* bytes, uint32_t value) {
  for (unsigned int byte = 0; byte < sizeof(value); ++byte) {
    bytes->push_back(static_cast<unsigned char>(value >> (byte * 8)));
  }
}

void check_invalid_domain() {
  const std::vector<Input> invalid{
      {160.0, 1.0, 1, 1}, {-160.0, 1.0, 2, 1},
      {INFINITY, 1.0, 3, 1}, {1.0, -INFINITY, 4, 1},
      {NAN, 1.0, 5, 1}, {1.0, NAN, 6, 1},
  };
  Input* device_inputs = nullptr;
  Result* device_results = nullptr;
  check(cudaMalloc(&device_inputs, invalid.size() * sizeof(*device_inputs)), "cudaMalloc invalid inputs");
  check(cudaMalloc(&device_results, invalid.size() * sizeof(*device_results)), "cudaMalloc invalid results");
  check(cudaMemcpy(device_inputs, invalid.data(), invalid.size() * sizeof(*device_inputs), cudaMemcpyHostToDevice), "cudaMemcpy invalid inputs");
  angles<<<1, static_cast<unsigned int>(invalid.size())>>>(device_inputs, device_results, static_cast<int>(invalid.size()));
  check(cudaGetLastError(), "invalid-domain kernel launch");
  check(cudaDeviceSynchronize(), "invalid-domain kernel synchronize");
  std::vector<Result> results(invalid.size());
  check(cudaMemcpy(results.data(), device_results, results.size() * sizeof(*device_results), cudaMemcpyDeviceToHost), "cudaMemcpy invalid results");
  check(cudaFree(device_results), "cudaFree invalid results");
  check(cudaFree(device_inputs), "cudaFree invalid inputs");
  for (const Result& result : results) {
    if (result.angle != kCanonicalNan) fail("invalid contour delta did not return canonical NaN");
  }
}
}  // namespace

int main() {
  int angle_cases = 0, order_cases = 0;
  std::array<char, SHA256_DIGEST_LENGTH * 2 + 1> provider_digest{};
  const HostAtan2 host_atan2 = verified_host_atan2(&provider_digest);
  const auto table_digest = table_payload_digest();
  if (std::strcmp(table_digest.data(), "48e09fcdce6990c03bd03b476de3a3cfdc24d524751623c2e2c140ea3fbd6c3b") != 0) fail("device C payload SHA-256 mismatch");
  check_invalid_domain();
  std::vector<Input> direct;
  // Every table cell and both selection boundaries, across all quadrants.
  for (int i = 0; i < 241; ++i) for (double delta : {-0x1p-54, 0.0, 0x1p-54}) for (int sy : {-1, 1}) for (int sx : {-1, 1}) {
    const double u = (static_cast<double>(i + 16) / 256.0) + delta;
    direct.push_back({sy * u, static_cast<double>(sx), 100000 + i * 12 + (sy > 0) * 6 + (sx > 0) * 3 + (delta > 0), i});
  }
  direct.insert(direct.end(), {{0.0, 1.0, 1, 1}, {-0.0, 1.0, 2, 1}, {0.0, -1.0, 3, 1}, {-0.0, -1.0, 4, 1}, {1.0, 0.0, 5, 1}, {-1.0, 0.0, 6, 1}, {1.0 / 16.0, 1.0, 7, 1}, {1.0, 1.0 / 16.0, 8, 1}});
  compare(direct, host_atan2, &angle_cases, &order_cases);

  // n=1 and n=25600 division/centroid edges.
  compare(active_inputs({{0, 0}}), host_atan2, &angle_cases, &order_cases);
  std::vector<std::pair<int, int>> full; full.reserve(160 * 160);
  for (int y = 0; y < 160; ++y) for (int x = 0; x < 160; ++x) full.emplace_back(y, x);
  compare(active_inputs(full), host_atan2, &angle_cases, &order_cases);

  // Exact deterministic S2 corpus that exposed the trial-936 order inversion.
  std::mt19937 rng(477);
  std::vector<unsigned char> corpus_bytes;
  corpus_bytes.reserve(2000 * (1 + 2 * 300) * sizeof(uint32_t));
  for (int trial = 0; trial < 2000; ++trial) {
    const int count = 1 + static_cast<int>(rng() % 300);
    append_le_u32(&corpus_bytes, static_cast<uint32_t>(count));
    std::vector<std::pair<int, int>> points;
    std::vector<bool> used(160 * 160);
    while (static_cast<int>(points.size()) < count) {
      const int y = static_cast<int>(rng() % 160);
      const int x = static_cast<int>(rng() % 160);
      const int index = y * 160 + x;
      if (!used[index]) {
        used[index] = true;
        points.emplace_back(y, x);
        append_le_u32(&corpus_bytes, static_cast<uint32_t>(y));
        append_le_u32(&corpus_bytes, static_cast<uint32_t>(x));
      }
    }
    compare(active_inputs(points), host_atan2, &angle_cases, &order_cases);
  }
  std::array<unsigned char, SHA256_DIGEST_LENGTH> corpus_digest_bytes{};
  if (SHA256(corpus_bytes.data(), corpus_bytes.size(), corpus_digest_bytes.data()) == nullptr) {
    fail("S2 corpus SHA-256 failed");
  }
  const auto corpus_digest = hex_digest(corpus_digest_bytes);
  if (std::strcmp(corpus_digest.data(), kS2CorpusDigest) != 0) fail("S2 corpus SHA-256 mismatch");

  // Permanent S2 regression: trial 936's documented centroid and inversion pair.
  const double cy = static_cast<double>(20746) / 264.0, cx = static_cast<double>(21505) / 264.0;
  const Input inversion{70.0 - cy, 60.0 - cx, 70, 60};
  if (raw(host_atan2(inversion.y, inversion.x)) != 0xc00616b466d73d61ULL) fail("S2 host fixture identity mismatch");
  compare({inversion, {46.0 - cy, 0.0 - cx, 46, 0}}, host_atan2, &angle_cases, &order_cases);
  std::printf("pinned-host-atan2 receipt variant=glibc-2.39-x86_64-fma libm=%s source=uatan.tbl:8072e14d43f1b897ab7013b70954e18b113ef2fcac942e8a263a579d4baf531a table_payload=%s corpus_digest=%s angle_cases=%d order_cases=%d mismatches=0\n", provider_digest.data(), table_digest.data(), corpus_digest.data(), angle_cases, order_cases);
}
