// Cross-language parity driver for native_perception.{hpp,cpp}.
//
// Reads a line protocol on stdin and answers on stdout so the Python parity
// suite (tests/test_native_perception_cpp_parity.py) can compare this exact
// production code against the C3/C4 Python references. All floating-point
// values cross the boundary as C99 hexfloats ("%a"), never decimal, so the
// comparison is bit-exact. Prototype tensors are synthesized on both sides
// from the same SplitMix64 stream to keep fixtures small.

#include "native_perception.hpp"
#include "preprocess_cpu.hpp"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using seeon::perception::AffineMetadata;
using seeon::perception::LegacyGreedyBboxIou;
using seeon::perception::ParsedBox;

double parse_hex(const std::string& token) { return std::strtod(token.c_str(), nullptr); }

std::string hex(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%a", value);
  return buffer;
}

std::vector<double> read_rows(std::istream& input, int count, int stride) {
  std::vector<double> rows;
  rows.reserve(static_cast<std::size_t>(count) * static_cast<std::size_t>(stride));
  for (int row = 0; row < count; ++row) {
    std::string line;
    std::getline(input, line);
    std::istringstream tokens{line};
    std::string token;
    while (tokens >> token) rows.push_back(parse_hex(token));
  }
  return rows;
}

std::uint32_t rotr(std::uint32_t value, int bits) {
  return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
 public:
  Sha256() : state_{0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19} {}

  void update(const std::uint8_t* data, std::size_t size) {
    bit_count_ += static_cast<std::uint64_t>(size) * 8;
    while (size > 0) {
      const std::size_t copied = std::min(size, block_.size() - block_size_);
      std::copy_n(data, copied, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
      block_size_ += copied;
      data += copied;
      size -= copied;
      if (block_size_ == block_.size()) {
        transform();
        block_size_ = 0;
      }
    }
  }

  std::string finish() {
    const std::uint64_t original_bit_count = bit_count_;
    const std::uint8_t one = 0x80;
    update(&one, 1);
    const std::uint8_t zero = 0;
    while (block_size_ != 56) update(&zero, 1);
    std::array<std::uint8_t, 8> length{};
    for (int index = 0; index < 8; ++index) {
      length[static_cast<std::size_t>(index)] =
          static_cast<std::uint8_t>(original_bit_count >> ((7 - index) * 8));
    }
    update(length.data(), length.size());
    static constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const std::uint32_t word : state_) {
      for (int shift = 28; shift >= 0; shift -= 4) {
        result.push_back(kHex[(word >> shift) & 0xF]);
      }
    }
    return result;
  }

 private:
  void transform() {
    static constexpr std::array<std::uint32_t, 64> kConstants{
        0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1,
        0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
        0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786,
        0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147,
        0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
        0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B,
        0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A,
        0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
        0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2};
    std::array<std::uint32_t, 64> words{};
    for (int index = 0; index < 16; ++index) {
      const std::size_t offset = static_cast<std::size_t>(index) * 4;
      words[static_cast<std::size_t>(index)] =
          (static_cast<std::uint32_t>(block_[offset]) << 24) |
          (static_cast<std::uint32_t>(block_[offset + 1]) << 16) |
          (static_cast<std::uint32_t>(block_[offset + 2]) << 8) | block_[offset + 3];
    }
    for (int index = 16; index < 64; ++index) {
      const auto s0 = rotr(words[static_cast<std::size_t>(index - 15)], 7) ^
                      rotr(words[static_cast<std::size_t>(index - 15)], 18) ^
                      (words[static_cast<std::size_t>(index - 15)] >> 3);
      const auto s1 = rotr(words[static_cast<std::size_t>(index - 2)], 17) ^
                      rotr(words[static_cast<std::size_t>(index - 2)], 19) ^
                      (words[static_cast<std::size_t>(index - 2)] >> 10);
      words[static_cast<std::size_t>(index)] =
          words[static_cast<std::size_t>(index - 16)] + s0 +
          words[static_cast<std::size_t>(index - 7)] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (int index = 0; index < 64; ++index) {
      const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const auto choice = (e & f) ^ (~e & g);
      const auto temporary1 = h + s1 + choice + kConstants[static_cast<std::size_t>(index)] +
                              words[static_cast<std::size_t>(index)];
      const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temporary2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t bit_count_ = 0;
};

bool decode_hex(const std::string& encoded, std::vector<std::uint8_t>* output) {
  if (encoded.size() % 2 != 0) return false;
  output->resize(encoded.size() / 2);
  for (std::size_t index = 0; index < output->size(); ++index) {
    const auto digit = [](char value) -> int {
      if (value >= '0' && value <= '9') return value - '0';
      if (value >= 'a' && value <= 'f') return value - 'a' + 10;
      if (value >= 'A' && value <= 'F') return value - 'A' + 10;
      return -1;
    };
    const int high = digit(encoded[index * 2]);
    const int low = digit(encoded[index * 2 + 1]);
    if (high < 0 || low < 0) return false;
    (*output)[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return true;
}

std::string tensor_digest(const std::vector<float>& tensor) {
  static_assert(sizeof(float) == 4);
  Sha256 digest;
  for (float value : tensor) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    digest.update(bytes, sizeof(value));
#else
    for (int index = 3; index >= 0; --index) digest.update(bytes + index, 1);
#endif
  }
  return digest.finish();
}

// Deterministic cross-language PRNG: SplitMix64 -> double in [-1, 1).
class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
  double next_signed_unit() {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    const double unit = static_cast<double>(z >> 11) * 0x1.0p-53;
    return unit * 2.0 - 1.0;
  }

 private:
  std::uint64_t state_;
};

std::vector<double> synthesize_prototypes(std::uint64_t seed) {
  constexpr std::size_t kSize = 32ULL * 160ULL * 160ULL;
  SplitMix64 generator{seed};
  std::vector<double> values;
  values.reserve(kSize);
  for (std::size_t index = 0; index < kSize; ++index) {
    values.push_back(generator.next_signed_unit());
  }
  return values;
}

void print_affine(const AffineMetadata& affine) {
  std::printf("AFFINE %d %d %d %d %d %d %s %d %d %s %s\n", affine.source_height,
              affine.source_width, affine.tensor_height, affine.tensor_width,
              affine.content_height, affine.content_width, hex(affine.gain).c_str(),
              affine.box_pad_x, affine.box_pad_y, hex(affine.keypoint_pad_x).c_str(),
              hex(affine.keypoint_pad_y).c_str());
}

int run() {
  std::string line;
  std::unique_ptr<LegacyGreedyBboxIou> association;
  while (std::getline(std::cin, line)) {
    std::istringstream tokens{line};
    std::string command;
    tokens >> command;
    if (command == "AFFINE") {
      int height = 0;
      int width = 0;
      tokens >> height >> width;
      print_affine(seeon::perception::letterbox_affine(height, width));
    } else if (command == "PREPROC") {
      int height = 0;
      int width = 0;
      int stride = 0;
      std::string encoded_rgba;
      tokens >> height >> width >> stride >> encoded_rgba;
      std::vector<std::uint8_t> rgba;
      if (height <= 0 || width <= 0 || stride < width * 4) {
        std::printf("ERR invalid-preproc-frame\n");
        return 2;
      }
      const std::size_t expected_size =
          static_cast<std::size_t>(height) * static_cast<std::size_t>(stride);
      if (!decode_hex(encoded_rgba, &rgba) || rgba.size() != expected_size) {
        std::printf("ERR invalid-preproc-frame\n");
        return 2;
      }
      const auto affine = seeon::perception::letterbox_affine(height, width);
      std::vector<float> tensor(
          3ULL * static_cast<std::size_t>(affine.tensor_height) * affine.tensor_width);
      seeon::trt::preprocess_rgba_to_bgr_tensor(rgba.data(), width, height, stride, affine,
                                                 tensor.data());
      std::printf("PREPROC %s\n", tensor_digest(tensor).c_str());
    } else if (command == "INVBOX") {
      int height = 0;
      int width = 0;
      std::string x1, y1, x2, y2;
      tokens >> height >> width >> x1 >> y1 >> x2 >> y2;
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto box = affine.invert_box(parse_hex(x1), parse_hex(y1), parse_hex(x2),
                                         parse_hex(y2));
      std::printf("BOX %d %d %d %d\n", box.x1, box.y1, box.x2, box.y2);
    } else if (command == "INVKP") {
      int height = 0;
      int width = 0;
      std::string x, y;
      tokens >> height >> width >> x >> y;
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto [sx, sy] = affine.invert_keypoint(parse_hex(x), parse_hex(y));
      std::printf("KP %d %d\n", sx, sy);
    } else if (command == "POSE") {
      int height = 0;
      int width = 0;
      int count = 0;
      tokens >> height >> width >> count;
      const auto rows = read_rows(std::cin, count, seeon::perception::kPoseRowStride);
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto parsed = seeon::perception::parse_pose_rows(rows, affine);
      std::printf("POSECOUNT %zu\n", parsed.boxes.size());
      for (std::size_t index = 0; index < parsed.boxes.size(); ++index) {
        const ParsedBox& box = parsed.boxes[index];
        std::printf("POSEBOX %d %d %d %d %s\n", box.x1, box.y1, box.x2, box.y2,
                    hex(box.confidence).c_str());
        std::printf("POSEKP");
        for (const auto& point : parsed.poses[index]) {
          std::printf(" %d %d %s", point.x, point.y, hex(point.score).c_str());
        }
        std::printf("\n");
      }
    } else if (command == "PERSON") {
      int height = 0;
      int width = 0;
      std::string confidence;
      int count = 0;
      tokens >> height >> width >> confidence >> count;
      const auto rows = read_rows(std::cin, count, seeon::perception::kPersonRowStride);
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto boxes =
          seeon::perception::parse_person_rows(rows, affine, parse_hex(confidence));
      std::printf("PERSONCOUNT %zu\n", boxes.size());
      for (const ParsedBox& box : boxes) {
        std::printf("PERSONBOX %d %d %d %d %s\n", box.x1, box.y1, box.x2, box.y2,
                    hex(box.confidence).c_str());
      }
    } else if (command == "BED") {
      int height = 0;
      int width = 0;
      std::string confidence;
      std::uint64_t seed = 0;
      int max_points = 0;
      int count = 0;
      tokens >> height >> width >> confidence >> seed >> max_points >> count;
      const auto rows = read_rows(std::cin, count, seeon::perception::kBedRowStride);
      const auto prototypes = synthesize_prototypes(seed);
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto regions = seeon::perception::parse_bed_rows(
          rows, prototypes, affine, parse_hex(confidence), max_points);
      std::printf("BEDCOUNT %zu\n", regions.size());
      for (const auto& region : regions) {
        std::printf("BEDBOX %d %d %d %d %s\n", region.bounds.x1, region.bounds.y1,
                    region.bounds.x2, region.bounds.y2, hex(region.bounds.confidence).c_str());
        std::printf("BEDPOLY %zu", region.polygon.size());
        for (const auto& [x, y] : region.polygon) std::printf(" %d %d", x, y);
        std::printf("\n");
      }
    } else if (command == "ASSOC") {
      std::string action;
      tokens >> action;
      if (action == "NEW") {
        std::string min_iou;
        int max_misses = 0;
        tokens >> min_iou >> max_misses;
        association = std::make_unique<LegacyGreedyBboxIou>(parse_hex(min_iou), max_misses);
        std::printf("ACK\n");
      } else if (action == "OBSERVE") {
        int count = 0;
        tokens >> count;
        std::vector<ParsedBox> boxes;
        boxes.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
          std::string row;
          std::getline(std::cin, row);
          std::istringstream fields{row};
          ParsedBox box{};
          std::string conf;
          fields >> box.x1 >> box.y1 >> box.x2 >> box.y2 >> conf;
          box.confidence = parse_hex(conf);
          boxes.push_back(box);
        }
        const auto output = association->observe(boxes);
        std::printf("ASSOC %zu", output.track_ids.size());
        for (std::size_t index = 0; index < output.track_ids.size(); ++index) {
          std::printf(" %" PRId64 ":%u", output.track_ids[index],
                      output.selected_cue_indexes[index]);
        }
        std::printf("\n");
      } else if (action == "COAST") {
        association->coast();
        std::printf("ACK\n");
      } else if (action == "RESET") {
        association->reset();
        std::printf("ACK\n");
      } else {
        std::printf("ERR unknown-assoc-action\n");
        return 2;
      }
    } else if (command.empty()) {
      continue;
    } else {
      std::printf("ERR unknown-command\n");
      return 2;
    }
    std::fflush(stdout);
  }
  return 0;
}

}  // namespace

int main() { return run(); }
