#include "native_perception.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using seeon::perception::ParsedBedRegion;
using seeon::perception::ParsedBox;
using seeon::perception::ParsedKeypoint;
using seeon::perception::ParsedPose;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}

std::vector<double> promote(const std::vector<float>& values) {
  std::vector<double> promoted;
  promoted.reserve(values.size());
  for (float value : values) promoted.push_back(static_cast<double>(value));
  return promoted;
}

void check_box(const ParsedBox& expected, const ParsedBox& actual, const std::string& name) {
  check(expected.x1 == actual.x1 && expected.y1 == actual.y1 &&
            expected.x2 == actual.x2 && expected.y2 == actual.y2,
        name + " bounds differ");
  check(expected.confidence == actual.confidence, name + " confidence differs");
}

void check_keypoint(const ParsedKeypoint& expected, const ParsedKeypoint& actual,
                    const std::string& name) {
  check(expected.x == actual.x && expected.y == actual.y, name + " coordinates differ");
  check(expected.score == actual.score, name + " score differs");
}

void check_pose(const ParsedPose& expected, const ParsedPose& actual) {
  check(expected.boxes.size() == actual.boxes.size(), "pose box count differs");
  check(expected.poses.size() == actual.poses.size(), "pose keypoint row count differs");
  for (std::size_t row = 0; row < expected.boxes.size(); ++row) {
    check_box(expected.boxes[row], actual.boxes[row], "pose box");
    check(expected.poses[row].size() == actual.poses[row].size(),
          "pose keypoint count differs");
    for (std::size_t point = 0; point < expected.poses[row].size(); ++point) {
      check_keypoint(expected.poses[row][point], actual.poses[row][point], "pose keypoint");
    }
  }
}

void check_boxes(const std::vector<ParsedBox>& expected, const std::vector<ParsedBox>& actual,
                 const std::string& name) {
  check(expected.size() == actual.size(), name + " count differs");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    check_box(expected[index], actual[index], name);
  }
}

void check_bed_regions(const std::vector<ParsedBedRegion>& expected,
                       const std::vector<ParsedBedRegion>& actual) {
  check(expected.size() == actual.size(), "bed region count differs");
  for (std::size_t region = 0; region < expected.size(); ++region) {
    check_box(expected[region].bounds, actual[region].bounds, "bed bounds");
    check(expected[region].polygon.size() == actual[region].polygon.size(),
          "bed polygon count differs");
    for (std::size_t point = 0; point < expected[region].polygon.size(); ++point) {
      check(expected[region].polygon[point] == actual[region].polygon[point],
            "bed polygon point differs");
    }
  }
}

void set_box_row(std::vector<float>* rows, int stride, int row, float x1, float y1, float x2,
                 float y2, float score, float class_id) {
  const std::size_t offset = static_cast<std::size_t>(row * stride);
  (*rows)[offset] = x1;
  (*rows)[offset + 1] = y1;
  (*rows)[offset + 2] = x2;
  (*rows)[offset + 3] = y2;
  (*rows)[offset + 4] = score;
  (*rows)[offset + 5] = class_id;
}

}  // namespace

int main() {
  namespace perception = seeon::perception;
  const auto affine = perception::letterbox_affine(480, 640);

  const float pose_literal_threshold = 0.05F;
  std::vector<float> pose_rows(static_cast<std::size_t>(3 * perception::kPoseRowStride), 0.0F);
  for (int row = 0; row < 3; ++row) {
    set_box_row(&pose_rows, perception::kPoseRowStride, row, 40.25F + row, 32.5F + row,
                140.75F + row, 232.25F + row,
                row == 0 ? std::nextafterf(pose_literal_threshold, 0.0F)
                         : row == 1 ? pose_literal_threshold
                                    : std::nextafterf(pose_literal_threshold, INFINITY),
                0.0F);
    const std::size_t offset = static_cast<std::size_t>(row * perception::kPoseRowStride);
    for (int point = 0; point < perception::kCocoKeypointCount; ++point) {
      const std::size_t base = offset + perception::kPersonRowStride + point * 3;
      pose_rows[base] = 50.125F + row + point * 1.25F;
      pose_rows[base + 1] = 60.875F + row + point * 0.75F;
      pose_rows[base + 2] = 0.125F + point * 0.03125F;
    }
  }
  const auto promoted_pose_rows = promote(pose_rows);
  const auto expected_pose = perception::parse_pose_rows(promoted_pose_rows, affine);
  const auto actual_pose = perception::parse_pose_rows(std::span<const float>{pose_rows}, affine);
  check_pose(expected_pose, actual_pose);
  check(actual_pose.boxes.size() == 2, "pose threshold cases did not retain literal and above rows");
  check(actual_pose.boxes[0].confidence == static_cast<double>(pose_literal_threshold),
        "pose literal threshold row was not first retained source row");

  const float person_confidence = 0.25F;
  std::vector<float> person_rows(static_cast<std::size_t>(4 * perception::kPersonRowStride),
                                 0.0F);
  set_box_row(&person_rows, perception::kPersonRowStride, 0, 12.25F, 20.5F, 110.75F, 220.25F,
              person_confidence, 0.0F);
  set_box_row(&person_rows, perception::kPersonRowStride, 1, 24.25F, 30.5F, 124.75F, 230.25F,
              0.5F, 1.0F);
  set_box_row(&person_rows, perception::kPersonRowStride, 2, 36.25F, 40.5F, 136.75F, 240.25F,
              std::nextafterf(person_confidence, 0.0F), 0.0F);
  set_box_row(&person_rows, perception::kPersonRowStride, 3, 48.25F, 50.5F, 148.75F, 250.25F,
              0.5F, 0.0F);
  const auto promoted_person_rows = promote(person_rows);
  const auto expected_people =
      perception::parse_person_rows(promoted_person_rows, affine, static_cast<double>(person_confidence));
  const auto actual_people = perception::parse_person_rows(
      std::span<const float>{person_rows}, affine, static_cast<double>(person_confidence));
  check_boxes(expected_people, actual_people, "person");
  check(actual_people.size() == 2, "person class or equality confidence filtering differs");
  check(actual_people[0].confidence == static_cast<double>(person_confidence),
        "person equality confidence row was not retained first");

  std::vector<float> prototypes(static_cast<std::size_t>(perception::kBedPrototypeChannels *
                                                          perception::kBedPrototypeHeight *
                                                          perception::kBedPrototypeWidth));
  for (int channel = 0; channel < perception::kBedPrototypeChannels; ++channel) {
    for (int y = 0; y < perception::kBedPrototypeHeight; ++y) {
      for (int x = 0; x < perception::kBedPrototypeWidth; ++x) {
        const std::size_t index = static_cast<std::size_t>(channel) *
                                      perception::kBedPrototypeHeight *
                                      perception::kBedPrototypeWidth +
                                  static_cast<std::size_t>(y) * perception::kBedPrototypeWidth + x;
        const float base = channel % 2 == 0 ? 0.75F : -0.5F;
        prototypes[index] = base + static_cast<float>((channel + 3 * y + x) % 5) * 0.001F;
      }
    }
  }
  std::vector<float> bed_rows(static_cast<std::size_t>(3 * perception::kBedRowStride), 0.0F);
  const float bed_confidence = 0.25F;
  set_box_row(&bed_rows, perception::kBedRowStride, 0, 80.0F, 60.0F, 160.0F, 180.0F,
              bed_confidence, static_cast<float>(perception::kCocoBedClassId));
  set_box_row(&bed_rows, perception::kBedRowStride, 1, 0.0F, 0.0F, 64.0F, 64.0F, bed_confidence,
              static_cast<float>(perception::kCocoBedClassId));
  set_box_row(&bed_rows, perception::kBedRowStride, 2, 200.0F, 100.0F, 300.0F, 220.0F, 0.5F,
              0.0F);
  const std::size_t active_offset = static_cast<std::size_t>(perception::kBedRowStride);
  for (int channel = 0; channel < perception::kBedPrototypeChannels; ++channel) {
    bed_rows[active_offset + perception::kPersonRowStride + channel] =
        channel % 2 == 0 ? 0.5F : -0.25F;
  }
  const auto promoted_bed_rows = promote(bed_rows);
  const auto promoted_prototypes = promote(prototypes);
  const auto expected_bed = perception::parse_bed_rows(
      promoted_bed_rows, promoted_prototypes, affine, static_cast<double>(bed_confidence), 3);
  const auto actual_bed = perception::parse_bed_rows(
      std::span<const float>{bed_rows}, std::span<const float>{prototypes}, affine,
      static_cast<double>(bed_confidence), 3);
  check_bed_regions(expected_bed, actual_bed);
  check(actual_bed.size() == 2, "bed source order or class filtering differs");
  check(actual_bed[0].polygon.empty(), "empty bed mask was not preserved first");
  check(actual_bed[1].polygon.size() == 3, "bed max_points sampling was not applied");

  return 0;
}
