#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace seeon {
using FrameCallback = std::function<void(const std::string&, std::uint64_t)>;

class SourceRuntime {
 public:
  explicit SourceRuntime(FrameCallback callback);
  ~SourceRuntime();
  SourceRuntime(const SourceRuntime&) = delete;
  SourceRuntime& operator=(const SourceRuntime&) = delete;

  [[nodiscard]] bool add(const std::string& camera, const std::string& uri, std::string* error);
  [[nodiscard]] bool remove(const std::string& camera);
  [[nodiscard]] bool rebuild(const std::string& camera, std::string* error);
  [[nodiscard]] std::size_t count() const;
  [[nodiscard]] bool custom_transform_available() const;

 private:
  class Impl;
  FrameCallback callback_;
  std::unique_ptr<Impl> impl_;
};
}  // namespace seeon
