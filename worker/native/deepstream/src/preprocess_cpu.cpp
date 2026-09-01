#include "preprocess_cpu.hpp"

#include <algorithm>
#include <cstddef>

namespace seeon::trt {

void preprocess_rgba_to_bgr_tensor(const std::uint8_t* rgba, int width, int height,
                                   int stride, const perception::AffineMetadata& affine,
                                   float* output_chw) {
  const int tensor_width = affine.tensor_width;
  const int tensor_height = affine.tensor_height;
  const int content_width = affine.content_width;
  const int content_height = affine.content_height;
  const int pad_left = affine.pad_left();
  const int pad_top = affine.pad_top();
  const std::size_t plane = static_cast<std::size_t>(tensor_width) * tensor_height;
  constexpr float kPad = static_cast<float>(seeon::perception::kLetterboxPadValue) / 255.0F;
  std::fill(output_chw, output_chw + 3 * plane, kPad);
  const double scale_x = static_cast<double>(width) / content_width;
  const double scale_y = static_cast<double>(height) / content_height;
  for (int y = 0; y < content_height; ++y) {
    const double source_y = (y + 0.5) * scale_y - 0.5;
    const double clamped_y = std::clamp(source_y, 0.0, static_cast<double>(height - 1));
    const int y0 = static_cast<int>(clamped_y);
    const int y1 = std::min(y0 + 1, height - 1);
    const double wy = clamped_y - y0;
    const std::uint8_t* row0 = rgba + static_cast<std::size_t>(y0) * stride;
    const std::uint8_t* row1 = rgba + static_cast<std::size_t>(y1) * stride;
    const std::size_t out_row =
        static_cast<std::size_t>(y + pad_top) * tensor_width + pad_left;
    for (int x = 0; x < content_width; ++x) {
      const double source_x = (x + 0.5) * scale_x - 0.5;
      const double clamped_x = std::clamp(source_x, 0.0, static_cast<double>(width - 1));
      const int x0 = static_cast<int>(clamped_x);
      const int x1 = std::min(x0 + 1, width - 1);
      const double wx = clamped_x - x0;
      const std::size_t offset = out_row + static_cast<std::size_t>(x);
      for (int channel = 0; channel < 3; ++channel) {
        const int source_byte = 2 - channel;
        const double top =
            (1.0 - wx) * row0[static_cast<std::size_t>(x0) * 4 + source_byte] +
            wx * row0[static_cast<std::size_t>(x1) * 4 + source_byte];
        const double bottom =
            (1.0 - wx) * row1[static_cast<std::size_t>(x0) * 4 + source_byte] +
            wx * row1[static_cast<std::size_t>(x1) * 4 + source_byte];
        const double value = (1.0 - wy) * top + wy * bottom;
        output_chw[static_cast<std::size_t>(channel) * plane + offset] =
            static_cast<float>(value / 255.0);
      }
    }
  }
}

}  // namespace seeon::trt
