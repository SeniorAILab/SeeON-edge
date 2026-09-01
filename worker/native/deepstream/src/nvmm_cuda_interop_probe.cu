#include <cuda_runtime.h>

#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <nvbufsurface.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr guint kWidth = 64;
constexpr guint kHeight = 32;
constexpr guint kBytesPerPixel = 4;
constexpr guint64 kSampleTimeout = 5 * GST_SECOND;

__global__ void digest_rgba_kernel(const std::uint8_t* rgba, std::size_t pitch, guint width,
                                   guint height, std::uint64_t* digest) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  std::uint64_t value = 1469598103934665603ULL;
  for (guint y = 0; y < height; ++y) {
    const std::uint8_t* row = rgba + static_cast<std::size_t>(y) * pitch;
    for (guint x = 0; x < width * kBytesPerPixel; ++x) {
      value ^= row[x];
      value *= 1099511628211ULL;
    }
  }
  *digest = value;
}

void fill_deterministic_rgba(std::uint8_t* pixels) {
  for (guint y = 0; y < kHeight; ++y) {
    for (guint x = 0; x < kWidth; ++x) {
      const std::size_t offset = (static_cast<std::size_t>(y) * kWidth + x) * kBytesPerPixel;
      pixels[offset] = static_cast<std::uint8_t>((x * 17U + y * 3U + 11U) & 0xffU);
      pixels[offset + 1] = static_cast<std::uint8_t>((x * 5U + y * 29U + 7U) & 0xffU);
      pixels[offset + 2] = static_cast<std::uint8_t>((x * 31U + y * 13U + 19U) & 0xffU);
      pixels[offset + 3] = 0xffU;
    }
  }
}

bool cuda_ok(cudaError_t result, const char* action) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "nvmm_cuda_interop_probe: %s: %s\n", action,
               cudaGetErrorString(result));
  return false;
}

bool fail(const char* message) {
  std::fprintf(stderr, "nvmm_cuda_interop_probe: %s\n", message);
  return false;
}

}  // namespace

int main() {
  gst_init(nullptr, nullptr);

  GstElement* pipeline = gst_pipeline_new("nvmm-cuda-interop-probe");
  GstElement* appsrc = gst_element_factory_make("appsrc", "source");
  GstElement* convert = gst_element_factory_make("nvvideoconvert", "convert");
  GstElement* capsfilter = gst_element_factory_make("capsfilter", "nvmm-rgba");
  GstElement* appsink = gst_element_factory_make("appsink", "sink");
  GstSample* sample = nullptr;
  GstMapInfo mapped{};
  bool buffer_mapped = false;
  std::uint64_t* device_digest = nullptr;
  bool kernel_launched = false;
  bool success = false;

  do {
    if (pipeline == nullptr || appsrc == nullptr || convert == nullptr || capsfilter == nullptr ||
        appsink == nullptr) {
      fail("required GStreamer or DeepStream element is unavailable");
      break;
    }

    GstCaps* input_caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA",
                                              "width", G_TYPE_INT, kWidth, "height", G_TYPE_INT,
                                              kHeight, "framerate", GST_TYPE_FRACTION, 1, 1, nullptr);
    GstCaps* output_caps = gst_caps_from_string(
        "video/x-raw(memory:NVMM),format=RGBA,width=64,height=32,framerate=1/1");
    if (input_caps == nullptr || output_caps == nullptr) {
      if (input_caps != nullptr) {
        gst_caps_unref(input_caps);
      }
      if (output_caps != nullptr) {
        gst_caps_unref(output_caps);
      }
      fail("could not create required RGBA caps");
      break;
    }

    g_object_set(appsrc, "caps", input_caps, "format", GST_FORMAT_TIME, "is-live", FALSE, nullptr);
    g_object_set(convert, "nvbuf-memory-type", 2, nullptr);
    g_object_set(capsfilter, "caps", output_caps, nullptr);
    g_object_set(appsink, "sync", FALSE, "max-buffers", 1U, "drop", FALSE, nullptr);
    gst_caps_unref(input_caps);
    gst_caps_unref(output_caps);

    gst_bin_add_many(GST_BIN(pipeline), appsrc, convert, capsfilter, appsink, nullptr);
    if (!gst_element_link_many(appsrc, convert, capsfilter, appsink, nullptr)) {
      fail("could not link appsrc to NVMM RGBA appsink pipeline");
      break;
    }

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
      fail("could not start pipeline");
      break;
    }

    GstBuffer* input = gst_buffer_new_allocate(nullptr, kWidth * kHeight * kBytesPerPixel, nullptr);
    if (input == nullptr) {
      fail("could not allocate deterministic appsrc buffer");
      break;
    }
    GstMapInfo input_map{};
    if (!gst_buffer_map(input, &input_map, GST_MAP_WRITE)) {
      gst_buffer_unref(input);
      fail("could not map deterministic appsrc buffer");
      break;
    }
    fill_deterministic_rgba(input_map.data);
    gst_buffer_unmap(input, &input_map);
    GST_BUFFER_PTS(input) = 0;
    GST_BUFFER_DURATION(input) = GST_SECOND;

    if (gst_app_src_push_buffer(GST_APP_SRC(appsrc), input) != GST_FLOW_OK) {
      fail("could not push deterministic appsrc buffer");
      break;
    }
    if (gst_app_src_end_of_stream(GST_APP_SRC(appsrc)) != GST_FLOW_OK) {
      fail("could not end appsrc stream");
      break;
    }

    sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), kSampleTimeout);
    if (sample == nullptr) {
      fail("timed out waiting for NVMM RGBA GstSample");
      break;
    }
    GstBuffer* output = gst_sample_get_buffer(sample);
    if (output == nullptr || !gst_buffer_map(output, &mapped, GST_MAP_READ)) {
      fail("could not map NVMM GstBuffer");
      break;
    }
    buffer_mapped = true;

    auto* surface = reinterpret_cast<NvBufSurface*>(mapped.data);
    if (surface == nullptr || surface->batchSize != 1 || surface->numFilled != 1 ||
        surface->memType != NVBUF_MEM_CUDA_DEVICE) {
      fail("output is not a single CUDA-device NVMM surface");
      break;
    }
    if (!cuda_ok(cudaSetDevice(surface->gpuId), "selecting NVMM surface GPU")) {
      break;
    }

    const NvBufSurfaceParams& params = surface->surfaceList[0];
    if (params.dataPtr == nullptr || params.width != kWidth || params.height != kHeight ||
        params.pitch < kWidth * kBytesPerPixel || params.colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
      fail("NVMM surface has invalid RGBA dimensions, pitch, format, or device pointer");
      break;
    }
    cudaPointerAttributes attributes{};
    if (!cuda_ok(cudaPointerGetAttributes(&attributes, params.dataPtr),
                 "validating NVMM device pointer")) {
      break;
    }
#if CUDART_VERSION >= 10000
    if (attributes.type != cudaMemoryTypeDevice) {
#else
    if (attributes.memoryType != cudaMemoryTypeDevice) {
#endif
      fail("NVMM surface pointer is not CUDA device memory");
      break;
    }

    cudaDeviceProp properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&properties, surface->gpuId), "reading GPU properties") ||
        !cuda_ok(cudaMalloc(&device_digest, sizeof(*device_digest)), "allocating device digest")) {
      break;
    }
    digest_rgba_kernel<<<1, 1>>>(static_cast<const std::uint8_t*>(params.dataPtr), params.pitch,
                                 params.width, params.height, device_digest);
    kernel_launched = true;
    if (!cuda_ok(cudaGetLastError(), "launching NVMM RGBA digest kernel") ||
        !cuda_ok(cudaDeviceSynchronize(), "synchronizing NVMM RGBA digest kernel")) {
      break;
    }
    kernel_launched = false;

    std::uint64_t digest = 0;
    if (!cuda_ok(cudaMemcpy(&digest, device_digest, sizeof(digest), cudaMemcpyDeviceToHost),
                 "copying bounded digest result")) {
      break;
    }
    std::printf("NVMM_CUDA_INTEROP_RECEIPT cc=%d.%d mem_type=CUDA_DEVICE width=%u height=%u "
                "pitch=%u digest=%016llx\n",
                properties.major, properties.minor, params.width, params.height, params.pitch,
                static_cast<unsigned long long>(digest));
    success = true;
  } while (false);

  if (kernel_launched && !cuda_ok(cudaDeviceSynchronize(), "synchronizing failed kernel before release")) {
    success = false;
  }
  if (device_digest != nullptr && !cuda_ok(cudaFree(device_digest), "freeing device digest")) {
    success = false;
  }
  if (buffer_mapped) {
    gst_buffer_unmap(gst_sample_get_buffer(sample), &mapped);
  }
  if (sample != nullptr) {
    gst_sample_unref(sample);
  }
  if (pipeline != nullptr) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
