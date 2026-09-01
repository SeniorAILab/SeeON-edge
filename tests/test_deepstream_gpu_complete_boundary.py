from __future__ import annotations

import json
import re
from pathlib import Path

from worker.native.deepstream.preflight import (
    ATAN2_COMPAT_CORPUS_DIGEST,
    ATAN2_COMPAT_LIBM_SHA256,
    ATAN2_COMPAT_RECEIPT,
    ATAN2_COMPAT_TABLE_PAYLOAD_SHA256,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
NATIVE_ROOT = REPOSITORY_ROOT / "worker/native/deepstream"
SOURCE_ROOT = NATIVE_ROOT / "src"


def _source(name: str) -> str:
    return (SOURCE_ROOT / name).read_text(encoding="utf-8")


def _function_region(source: str, qualified_name: str) -> str:
    """Return one C++ function definition, bounded by its balanced braces."""
    match = re.search(rf"\b{re.escape(qualified_name)}\s*\(", source)
    assert match is not None, f"missing definition for {qualified_name}"
    opening = source.find("{", match.end())
    assert opening >= 0, f"missing body for {qualified_name}"

    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[match.start() : index + 1]
    raise AssertionError(f"unterminated body for {qualified_name}")


def _declaration_region(source: str, qualified_name: str) -> str:
    """Return one C++ declaration, bounded by its parameter list and semicolon."""
    match = re.search(rf"\b{re.escape(qualified_name)}\s*\(", source)
    assert match is not None, f"missing declaration for {qualified_name}"

    depth = 0
    for index in range(source.find("(", match.start()), len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                ending = source.find(";", index)
                assert ending >= 0, f"missing declaration terminator for {qualified_name}"
                return source[match.start() : ending + 1]
    raise AssertionError(f"unterminated declaration for {qualified_name}")


def _cmake_target_region(cmake: str, target: str) -> str:
    match = re.search(rf"add_executable\s*\(\s*{re.escape(target)}\b", cmake)
    assert match is not None, f"missing CMake target {target}"
    next_target = re.search(r"\nadd_executable\s*\(", cmake[match.end() :])
    end = match.end() + next_target.start() if next_target else len(cmake)
    return cmake[match.start() : end]


def _production_sources() -> dict[Path, str]:
    excluded = {"trt_perception_stub.cpp"}
    paths = (
        path
        for path in SOURCE_ROOT.iterdir()
        if path.suffix in {".cpp", ".cu", ".hpp"}
        and not path.name.endswith("_test.cpp")
        and path.name not in excluded
    )
    return {path: path.read_text(encoding="utf-8") for path in paths}


def test_encoded_rtsp_branch_keeps_raw_frames_in_nvmm_cuda_device_memory() -> None:
    branch = _source("encoded_source_branch.cpp")
    rtp_builder = _function_region(branch, "on_rtp_pad")
    inference_sample = _function_region(branch, "on_inference_sample")
    failure = _function_region(branch, "inference_contract_failure")

    raw_caps = re.findall(r'gst_caps_from_string\("([^"]*video/x-raw[^"]*)"\)', rtp_builder)
    assert raw_caps
    assert set(raw_caps) == {
        "video/x-raw(memory:NVMM),format=RGBA",
        "video/x-raw(memory:NVMM),format=I420",
    }
    assert all("memory:NVMM" in caps for caps in raw_caps)

    for check in (
        "surface->batchSize != 1",
        "surface->numFilled != 1",
        "surface->surfaceList == nullptr",
        "surface->memType != NVBUF_MEM_CUDA_DEVICE",
        "surface->gpuId != kExpectedGpuId",
        "params.dataPtr == nullptr",
        "params.width != static_cast<guint>(caps_width)",
        "params.height != static_cast<guint>(caps_height)",
        "params.colorFormat != NVBUF_COLOR_FORMAT_RGBA",
        "params.layout != NVBUF_LAYOUT_PITCH",
        "cudaPointerGetAttributes(&attributes, params.dataPtr)",
        "attributes.device != kExpectedGpuId",
    ):
        assert check in inference_sample
    assert "#include <cuda_runtime_api.h>" in branch
    assert 'gst_structure_get_int(caps_structure, "width", &caps_width)' in inference_sample
    assert 'gst_structure_get_int(caps_structure, "height", &caps_height)' in inference_sample
    assert "caps_width <= 0" in inference_sample
    assert "caps_height <= 0" in inference_sample
    assert (
        "attributes.type != cudaMemoryTypeDevice" in inference_sample
        or "attributes.memoryType != cudaMemoryTypeDevice" in inference_sample
    )
    assert inference_sample.index("caps_structure") < inference_sample.index("BufferMap descriptor")
    assert inference_sample.index("surface->surfaceList == nullptr") < inference_sample.index(
        "surface->surfaceList[0]"
    )
    assert inference_sample.index("cudaPointerGetAttributes") > inference_sample.index(
        "params.dataPtr == nullptr"
    )
    assert '"nvmm_surface_contract", FailureScope::kFatal' in failure
    assert "inference_contract_failure(context)" in inference_sample


def test_encoded_rtsp_decoder_pins_and_reads_back_nvdec_properties() -> None:
    rtp_builder = _function_region(_source("encoded_source_branch.cpp"), "on_rtp_pad")

    assert (
        'g_object_set(decoder, "cudadec-memtype", 0, '
        '"num-extra-surfaces", 4U, nullptr)'
    ) in rtp_builder
    assert 'verify_int_property(decoder, "cudadec-memtype", 0)' in rtp_builder
    assert 'verify_int_property(decoder, "num-extra-surfaces", 4)' in rtp_builder
    assert '"nvmm_graph_contract", FailureScope::kFatal' in rtp_builder


def test_encoded_rtsp_builder_accepts_device_frames_not_host_frames() -> None:
    declaration = _source("encoded_source_branch.hpp")
    definition = _source("encoded_source_branch.cpp")

    builder_declaration = _declaration_region(declaration, "build_encoded_rtsp_pipeline")
    builder_definition = _function_region(definition, "build_encoded_rtsp_pipeline")
    for builder in (builder_declaration, builder_definition):
        assert "const DeviceFrameCallback& frames" in builder
        assert "HostFrameCallback" not in builder


def test_device_inference_has_no_host_upload_or_cpu_preprocess_fallback() -> None:
    infer_device = _function_region(_source("trt_perception.cpp"), "TrtPerception::infer_device")

    assert "preprocess_rgba_device_to_bgr_tensor" in infer_device
    for forbidden in (
        "cudaMemcpyHostToDevice",
        "infer_host",
        "preprocess_rgba_to_bgr_tensor",
        "HostFrameView",
    ):
        assert forbidden not in infer_device


def test_production_tensorrt_inference_uses_nonblocking_workspace_acquisition_only() -> None:
    perception = _source("trt_perception.cpp")
    infer_device = _function_region(perception, "TrtPerception::infer_device")

    assert "impl_->pool.try_acquire()" in infer_device
    assert re.search(r"\.available\s*\(", perception) is None


def test_device_inference_copies_only_compact_pose_and_optional_person_rows() -> None:
    perception = _source("trt_perception.cpp")
    header = _source("postprocess_gpu.hpp")
    infer_device = _function_region(perception, "TrtPerception::infer_device")
    validator = _function_region(
        _source("postprocess_gpu.cu"), "validate_postprocess_channel_headers"
    )

    assert "static_assert(sizeof(PostprocessChannelHeader) == 8);" in header
    for engine, context, device_rows, capacity in (
        ("pose", "pose_context", "device_pose", "kPoseOutput"),
        ("person", "person_context", "device_person", "kPersonOutput"),
    ):
        pattern = (
            rf"impl_->{engine},\s+available->{context}\.get\(\),\s+\*available,\s+"
            rf"affine\.tensor_height,\s+affine\.tensor_width,\s+"
            rf"available->{device_rows},\s+{capacity},\s+"
            rf"available->host_{engine}\.data\(\),\s+nullptr,\s+0,\s+nullptr,\s+false,"
        )
        assert re.search(pattern, infer_device)
    assert "sizeof(PostprocessChannelHeader), cudaMemcpyDeviceToHost" in infer_device
    assert "validate_postprocess_channel_headers(" in infer_device
    assert "valid_required_channel_header(pose)" in validator
    assert "valid_required_channel_header(bed)" in validator
    assert "? valid_required_channel_header(person)" in validator
    assert "person.kernel_executed == 0 && person.count == 0" in validator
    assert "pose_count * kPostprocessPoseRowStride * sizeof(float)" in infer_device
    assert "person_count * kPostprocessPersonRowStride * sizeof(float)" in infer_device
    assert "kPoseOutput * sizeof(float)" not in infer_device
    assert "kPersonOutput * sizeof(float)" not in infer_device
    assert "std::vector<double>" not in perception


def test_packed_bed_transfer_abi_and_budget_are_fixed_without_an_envelope() -> None:
    header = _source("postprocess_gpu.hpp")

    for declaration in (
        "kPostprocessBedRowStride = 38",
        "kPostprocessBedMaxPoints = 48",
        "kBedFinalizeSegments = 300",
        "kBedFinalizePixels = 160 * 160",
        "sizeof(PackedBedRecord) == 416",
        "offsetof(PackedBedRecord, box) == 0",
        "offsetof(PackedBedRecord, confidence) == 16",
        "offsetof(PackedBedRecord, point_count) == 24",
        "offsetof(PackedBedRecord, pad) == 28",
        "offsetof(PackedBedRecord, points) == 32",
        "kPostprocessMaxPersonTransferBytes == 200424",
        "kPostprocessPoseOnlyTransferBytes == 193224",
    ):
        assert declaration in header
    assert "postprocess_bed_transfer_bytes(std::size_t count)" in header
    assert "count * sizeof(PackedBedRecord)" in header
    assert "Envelope" not in header


def test_production_and_gpu_postprocess_targets_pin_finalizer_math() -> None:
    cmake = (NATIVE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    for target in (
        "seeon-deepstream-child",
        "seeon-deepstream-preflight",
        "seeon-deepstream-postprocess-gpu-test",
    ):
        region = _cmake_target_region(cmake, target)
        assert "src/pinned_host_atan2.cu" in region
        assert "src/postprocess_gpu.cu" in region
        assert "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>" in region


def test_bed_finalizer_is_device_segmented_and_uses_pinned_host_atan2() -> None:
    postprocess = _source("postprocess_gpu.cu")
    finalizer = _function_region(postprocess, "finalize_bed_rows_device")

    assert "#include <cub/device/device_segmented_radix_sort.cuh>" in postprocess
    assert "pinned_host_atan2(" in postprocess
    assert "cub::DeviceSegmentedRadixSort::SortPairs(" in finalizer
    assert "kBedFinalizeSegments" in finalizer
    assert "points.Current()" in finalizer
    for forbidden in (
        "cudaMalloc",
        "cudaFree",
        "std::sort",
        "std::stable_sort",
        "std::nth_element",
        "std::partial_sort",
        "malloc(",
        "new ",
    ):
        assert forbidden not in postprocess
    assert not re.search(r"(?<![\w:])atan2\s*\(", postprocess)


def test_device_bed_path_copies_only_headers_then_count_sized_final_records() -> None:
    infer_device = _function_region(_source("trt_perception.cpp"), "TrtPerception::infer_device")
    validator = _function_region(
        _source("postprocess_gpu.cu"), "validate_postprocess_channel_headers"
    )

    bed_enqueue = re.search(
        r"enqueue_engine\(\s*impl_->bed,[\s\S]*?available->device_bed,\s*"
        r"kBedOutput,\s*nullptr,\s*available->device_bed_prototypes,\s*"
        r"kBedPrototypes,\s*nullptr,\s*false,\s*error\)",
        infer_device,
    )
    assert bed_enqueue is not None
    for channel in ("pose", "person", "bed"):
        assert re.search(
            rf"&available->host_{channel}_header,\s*available->device_{channel}_header,\s*"
            r"sizeof\(PostprocessChannelHeader\),\s*cudaMemcpyDeviceToHost",
            infer_device,
        )
    assert "cudaMemsetAsync(available->device_person_header, 0" in infer_device
    assert not re.search(
        r"device_person_header[\s\S]{0,160}cudaMemcpyHostToDevice", infer_device
    )
    assert re.search(
        r"host_bed_records\.data\(\),\s*available->device_bed_records,\s*"
        r"bed_count \* sizeof\(PackedBedRecord\),\s*cudaMemcpyDeviceToHost",
        infer_device,
    )
    for forbidden in (
        "available->device_bed,",
        "available->device_bed_prototypes,",
        "parse_bed_rows",
    ):
        assert forbidden not in infer_device[infer_device.index("cudaStreamSynchronize") :]
    assert "validate_postprocess_channel_headers(" in infer_device
    assert "valid_required_channel_header(pose)" in validator
    assert "valid_required_channel_header(bed)" in validator
    assert "? valid_required_channel_header(person)" in validator


def test_bed_finalizer_workspace_is_fixed_and_allocated_only_at_load() -> None:
    perception = _source("trt_perception.cpp")
    load = _function_region(perception, "TrtPerception::load")
    infer_device = _function_region(perception, "TrtPerception::infer_device")

    for allocation in (
        "kPostprocessTensorRows * sizeof(PackedBedRecord)",
        "kBedFinalizeEntries * sizeof(std::uint64_t)",
        "kBedFinalizeEntries * sizeof(std::uint32_t)",
        "kBedFinalizeSegments * sizeof(std::int32_t)",
        "(kBedFinalizeSegments + 1) * sizeof(std::int32_t)",
    ):
        assert allocation in load
    assert "offsets[segment] = segment * kBedFinalizePixels" in load
    assert "cudaMemcpy(workspace->bed_finalize.offsets" in load
    assert "cudaMemcpyHostToDevice" in load
    assert "query_bed_finalize_workspace_temp_bytes(&workspace->bed_finalize, error)" in load
    assert "cudaMalloc(&workspace->bed_finalize.cub_temp" in load
    assert "cudaMalloc" not in infer_device


def test_postprocess_kernel_preserves_source_order_and_current_row_cuts() -> None:
    kernel = _source("postprocess_gpu.cu")

    assert "for (int row = 0; row < kPostprocessTensorRows; ++row)" in kernel
    assert "static_cast<double>(score) > 0.05" in kernel
    assert (
        "static_cast<int>(static_cast<double>(\n"
        "                                 source_rows[row * kStride + 5])) == 0"
    ) in kernel
    assert "!(static_cast<double>(score) < 0.25)" in kernel
    assert kernel.index("const int source_offset = row * kStride;") < kernel.index(
        "const int compact_offset = count * kStride;"
    )
    assert kernel.index("compact_words[compact_offset + column]") < kernel.index("++count;")


def test_production_native_sources_have_no_legacy_frame_or_raw_infer_api() -> None:
    production = _production_sources()
    legacy_symbols = {
        "DecodedFrameView": re.compile(r"\bDecodedFrameView\b"),
        "raw TrtPerception::infer": re.compile(r"\bTrtPerception::infer\s*\("),
    }

    for label, pattern in legacy_symbols.items():
        offenders = [
            str(path.relative_to(REPOSITORY_ROOT))
            for path, source in production.items()
            if pattern.search(source)
        ]
        assert not offenders, f"{label} remains in production sources: {offenders}"


def test_edge_docker_ctest_lanes_exclude_gpu_tagged_tests() -> None:
    dockerfile = (REPOSITORY_ROOT / "Dockerfile.edge").read_text(encoding="utf-8")
    invocations = re.findall(r"ctest --test-dir ([^\s]+) --output-on-failure([^\n]*)", dockerfile)

    assert [directory for directory, _ in invocations] == [
        "build",
        "build-sanitized",
        "/opt/seeon/native-build",
    ]
    assert all(re.search(r"(?:^|\s)-LE\s+gpu(?:\s|$)", arguments) for _, arguments in invocations)


def test_cuda_build_contract_pins_blackwell_architecture_and_nonfused_preprocess() -> None:
    cmake = (NATIVE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert re.search(
        r'set\(CMAKE_CUDA_ARCHITECTURES\s+"120-real;120-virtual"\s+CACHE\s+STRING',
        cmake,
    )
    for target in (
        "seeon-deepstream-nvmm-cuda-interop-probe",
        "seeon-deepstream-preflight",
        "seeon-deepstream-child",
    ):
        target_region = _cmake_target_region(cmake, target)
        assert "src/preprocess_gpu.cu" in target_region
        assert "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>" in target_region

    child = _cmake_target_region(cmake, "seeon-deepstream-child")
    assert "${SEEON_CUDA_INCLUDE}" in child
    assert "${SEEON_CUDART}" in child

    gstreamer_test = _cmake_target_region(
        cmake, "seeon-deepstream-source-runtime-gstreamer-test"
    )
    assert "${SEEON_CUDA_INCLUDE}" in gstreamer_test
    assert "${SEEON_CUDART}" in gstreamer_test


def test_postprocess_gpu_runtime_ctest_matches_build_install_and_docker_gpu_lanes() -> None:
    cmake = (NATIVE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    runtime_ctest = (NATIVE_ROOT / "runtime-ctest.cmake.in").read_text(encoding="utf-8")
    target = "seeon-deepstream-postprocess-gpu-test"
    target_region = _cmake_target_region(cmake, target)

    assert "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>" in target_region
    assert f"NAME {target}" in target_region
    assert f"set_tests_properties({target} PROPERTIES LABELS gpu)" in target_region
    assert re.search(
        rf"install\(\s*TARGETS\b[\s\S]*\b{re.escape(target)}\b[\s\S]*?"
        r"RUNTIME DESTINATION native-build/bin\s*\)",
        cmake,
    )
    assert f'add_test(\n  {target}\n  "bin/{target}"\n)' in runtime_ctest
    assert f"set_tests_properties({target} PROPERTIES LABELS gpu)" in runtime_ctest


def test_pinned_host_atan2_gpu_contract_is_registered_and_has_no_native_cuda_atan2() -> None:
    cmake = (NATIVE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    runtime_ctest = (NATIVE_ROOT / "runtime-ctest.cmake.in").read_text(encoding="utf-8")
    target = "seeon-deepstream-pinned-host-atan2-test"
    target_region = _cmake_target_region(cmake, target)

    assert "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>" in target_region
    assert f"NAME {target}" in target_region
    assert f"set_tests_properties({target} PROPERTIES LABELS gpu)" in target_region
    assert re.search(
        rf"install\(\s*TARGETS\b[\s\S]*\b{re.escape(target)}\b[\s\S]*?"
        r"RUNTIME DESTINATION native-build/bin\s*\)",
        cmake,
    )
    assert f'add_test(\n  {target}\n  "bin/{target}"\n)' in runtime_ctest
    assert f"set_tests_properties({target} PROPERTIES LABELS gpu)" in runtime_ctest

    for path, source in _production_sources().items():
        if path.name == "pinned_host_atan2.cu":
            continue
        assert not re.search(r"(?<![:\w])atan2\s*\(", source), path


def test_pinned_host_libm_golden_digest_binds_manifest_preflight_and_receipt() -> None:
    production_digest = (
        "1b87a1a50b496cfead2b0ad134c2ff536705c82608db240c7e8aa48d6c0e4217"
    )
    manifest = json.loads(
        (NATIVE_ROOT / "manifest.template.json").read_text(encoding="utf-8")
    )
    preflight_source = (NATIVE_ROOT / "preflight.py").read_text(encoding="utf-8")

    assert production_digest == ATAN2_COMPAT_LIBM_SHA256
    assert f'ATAN2_COMPAT_LIBM_SHA256: Final = "{production_digest}"' in preflight_source
    assert manifest["atan2_compat"]["host_libm_sha256"] == production_digest
    assert f"libm={production_digest}" in ATAN2_COMPAT_RECEIPT.pattern
    native_test = _source("pinned_host_atan2_test.cu")
    assert f'kPinnedLibmDigest[] = "{production_digest}"' in native_test
    assert "libm=%s" in native_test
    assert "provider_digest.data()" in native_test


def test_pinned_host_table_and_corpus_digests_bind_measured_receipt() -> None:
    manifest = json.loads(
        (NATIVE_ROOT / "manifest.template.json").read_text(encoding="utf-8")
    )
    native_test = _source("pinned_host_atan2_test.cu")

    assert (
        manifest["atan2_compat"]["table_payload_sha256"]
        == ATAN2_COMPAT_TABLE_PAYLOAD_SHA256
    )
    assert manifest["atan2_compat"]["corpus_digest"] == ATAN2_COMPAT_CORPUS_DIGEST
    assert f"table_payload={ATAN2_COMPAT_TABLE_PAYLOAD_SHA256}" in (
        ATAN2_COMPAT_RECEIPT.pattern
    )
    assert f"corpus_digest={ATAN2_COMPAT_CORPUS_DIGEST}" in ATAN2_COMPAT_RECEIPT.pattern
    assert "pinned_host_atan2_copy_table(table)" in native_test
    assert "SHA256(bytes.data(), bytes.size(), digest.data())" in native_test
