# worker/native/deepstream: supervised NVIDIA child

Native C++ media/inference for the `nvidia` profile plus the Python-owned
cross-language contract. The pinned DeepStream image builds the binaries;
CPU, MPS, and iGPU profiles do not run them.

## Ownership

- `src/` owns per-source RTSP/depay/parser and the encoded-AU tee before decode;
  the decode branch owns `nvv4l2decoder`/NVMM, custom TensorRT inference, tensor
  parse, inverse geometry, association, and bounded preview output. Production
  RTSP inference accepts CUDA-device NVMM RGBA only and uses custom
  `TrtPerception::infer_device`; host frames are confined to explicit
  non-RTSP/preflight paths. There is no host or unified-memory fallback.
  Production uses custom `TrtPerception`, not `nvinfer` or `nvtracker`;
  `nvstreammux batch-size=1` appears only in the preflight warmup.
- Python modules here own control/IPC codecs, metadata admission, manifest
  preflight, engine-cache identity, and parity helpers. PID-1 supervision and
  source lifecycle belong to `worker/runtime/deepstream/`; C++
  `ChildServer`/`SourceRuntime` execute child commands and source graphs.
- `PerceptionFrameV1` is the image-free boundary. Its identity is worker boot,
  camera, stream epoch, sequence, and optional source PTS; independent person,
  pose, and bed channels each report `inferred`, `inferred_empty`, or `skipped`.
  Raw tensors and host frames do not cross this boundary.
- The child alone receives `CUDA_VISIBLE_DEVICES=0`; the Python parent must not
  create a second CUDA context.
- `copy_telemetry.cpp` is an opt-in native child/preflight sidecar only. Its
  `SEEON_CANARY_TELEMETRY_PATH` input derives the distinct
  `.child-copy.jsonl` sibling; it must never write or alter parent
  `native-telemetry.jsonl`, public IPC, `PerceptionFrameV1`, or stderr. Disabled
  telemetry is a no-op and must not allocate, record, or read CUDA timing
  events. Completed device frames record the caller's explicit `camera_id`,
  literal zero H2D bytes, exact calculated D2H transfer bytes, and busy-pool
  drops. Schema-1 sidecar data is qualification evidence, not a production
  runtime contract.

## Association and media invariants

`association/registry.py` enables only `legacy-greedy-bbox-iou.v1`.
`pose-aware-bbox-iou.v1` is registered but disabled. Keep source-order pose
rows, the legacy score threshold/integer conversion, and the no-second-NMS
rule unless a separate behavior change is approved.

Device contour ordering uses the private pinned-host `atan2` compatibility
component, not CUDA `atan2`. Its glibc/libm/table identities and raw-bit/order
corpus are boot-gated; changing the host image or expression graph without a
zero-mismatch replacement must refuse rollout. Device owns bed thresholding,
mask/prototype evaluation, contour finalization, linspace sampling, inverse
geometry, and source-order compaction. Bed crosses to CPU only as the exact
count-sized `PackedBedRecord` transfer after device finalization; CPU owns
policy and conversion only. Do not add a second NMS.

The encoded-AU tee is source-primary and precedes decode. Primary evidence
remuxes that stream; preview JPEGs are bounded derivatives.
Do not add stock `nvtracker`, a second encoder on the primary path, or a public
`PerceptionFrameV1` contract.

## Preflight and engines

`run_configured_deepstream_preflight` runs only when
`SEEON_DEEPSTREAM_MANIFEST` is set. It checks GPU compatibility, pinned runtime
and plugin identity, read-only models, native and NVMM/CUDA interop binary
identity, deterministic GPU-preprocess parity, both engine-cache identities,
and one bounded loopback inference receipt. Failure writes the configured
first-fault JSON and refuses boot.

Engine plans are content-addressed from weights, exporter, `fp32` precision,
and builder identity under `c7-<plan-key[:32]>`. Deploy builds explicitly;
boot calls `verify_plan_cache` and never builds an engine.

## Commands and focused tests

- Build/verify cache: `uv run python -m worker.native.deepstream.engine_cache build <manifest>` / `uv run python -m worker.native.deepstream.engine_cache verify <manifest>`.
- Standalone preflight: `uv run python -m worker.native.deepstream.preflight <manifest>`.
- Native compile and CTest run in `docker build -f Dockerfile.edge --target runtime .`.
  (`--target` matters: the file's last stage is the CI-only `bootsmoke`.)
- GPU-labelled native tests are excluded from image-build CTest and must be run
  explicitly on the deployment GPU; they cover NVMM interop and bit-exact
  preprocess parity.
- Focused: `uv run pytest -q tests/test_perception_frame_v1.py tests/test_deepstream_full_wire.py tests/test_deepstream_preflight.py tests/test_deepstream_model_parity.py tests/test_native_association_registry.py`.
- `uv run --group lint lint-imports` checks configured contracts; the native
  types-only ceiling is checked by focused dependency tests and manual import
  path review.
- `seeon-deepstream-copy-telemetry-test` is a normal CTest registration; retain
  it with the child/preflight compilation linkage. Static boundary coverage in
  `tests/test_deepstream_gpu_complete_boundary.py` must fail closed on sidecar
  separation, exact transfer accounting, disabled timing gating, and public
  schema isolation.
