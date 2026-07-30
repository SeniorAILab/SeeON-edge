# GPU Inference Serving Modularization — Best Practice (edge worker, 2→50 cameras)

- Date: 2026-07-20
- Scope: real-time multi-model inference (pose/bed/person/fall) over N RTSP cameras on a single edge GPU.
- Related: ADR-0002 (fail-fast modularization), deep-interview spec, ralplan plan.
- Confidence: high on the decode/compute bottleneck framing and batching; medium on exact 50-camera capacity (needs measurement).

## Decision question

How should the edge GPU inference plane be structured so it (a) works correctly and fails loud today at 2 cameras, and (b) scales to ~50 cameras on a single mid-high RTX GPU without a rewrite?

## Findings (with sources)

1. **The real bottleneck at scale is batched inference compute, not decode.**
   NVDEC (decode) has no hard concurrent-session cap like NVENC; it is throughput (pixel/s) bound, so 50×640×360@30fps decode fits a single modern GPU with margin.
   Source: NVIDIA Video Encode/Decode GPU support matrix.

2. **NVENC (clip encoding) is the real hard cap.** Consumer/RTX GeForce exposes a bounded number of concurrent NVENC sessions (recent gens ~12; RTX 5070 Ti = 12). At 50 cameras, simultaneous incident-clip encodes can saturate it → this is a scale trigger, not a 2-camera concern.
   Source: NVIDIA Video Encode/Decode GPU support matrix (NVENC concurrent sessions column).

3. **Cross-camera micro-batching is the enabler that fits 50 streams on one GPU.** Running one model per stream unbatched oversubscribes the GPU; batching frames from many cameras into one forward pass amortizes kernel-launch overhead and raises occupancy.
   Sources: NVIDIA DeepStream (`Gst-nvstreammux` batches multi-stream frames before `Gst-nvinfer`); NVIDIA Triton dynamic batching (`max_batch_size` + dynamic batcher auto-batches requests across clients).

4. **The seam belongs behind `edge/serving_client`.** Keeping inference provisioning behind `ServingClient` lets the in-process runner be swapped for a networked batched service without touching edge orchestration/perception/domain code (enforced by the import-linter "serving seam" contract).

## Options and tradeoffs

| Option | Summary | Pros | Cons | When |
|--------|---------|------|------|------|
| **A. Multi-process camera sharding** | M processes × K cameras, each in-process inference | Simplest evolution, strong isolation, low new-infra | No cross-camera batching → poor GPU utilization; model VRAM × M | ~10–20 cameras / quick |
| **B. Batched inference service behind `serving_client` (Triton-class)** | Decode/inference split; cross-camera dynamic batching; single model load | Best GPU utilization, VRAM-efficient, reuses the seam, incrementally adoptable | Operate a serving service; convert models to TensorRT/ONNX; config.pbtxt | **50-camera standard** |
| **C. DeepStream / GStreamer pipeline** | `nvstreammux`→`nvinfer`(TensorRT)→NVDEC zero-copy | Highest raw efficiency, purpose-built multi-stream | Heavy framework adoption; Python bindings deprecated (→C/C++ or pyservicemaker); large rewrite of edge structure | NVENC saturation / extreme scale |

## Recommendation

- **Now (2 cameras):** keep in-process behind `serving_client`; apply fail-fast (ADR-0002) so GPU/NVDEC failures are loud, not masked as CPU fallback.
- **Design for 50 (this cycle):** define the seam's batch-input contract (`BatchServingClient`, ADR-0002) so **Option B** can slot in without a caller rewrite. Do not build the batching backend yet.
- **Trigger to adopt B/C:** GPU compute saturation or NVENC 12-session saturation → promote a batched serving service (B), or adopt DeepStream (C) if a full GStreamer rewrite is warranted.

## Cautionary note — verify the CUDA build, not just the code

A correct modular pipeline still silently degrades to CPU if the installed torch wheel lacks the GPU's compute-capability kernels. Observed root cause (ADR-0002): the default PyPI `torch` wheel resolved with an empty `torch.cuda.get_arch_list()` (no Blackwell `sm_120`), so `cuda.is_available()==False` and inference fell back to CPU. Pin torch to the PyTorch CUDA index and verify `get_arch_list()` includes the deployed GPU's arch (`sm_120` for RTX 50xx) in the actual container before trusting GPU inference. Fail-fast turns this silent failure into a loud one.

## Gaps / next

- Exact single-GPU 50-camera capacity for this model set (pose+bed+person+fall @ 5fps) needs measurement on the target RTX 5070 Ti with batched inference.
- The second-layer CUDA-context init issue (ADR-0002) must be resolved and verified in the real edge container before capacity numbers are meaningful.
