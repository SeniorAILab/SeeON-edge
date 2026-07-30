# macOS pre-restructure baseline

This baseline was captured on the current private repository before any
restructuring. Failures below are observations, not remediation work.

## Host and toolchain

| Item | Command | Exit | Recorded value |
| --- | --- | ---: | --- |
| Chip | `system_profiler SPHardwareDataType` | 0 | Apple M3 Max (MacBook Pro, Mac15,10; 14 cores; 36 GB memory) |
| macOS | `sw_vers` | 0 | macOS 26.5.1, build 25F80 |
| Python | `python --version` | 0 | Python 3.12.6 |
| uv | `uv --version` | 0 | `uv 0.10.4 (079e3fd05 2026-02-17)` |
| PyTorch / MPS | `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available(), torch.backends.mps.is_built())"` | 0 | `2.2.2 True True` |
| PyAV import | `python -c "import av; print(av.__version__)"` | 1 | Not importable: `ModuleNotFoundError: No module named 'av'` |

## FFmpeg VideoToolbox capability

| Capability | Command | Exit | Output / conclusion |
| --- | --- | ---: | --- |
| Decoder | `ffmpeg -hide_banner -decoders \| grep -i videotoolbox` | 1 | No matching decoder line was emitted. |
| Encoder | `ffmpeg -hide_banner -encoders \| grep -i videotoolbox` | 0 | `h264_videotoolbox`, `hevc_videotoolbox`, and `prores_videotoolbox` are available. |

## Existing suite state

| Suite command | Exit | State | Summary |
| --- | ---: | --- | --- |
| `uv run pytest -q` | 1 | FAIL | 624 passed, 10 failed, 1 warning in 41.56s |
| `uvx ruff check .` | 0 | PASS | `All checks passed!`; one invalid-`noqa` warning was emitted. |
| `uv run --group lint lint-imports` | 0 | PASS | 7 contracts kept; 0 broken. |
| `pnpm --dir front install --frozen-lockfile && pnpm --dir front test` | 0 | PASS | 25 test files passed; 381 tests passed. |

The frontend command's default Vitest invocation enters watch mode in an
interactive terminal. Its completed initial run reported the passing summary
above and then waited for file changes. To capture a terminating machine exit
code without changing the command itself, it was rerun with `CI=1` exported;
the exact canonical command then exited 0.

### Pre-existing failures: 10

The pre-existing failing test count is **10** (the Python suite); three other
canonical suites passed. The observed error text includes:

```text
edge.evidence.evidence_media.ClipEvidenceError: FINALIZE_FAILED: ffprobe unavailable
AssertionError: metadata.json missing in: ['/Users/beomsu/Documents/01_Project/Senior AI Lab/eldercare-fall-ml/models/fall/lstm']
AssertionError: dataset-ops's ml/contracts/ is missing files present in this repo's canonical copy: ['evidence_export.py']

FAILED tests/test_clip_export_reconciliation.py::test_manifest_v2_uses_finalized_descriptor_and_exact_ready_contract
FAILED tests/test_clip_export_reconciliation.py::test_verify_ready_manifest_refuses_immutable_byte_mismatch[sha256]
FAILED tests/test_clip_export_reconciliation.py::test_verify_ready_manifest_refuses_immutable_byte_mismatch[size_bytes]
FAILED tests/test_clip_export_reconciliation.py::test_reconcile_repairs_manifest_relations_and_persists_verified_outcome
FAILED tests/test_clip_export_reconciliation.py::test_reconcile_marks_mutated_final_media_corrupt_without_rewriting_manifest
FAILED tests/test_clip_recorder.py::test_clip_recorder_finalizes_atomic_manifest_with_pre_and_post_window
FAILED tests/test_clip_recorder.py::test_clip_recorder_fsyncs_media_and_manifest_before_staging_cleanup
FAILED tests/test_evidence_trust_boundaries.py::test_media_probe_uses_same_open_inode_when_path_is_swapped
FAILED tests/test_models_layout.py::test_every_model_folder_has_metadata
FAILED tests/test_vendor_drift.py::test_vendor_package_matches_dataset_ops[contracts]
```

## Protected-path receipt

Before any suite or install command, the five protected paths were recorded in
`../../private-repo-baseline.sha256` relative to this staged `docs/research/`
directory. Its `head_label` is informational only. The authoritative compared
values are the SHA-256 digest of `git ls-tree -r HEAD -- edge/ backend/ front/
shared/ contracts/`, plus separately base64-encoded `porcelain_normal` and
`porcelain_all` payloads for those same five paths.
