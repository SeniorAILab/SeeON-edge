# Local macOS end-to-end RTSP source

This runbook pins one restricted release clip and exposes it only through
operator-run processes outside the `eldercare-fall-ml` checkout. The edge worker
remains an RTSP client. Do not add MediaMTX, an FFmpeg publisher, a video file, or
any other RTSP serving surface to this repository. Do not commit a local worker
configuration or RTSP credentials.

## Pinned source clip

From the `eldercare-fall-ml` repository root, the pinned release path is:

```text
../eldercare-dataset-ops/ml/data/releases/v1/clips/bo1-77e200e0fe334433e287e551.mp4
```

Exact path on the exercised macOS host:

```text
/Users/beomsu/Documents/01_Project/Senior AI Lab/eldercare-dataset-ops/ml/data/releases/v1/clips/bo1-77e200e0fe334433e287e551.mp4
```

Pin:

```text
SHA-256: 4d5aff92898196fb78e461cc7fd484999f5953de9213900b66d5c52786aba209
Video: HEVC, 2520x970, yuv420p, 93/4 fps, 31.346392 s
```

Verify the file before serving it:

```bash
CLIP="/Users/beomsu/Documents/01_Project/Senior AI Lab/eldercare-dataset-ops/ml/data/releases/v1/clips/bo1-77e200e0fe334433e287e551.mp4"
test -f "$CLIP"
shasum -a 256 "$CLIP"
```

The hash must equal the pin above. Stop if it differs.

## Start the external source

The commands below were exercised with MediaMTX `v1.19.3` and FFmpeg `8.1.2`.
Run them from operator terminals whose working directory is outside the
`eldercare-fall-ml` checkout. Nothing from either process belongs in Git.

### Terminal 1: external RTSP server

```bash
cd "${TMPDIR:-/tmp}"
MTX_PATHS_QA_SOURCE=publisher \
MTX_RTMP=no \
MTX_HLS=no \
MTX_WEBRTC=no \
MTX_SRT=no \
MTX_MOQ=no \
mediamtx
```

This configures the single path `qa` for an external publisher while disabling
the unrelated protocol listeners. Keep this terminal running.

### Terminal 2: external clip publisher

```bash
cd "${TMPDIR:-/tmp}"
CLIP="/Users/beomsu/Documents/01_Project/Senior AI Lab/eldercare-dataset-ops/ml/data/releases/v1/clips/bo1-77e200e0fe334433e287e551.mp4"
RTSP_URL="rtsp://127.0.0.1:8554/qa"
ffmpeg -nostdin -hide_banner -loglevel warning \
  -re -stream_loop -1 -i "$CLIP" \
  -map 0:v:0 -an -c:v copy \
  -f rtsp -rtsp_transport tcp "$RTSP_URL"
```

Keep this terminal running. For a worker on the same Mac, the exact URL is:

```text
rtsp://127.0.0.1:8554/qa
```

The credential-free URL shape for a worker on another authorized host is:

```text
rtsp://<operator-host>:8554/qa
```

There is deliberately no `user:password@` userinfo in either form. If
authentication is later required, supply secrets through the operator's local,
untracked configuration; never put them in this runbook or a tracked URL.

## Required live-stream gate

Before starting the worker, run this probe from a third terminal:

```bash
RTSP_URL="rtsp://127.0.0.1:8554/qa"
ffprobe -v error -rw_timeout 5000000 -rtsp_transport tcp \
  -select_streams v:0 \
  -show_entries stream=index,codec_name,codec_type,width,height,pix_fmt,r_frame_rate,avg_frame_rate \
  -of json "$RTSP_URL"
```

Success is observable only when the command prints a non-empty `streams` array
containing a `codec_type` of `video`, dimensions, and frame-rate metadata. The
2026-07-30 exercised output was:

```json
{
    "programs": [],
    "stream_groups": [],
    "streams": [
        {
            "index": 0,
            "codec_name": "hevc",
            "codec_type": "video",
            "width": 2520,
            "height": 970,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "93/4",
            "avg_frame_rate": "93/4"
        }
    ]
}
```

Do not start the worker on an empty array, an error, or metadata for a non-video
stream. Copy `worker/ml-worker.example.yaml` to the ignored
`worker/ml-worker.local.yaml`, set its `rtsp_url` locally to the exact URL above,
and never commit that local file or any credential-bearing URL.

## Expected negative case

With both external serving processes stopped, run the exact same probe. It must
fail rather than printing live video metadata. The exercised negative result was:

```text
[tcp @ 0xb79048000] Connection to tcp://127.0.0.1:8554?timeout=0 failed: Connection refused
rtsp://127.0.0.1:8554/qa: Connection refused
{

}
```

The exit status was `1`. `Connection refused` is the expected pre-start symptom:
start Terminal 1, then Terminal 2, and repeat the live-stream gate.

## Stop and verify cleanup

Stop the FFmpeg publisher with `Ctrl-C`, then stop MediaMTX with `Ctrl-C`. Confirm
that no listener remains:

```bash
lsof -nP -iTCP:8554 -sTCP:LISTEN
lsof -nP -iUDP:8000
lsof -nP -iUDP:8001
```

All three commands must print no rows. If a process remains, terminate the exact
operator process before leaving the QA session; do not leave port `8554`, `8000`,
or `8001` bound.
