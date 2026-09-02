#!/usr/bin/env bash
# Publish a deterministic, video-only replay input without decoding or transcoding it.
set -euo pipefail

usage() {
    printf 'usage: %s SOURCE.mp4 [OUTPUT_ROOT]\n' "${0##*/}" >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
source_file=$1
output_root=${2:-/var/lib/seeon-canary/corpus}
output_file="$output_root/replay-v1.mp4"
checksum_file="$output_file.sha256"

[[ -f "$source_file" ]] || { printf 'source is not a regular file: %s\n' "$source_file" >&2; exit 1; }
command -v ffprobe >/dev/null || { printf 'ffprobe is required\n' >&2; exit 1; }
command -v ffmpeg >/dev/null || { printf 'ffmpeg is required\n' >&2; exit 1; }
command -v sha256sum >/dev/null || { printf 'sha256sum is required\n' >&2; exit 1; }

# There must be exactly one video stream, and it must already be H.264.
mapfile -t video_codecs < <(
    ffprobe -v error -select_streams v -show_entries stream=codec_name \
        -of default=noprint_wrappers=1:nokey=1 "$source_file"
)
[[ ${#video_codecs[@]} -eq 1 && ${video_codecs[0]} == h264 ]] || {
    printf 'source must contain exactly one H.264 video stream\n' >&2
    exit 1
}

# Never replace a published corpus artifact or its integrity record.
[[ ! -e "$output_file" && ! -e "$checksum_file" ]] || {
    printf 'refusing to overwrite replay corpus output under: %s\n' "$output_root" >&2
    exit 1
}
mkdir -p -- "$output_root"

work_file="$output_root/.replay-v1.$$.mp4"
work_checksum="$output_root/.replay-v1.mp4.sha256.$$"
cleanup() { rm -f -- "$work_file" "$work_checksum"; }
trap cleanup EXIT

# -c:v copy is intentional: this is container remuxing, never pixel conversion.
ffmpeg -nostdin -v error -n -i "$source_file" -map 0:v:0 -an -c:v copy "$work_file"
chmod 0444 -- "$work_file"
mv -n -- "$work_file" "$output_file"

checksum=$(sha256sum "$output_file")
printf '%s  replay-v1.mp4\n' "${checksum%% *}" > "$work_checksum"
chmod 0444 -- "$work_checksum"
mv -n -- "$work_checksum" "$checksum_file"
