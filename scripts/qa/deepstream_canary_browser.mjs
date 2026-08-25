#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const [cameraId, outputDir, clipPath] = process.argv.slice(2);
const token = process.env.CANARY_RELAY_TOKEN;
if (!cameraId || !outputDir || !clipPath || !token) {
  console.error("usage: CANARY_RELAY_TOKEN=<token> node deepstream_canary_browser.mjs <camera-id> <output-dir> <clip>");
  process.exit(2);
}
const screenshot = resolve(outputDir, `viewer-${cameraId}.png`);
const chromium = process.env.CHROMIUM ?? "chromium";
const viewer = `http://127.0.0.1:18090/mjpeg/${encodeURIComponent(cameraId)}`;
const browser = spawnSync(
  chromium,
  [
    "--headless=new",
    "--no-sandbox",
    `--extra-headers=${JSON.stringify({ "X-Edge-Relay-Token": token })}`,
    `--screenshot=${screenshot}`,
    "--window-size=1280,720",
    viewer,
  ],
  { encoding: "utf8", timeout: 30_000 },
);
const probe = spawnSync(
  "ffprobe",
  ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", clipPath],
  { encoding: "utf8", timeout: 15_000 },
);
const clip = readFileSync(clipPath);
const receipt = {
  schema_version: 1,
  camera_id: cameraId,
  viewer_ok: browser.status === 0,
  viewer_exit: browser.status,
  screenshot_sha256: browser.status === 0 ? createHash("sha256").update(readFileSync(screenshot)).digest("hex") : null,
  derivative_playable: probe.status === 0 && probe.stdout.trim().length > 0,
  derivative_sha256: createHash("sha256").update(clip).digest("hex"),
};
writeFileSync(resolve(outputDir, `browser-${cameraId}.json`), `${JSON.stringify(receipt)}\n`, { flag: "wx", mode: 0o400 });
console.log(JSON.stringify(receipt));
process.exit(receipt.viewer_ok && receipt.derivative_playable ? 0 : 1);
