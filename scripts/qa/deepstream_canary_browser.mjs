#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
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
const consumer = spawn(
  "curl",
  ["--silent", "--show-error", "--max-time", "30", "--header", `X-Edge-Relay-Token: ${token}`, viewer],
  { stdio: ["ignore", "pipe", "pipe"] },
);

const frame = await new Promise((resolveFrame, rejectFrame) => {
  let received = Buffer.alloc(0);
  const timeout = setTimeout(() => rejectFrame(new Error("MJPEG frame deadline exceeded")), 20_000);
  consumer.once("error", (error) => {
    clearTimeout(timeout);
    rejectFrame(error);
  });
  consumer.once("close", (code) => {
    if (received.indexOf(Buffer.from([0xff, 0xd9])) < 0) {
      clearTimeout(timeout);
      rejectFrame(new Error(`MJPEG consumer exited before a frame: ${code}`));
    }
  });
  consumer.stdout.on("data", (chunk) => {
    received = Buffer.concat([received, chunk]);
    const start = received.indexOf(Buffer.from([0xff, 0xd8]));
    const end = received.indexOf(Buffer.from([0xff, 0xd9]), Math.max(0, start + 2));
    if (start >= 0 && end > start) {
      clearTimeout(timeout);
      resolveFrame(received.subarray(start, end + 2));
    }
  });
}).catch((error) => {
  console.error(String(error));
  consumer.kill("SIGTERM");
  process.exit(1);
});

const browser = spawnSync(
  chromium,
  [
    "--headless=new",
    "--no-sandbox",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=5000",
    `--extra-headers=${JSON.stringify({ "X-Edge-Relay-Token": token })}`,
    `--screenshot=${screenshot}`,
    "--window-size=1280,720",
    viewer,
  ],
  { encoding: "utf8", timeout: 30_000 },
);
consumer.kill("SIGTERM");

const probe = spawnSync(
  "ffprobe",
  ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", clipPath],
  { encoding: "utf8", timeout: 15_000 },
);
const clip = readFileSync(clipPath);
const viewerOk = browser.status === 0 && existsSync(screenshot);
const receipt = {
  schema_version: 1,
  camera_id: cameraId,
  mjpeg_frame_sha256: createHash("sha256").update(frame).digest("hex"),
  viewer_ok: viewerOk,
  viewer_exit: browser.status,
  screenshot_sha256: viewerOk ? createHash("sha256").update(readFileSync(screenshot)).digest("hex") : null,
  derivative_playable: probe.status === 0 && probe.stdout.trim().length > 0,
  derivative_sha256: createHash("sha256").update(clip).digest("hex"),
};
writeFileSync(resolve(outputDir, `browser-${cameraId}.json`), `${JSON.stringify(receipt)}\n`, { flag: "wx", mode: 0o400 });
console.log(JSON.stringify(receipt));
process.exit(receipt.viewer_ok && receipt.derivative_playable ? 0 : 1);
