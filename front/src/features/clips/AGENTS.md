# Clip components

- Filter by real camera/event IDs and keep mixed event types distinct even when they share one clip.
- Use native video playback with stable aspect ratio, accessible labels, and clear unavailable states.
- Do not render resident identity, raw RTSP URLs, storage paths, or model debug data.
- Component tests cover playable/unavailable states and URL changes; visual QA uses synthetic or masked frames only.

