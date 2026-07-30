# Edge dashboard

- The dashboard talks only to `/api/v1` through the existing API client and server-validated HttpOnly session; never compile worker relay credentials into frontend assets.
- Clip video URLs come from the API; never read the clip-store path directly or expose camera credentials.
- Preserve explicit loading/unavailable/error states and native video controls. Do not compensate producer timestamp bugs with playbackRate.
- Run focused Vitest tests and the dashboard build for UI changes.

