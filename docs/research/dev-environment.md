# Development environment: local language servers

Date: 2026-07-30
Repository under test: `/Users/beomsu/Documents/01_Project/Senior AI Lab/eldercare-fall-ml`

## Scope

TODO 6 installs/verifies only the local Python and TypeScript language-server
toolchain. No GPU host was contacted or modified. No repository source was
modified. The LSP responses below are from the local `lsp_symbols` requests.

## Installed versions and response verdicts

### Python — basedpyright

- Executable: `/Users/beomsu/.local/bin/basedpyright-langserver`
- Version string: `basedpyright 1.39.9` (based on pyright `1.1.411`)
- Install state: already present via uv tool; no reinstall performed.
- Answered: **YES**. A document-symbol request against the real repository
  file returned a non-empty result (see evidence below).

### TypeScript — typescript-language-server

- Executable: `/Users/beomsu/.local/bin/typescript-language-server`
- Language-server version string: `5.3.0`
- TypeScript engine installed globally: `typescript@7.0.2`
- Install action: `npm install -g typescript-language-server typescript`
- Answered: **YES**. A document-symbol request against the real repository
  file returned a non-empty result (see evidence below).

## Required happy QA

Request: document symbols for
`backend/app/features/status/router.py`

Raw response:

```text
router (Variable) - line 15
status (Function) - line 18
  request (Variable) - line 19
  heartbeat_store (Variable) - line 20
  inventory (Variable) - line 21
  response (Variable) - line 22
__all__ (Variable) - line 27
```

Verdict: **PASS** — non-empty symbols were returned.

## Required negative QA

Request: document symbols for deliberately non-existent path
`backend/app/features/status/THIS_PATH_DOES_NOT_EXIST.py`

Raw response:

```text
ENOENT: no such file or directory, open '/Users/beomsu/Documents/01_Project/Senior AI Lab/eldercare-fall-ml/backend/app/features/status/THIS_PATH_DOES_NOT_EXIST.py'
```

Verdict: **PASS (expected negative case)** — the request returned an error and
was recorded rather than being treated as a successful server response.

## Additional TypeScript response

Request: document symbols for `front/src/shared/api/types.ts`.

Raw response:

```text
Camera (Variable) - line 5
CameraHeartbeat (Variable) - line 30
CameraInput (Variable) - line 82
CameraPatchInput (Variable) - line 88
CameraRegistry (Variable) - line 23
CameraStatus (Variable) - line 1
CameraTestResult (Variable) - line 98
Clip (Variable) - line 128
ClipLabel (Variable) - line 126
DecodeBackend (Variable) - line 3
HeartbeatStatus (Variable) - line 28
RuntimeCamera (Variable) - line 47
RuntimeClipRecorder (Variable) - line 52
RuntimeDecodeDiagnostics (Variable) - line 39
RuntimeFacility (Variable) - line 63
StatusSnapshot (Variable) - line 73
SystemSnapshot (Variable) - line 105
```

Verdict: **PASS** — non-empty symbols were returned.

## omo on happy-nursing-home (todo 7 — VERIFY, not install)

Verified 2026-07-30. The plan's greenfield premise was FALSE: omo was already
installed here before Wave 0, so this todo ran as a VERIFICATION and nothing was
installed (reinstalling risks overwriting a working configuration).

| Item | Value |
|---|---|
| omo version, reported BY the remote host | `4.19.3` |
| binary path | `/home/seniorsailab/.local/bin/omo` |
| `ls -d ~/.omo` | `/home/seniorsailab/.omo` |
| active tailnet profile during verification | `seniorsailab@gmail.com` |

Commands that WORK (the three `-o` overrides are mandatory — `~/.ssh/config`
forces `RemoteCommand tmux attach`, so the bare `ssh host 'cmd'` form dies with
"Cannot execute command-line and remote command"):

```
ssh <opts> happy-nursing-home '/home/seniorsailab/.local/bin/omo --version'  -> 4.19.3
ssh <opts> happy-nursing-home 'bash -lic "omo --version"'                    -> 4.19.3
ssh <opts> happy-nursing-home 'bash -lic "command -v omo"'                   -> /home/seniorsailab/.local/bin/omo
```

TWO NEGATIVE CASES, both recorded because each yields a WRONG "not installed"
verdict if taken at face value:
1. `command -v omo` over a NON-interactive shell returns EMPTY — `~/.local/bin` is
   not on the non-interactive PATH. A PATH artifact, not absence.
2. `zsh -lic "omo --version"` returns `bash: line 1: zsh: command not found` — **this host has no zsh**. A probe
   written against zsh reports nothing and looks like a missing install. Use
   `bash -lic`, or the absolute path, which needs no shell rc at all.

The plan's own rule — "a server that installs but does not answer is not installed
for this purpose" — is satisfied only by a probe that actually answers, above.

Third-party-owned RTX PRO 6000 peer: untouched. No driver, kernel module, or
service state changed.
