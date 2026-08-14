# Edge clip consistency repair

Use this runbook only for the schema-9 worker database repair shipped with
`repair_clip_consistency.py`. It reconciles `clip_events` from validated final
READY/UNAVAILABLE manifests. It does not modify events, clip records, final
manifests, or final media.

Do not run apply or resume against a live worker. Keep the generated journal,
backup, and receipts in the worker state volume. Apply defaults to off; a plain
invocation is a dry-run.

## 1. Select the production Compose command

Run from the deployed repository directory and use the same ordered Compose
files used to start Edge:

```sh
cd /opt/eldercare-fall-ml
export EDGE_DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'
# CPU-only host, instead:
# export EDGE_DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml -f compose.edge.cpu.yaml'
DC="$EDGE_DC"
```

The selected `ml-worker` image must contain
`/app/scripts/repair_clip_consistency.py` and its baked
`CLIP_CONSISTENCY_TOOL_REVISION`. The command intentionally uses the independent
persisted paths and explicit production authorities from `compose.edge.yaml`:

```text
state database: /root/.local/state/ml-worker/worker-state.sqlite3
clip store:     /var/lib/clip-store
maintenance:    /root/.local/state/ml-worker/clip-consistency-maintenance
state authority: UID/GID 0:0, DB mode 0644, state directory mode 0755
clip authority:  UID/GID 1000:1000, root/clips/.staging mode 0775
```

These are separate trust domains, not owners to discover from the filesystem.
Do not substitute the clip-store path for the state database path, copy an
observed unexpected owner into the command, or use `chown` to make a refusal go
away. A swapped owner, mixed final/staging owner, changed mode, unsafe writable
ancestor, symlink, lexical `..`, or revision mismatch is an incident signal.

## 2. Stop all worker-state writers and create operator proof

Stop the normal worker first. `--no-deps` on every one-off command prevents
Compose from starting another service.

```sh
$DC stop ml-worker
test -z "$($DC ps --status running -q ml-worker)"

$DC run --rm --no-deps --entrypoint python ml-worker - <<'PY'
import json
import os
import time
from pathlib import Path

root = Path('/root/.local/state/ml-worker/clip-consistency-maintenance')
root.mkdir(mode=0o700, parents=True, exist_ok=True)
os.chmod(root, 0o700)
now = int(time.time())
from worker.pipeline.output.evidence.clip_consistency_authority import RepairAuthority
from worker.pipeline.output.evidence.clip_consistency_operation import image_artifact_identity

authority = RepairAuthority(
    state_uid=0,
    state_gid=0,
    state_db_mode=0o644,
    state_dir_mode=0o755,
    clip_uid=1000,
    clip_gid=1000,
    clip_dir_mode=0o775,
    tool_revision=os.environ['CLIP_CONSISTENCY_TOOL_REVISION'],
)
receipt = {
    'format_version': 3,
    'state_db': '/root/.local/state/ml-worker/worker-state.sqlite3',
    'clip_store': '/var/lib/clip-store',
    'stopped_service': 'ml-worker',
    'stopped_db_writers': ['event', 'config', 'fault'],
    'operator_uid': 0,
    'authority_sha256': authority.sha256,
    'operation_digest_version': 1,
    'operation_digest': '0' * 64,  # bound by apply before PREPARED
    'image_artifact_identity': image_artifact_identity(authority),
    **authority.to_dict(),
    'issued_at': now,
    'expires_at': now + 1800,
}
path = root / 'quiescence.json'
path.write_text(json.dumps(receipt, sort_keys=True) + '\n', encoding='utf-8')
os.chmod(path, 0o600)
PY
```

The receipt is a short-lived assertion by the operator; it does not stop a
process. If it expires, first re-check that `ml-worker` is stopped, then recreate
it with the command above. Apply also acquires the clip-store lock and
`BEGIN IMMEDIATE`; either lock failing is a refusal, not a reason to bypass the
check.

## 3. Dry-run

```sh
$DC run --rm --no-deps --entrypoint sh ml-worker -ec '
  exec python scripts/repair_clip_consistency.py \
    --state-db /root/.local/state/ml-worker/worker-state.sqlite3 \
    --clip-store /var/lib/clip-store \
    --state-uid 0 --state-gid 0 \
    --state-db-mode 0644 --state-dir-mode 0755 \
    --clip-uid 1000 --clip-gid 1000 --clip-dir-mode 0775 \
    --tool-revision "$CLIP_CONSISTENCY_TOOL_REVISION"
'
```

Save the single JSON receipt. Review `mismatch_clips`, `mismatch_tuples`,
`sql_relations_deleted`, `sql_relations_inserted`, and `staging_to_delete`.
Logical mismatch counters are intentionally distinct from SQL mutation counts.
Any refusal must be investigated; do not edit a manifest or bypass schema,
path, lease, lock, or integrity validation.

## 4. Apply

Use a new journal path. Never delete or overwrite an existing journal to force
another apply.

```sh
$DC run --rm --no-deps --entrypoint sh ml-worker -ec '
  exec python scripts/repair_clip_consistency.py \
    --state-db /root/.local/state/ml-worker/worker-state.sqlite3 \
    --clip-store /var/lib/clip-store \
    --state-uid 0 --state-gid 0 \
    --state-db-mode 0644 --state-dir-mode 0755 \
    --clip-uid 1000 --clip-gid 1000 --clip-dir-mode 0775 \
    --tool-revision "$CLIP_CONSISTENCY_TOOL_REVISION" \
    --apply \
    --maintenance-root /root/.local/state/ml-worker/clip-consistency-maintenance \
    --journal /root/.local/state/ml-worker/clip-consistency-maintenance/apply.json \
    --quiescence-receipt /root/.local/state/ml-worker/clip-consistency-maintenance/quiescence.json
'
```

Apply replaces the proof's all-zero `operation_digest` placeholder in place
before durable PREPARED. The same versioned digest is written to the backup
receipt, PREPARED/DB_COMMITTED/DONE journal, and command receipt. It binds the
canonical paths, split authority and modes, packaged image identity, descriptor
identities, source/backup facts, schema and relation hashes, exact plan, and the
ordered quarantine set. Do not edit or regenerate any of those artifacts.

Apply creates an owner-only online SQLite backup and strict receipt before any
relation mutation. The receipt binds all advertised source DB/WAL raw facts,
the verified logical backup, and the exact schema. Success is a JSON receipt
with `state` equal to `DONE`. Record its `backup_receipt_path` and
`journal_path`.

## 5. Resume an interrupted apply

Do not restart the worker when apply exits nonzero after creating a journal.
Re-check quiescence, recreate the short-lived receipt, and run:

```sh
$DC run --rm --no-deps --entrypoint sh ml-worker -ec '
  exec python scripts/repair_clip_consistency.py \
    --state-db /root/.local/state/ml-worker/worker-state.sqlite3 \
    --clip-store /var/lib/clip-store \
    --state-uid 0 --state-gid 0 \
    --state-db-mode 0644 --state-dir-mode 0755 \
    --clip-uid 1000 --clip-gid 1000 --clip-dir-mode 0775 \
    --tool-revision "$CLIP_CONSISTENCY_TOOL_REVISION" \
    --resume \
    --maintenance-root /root/.local/state/ml-worker/clip-consistency-maintenance \
    --journal /root/.local/state/ml-worker/clip-consistency-maintenance/apply.json \
    --quiescence-receipt /root/.local/state/ml-worker/clip-consistency-maintenance/quiescence.json
'
```

Resume is idempotent. It first revalidates both explicit authorities against
all state and recursively scanned clip paths, then requires the proof, backup
receipt, and journal to carry the identical paths, modes, owners, tool revision,
descriptor identities, and operation digest. Revalidation occurs under the clip
lock before PREPARED, immediately before and after commit classification, before
each descriptor-backed quarantine deletion, and before returning DONE.

Any inode/owner/mode/content drift fails closed. Before commit, a durable
PREPARED journal and quarantine are preserved; after commit, PREPARED or
DB_COMMITTED is preserved according to the durable relation hashes; during
cleanup, each fsynced deletion is recorded before the next entry; drift during
the final DONE write downgrades a still-trusted journal to DB_COMMITTED. The
tool never deletes a changed quarantine entry and never returns a DONE receipt
for a drifted authority. A PREPARED journal then requires the schema and every
non-`clip_events` logical row to match the complete preimage, then determines
whether `clip_events` is at the exact before or after hash. DB_COMMITTED
completes canonical quarantine deletion; DONE returns the existing receipt.
ABORTED records a proven pre-commit rollback and is not converted into a new
apply. UNKNOWN records an ambiguous commit whose durable state matched neither
complete boundary and always fails closed. A `resume_conflict` or UNKNOWN state
requires incident review with the journal, quarantine, and backup preserved.

## 6. Verify and restart

Run the dry-run command from step 3 again. Success requires exit status 0 and
`changes: 0`. Verify that the backup and receipt paths reported by apply remain
inside the maintenance root, then restart only the normal worker:

```sh
$DC up -d --no-deps ml-worker
container="$($DC ps -q ml-worker)"
test -n "$container"
$DC logs --tail 100 ml-worker
```

Do not manually remove quarantine directories, journals, backups, or receipts.
Do not restore a backup over a post-commit database merely because cleanup was
interrupted; resume is the recovery path. A backup restore is a separate,
approved incident operation performed with the worker stopped and the current
database/WAL/SHM preserved for diagnosis.
