# 클립 보관·감사 로그 운영 정책

증거 클립(evidence clip)과 클립 감사 로그(audit log)의 보관·회전·접근 기록에 대한
운영/컴플라이언스 참조 문서다. `#131`에서 지적된 세 가지 라이브 격차(감사 로그
커버리지 누락, 감사 파일 무제한 증가, 레거시 액터 처리 경로에 대한 문서 부재)를
다룬다.

## 1. 클립 보관 정책 (retention)

> **현재 상태 (obsolete automatic rotation):** 이 문서가 예전에 설명하던
> `EvidenceRetention.rotate()` / `worker/pipeline/output/evidence/evidence_retention.py`
> 경로는 `db09fc1`에서 삭제됐다. 워커에는 보관 일수나 디스크 상한에 따라
> 클립을 **자동으로 삭제·회전하는 호출 경로가 없다**. 아래 상수·헬퍼는
> `clip_config.py`에 정의만 남아 있고 호출자가 없다 — [#595](https://github.com/SeniorAILab/SeeON-edge/issues/595),
> [`docs/operations/soak-test-plan.md`](soak-test-plan.md) metric 5와 동일한
> 결론이다. 이 절은 **의도된 하한 상수**와 **실제 동작(없음)** 을 구분해 적는다.

클립 자체(영상 파일 + `manifest.json`)의 생성·봉인·게시는 Smart Record /
clip publication 경로가 담당한다:

| 역할 | 심볼 | 파일 |
| --- | --- | --- |
| 일차 클립 생성 | `SmartRecordActor` | `worker/pipeline/output/evidence/smart_record_actor.py` |
| Flow 봉인 클립 게시 | `FlowClipPublisher` | `worker/pipeline/output/evidence/flow_clip_publication.py` |
| 재인코딩 클립 게시 | `ClipPublisher` | `worker/pipeline/output/evidence/clip_publication.py` |
| 스토어 경로·보관 상수 | `MIN_RETENTION_DAYS`, `configured_retention_days`, `configured_disk_high_watermark` | `worker/pipeline/output/evidence/clip_config.py` |

`configured_store_dir()` / `DEFAULT_CLIP_STORE_DIR`만 실제로 소비된다
(`snapshot_store.py`, `worker/runtime/worker.py`). 보관·워터마크 헬퍼는
소비되지 않는다.

### 1-1. 보관 일수 하한 상수 (floor; unused by any deleter)

- `worker/pipeline/output/evidence/clip_config.py` — `MIN_RETENTION_DAYS = 60`,
  `DEFAULT_RETENTION_DAYS = MIN_RETENTION_DAYS`.
- `configured_retention_days()`는 환경변수
  `CLIP_STORE_RETENTION_DAYS` → (없으면) `CLIP_RETENTION_DAYS` 순으로 읽고,
  둘 다 비어 있으면 `DEFAULT_RETENTION_DAYS`를 반환한다. 값이 있어도
  `max(MIN_RETENTION_DAYS, int(raw))`로 **60일 미만으로 내려가지 않도록**
  클램프한다.
- 이 함수를 호출해 클립을 지우는 코드는 현재 트리에 없다. 하한은
  “나중에 회전을 다시 붙일 때 지켜야 할 상수”로만 남아 있다.

| 환경변수 | 우선순위 | 기본값 | 하한 | 소비자 |
| --- | --- | --- | --- | --- |
| `CLIP_STORE_RETENTION_DAYS` | 1순위 | - | 60일 | 없음 |
| `CLIP_RETENTION_DAYS` | 2순위 (fallback) | - | 60일 | 없음 |
| (미설정) | - | 60일 | 60일 | 없음 |

### 1-2. 회전/삭제(purge) 메커니즘 — 없음

`EvidenceRetention.rotate()`, `_verify_candidate()`, `PurgeResult`,
`RotationReport.pressure_blocked`는 `evidence_retention.py`와 함께 삭제됐다.
디스크 상한 헬퍼 `configured_disk_high_watermark()`(기본
`DEFAULT_DISK_HIGH_WATERMARK = 0.80`, 환경변수
`CLIP_STORE_MAX_USAGE` / `CLIP_DISK_HIGH_WATERMARK`)도 정의만 있고 호출자가
없다.

따라서:

- 오래된 클립이 보관 기한·워터마크에 의해 자동 삭제된다고 가정하지 말 것.
- soak metric 5(“clip 디스크 회전 정확성”)는 회전 구현이 다시 생긴 뒤에야
  판정할 수 있다.
- 운영자가 디스크를 비우려면 수동/외부 절차가 필요하며, 그 절차는 이 문서의
  범위 밖이다. 자동 삭제를 복구할 때는 [#595](https://github.com/SeniorAILab/SeeON-edge/issues/595)를
  기준으로 `configured_retention_days()` / `configured_disk_high_watermark()`를
  실제 소비자에 연결해야 한다.

과거 문서가 말하던 `DELETE /api/v1/clips/{clip_id}` + worker
`deletion-preflight` 자동 삭제 파이프라인도 현재 `backend/app/features/clips/router.py`
표면에는 없다(목록·메타데이터·아티팩트·video/thumbnail 조회와 감사 append만
존재). 카탈로그 쪽 `retention_state` 컬럼은
`backend/app/features/clips/catalog_indexer.py`가 읽지만, 워커 자동 purge와
연결된 삭제 API는 없다.

## 2. 감사 로그 커버리지 (audit log coverage)

> **경로 정정:** 과거 JSONL `AuditLogStore` /
> `backend/app/features/clips/audit_log.py`는 없다. 감사는
> `backend/app/features/audit/`(PostgreSQL, `PostgresAuditRuntime` /
> `append_governed`)가 소유한다. 아래 표의 엔드포인트·액션 이름은 클립
> 라우터가 실제로 `append_governed(..., action=AuditAction.*)`를 호출하는
> 현재 표면에 맞춰 읽어야 한다. 표 안의 레거시 JSONL/`label` 행은
> 역사적 서술로 남기되, 파일 경로를 `audit_log.py`로 인용하지 말 것.

클립 API 접근 감사의 현재 액션 상수는 `backend/app/features/audit/catalog.py`의
`AuditAction`에 있다(`CLIP_LIST`, `CLIP_DETAIL`, `CLIP_PLAY`, `CLIP_THUMBNAIL`,
`CLIP_ARTIFACT`, `AUDIT_LIST`, `AUDIT_DETAIL` 등). `#131` 이전 격차(재생/라벨만
기록)를 다루던 서술의 맥락은 아래 표에 남아 있다.

| 엔드포인트 | 액션(action) | clip_id | 액터 해석 |
| --- | --- | --- | --- |
| `GET /api/v1/clips` | `list` | `-` (특정 클립에 국한되지 않음, `AUDIT_NO_CLIP_ID` 상수) | `_authorize()`의 반환값 |
| `GET /api/v1/clips/{clip_id}/video` | `play` | 실제 `clip_id` | 인증된 대시보드 세션 사용자명 |
| `PUT /api/v1/clips/{clip_id}/label` | `label` | 실제 `clip_id` | 요청의 `reviewer` (없으면 인증된 액터로 대체) |
| `GET /api/v1/audit` | `audit-view` | `-` (`AUDIT_NO_CLIP_ID`) | `_authorize()`의 반환값 |

`GET /api/v1/clips`와 `GET /api/v1/audit`는 하나의 클립에 국한되지 않는
액션이므로, `AuditLogStore.append()`가 요구하는 `clip_id` 필드에는 센티널 값
`AUDIT_NO_CLIP_ID = "-"`를 쓴다(`is_valid_clip_id()` 정규식
`^[A-Za-z0-9_-]{1,128}$`를 그대로 통과하므로 기존 검증 경로를 바꾸지 않는다).

`GET /api/v1/audit`는 자기 자신의 열람 기록이 응답에 섞이지 않도록, 먼저
`list_entries()`로 기존 항목을 스냅샷한 뒤 그 스냅샷을 응답으로 반환하고, 그
**다음에** `audit-view` 항목을 append한다(`router.py`의 `list_audit()`). 즉
연속으로 두 번 `GET /audit`를 호출하면 두 번째 응답에서만 첫 번째 호출의
`audit-view` 기록을 볼 수 있다.

모든 `append()` 호출은 기존과 동일하게 `post_backend_backup("clip_audit", entry)`를
통해 백엔드로도 best-effort 전송된다(`API_BACKEND_CLIP_EVENTS_URL` 설정 시).
즉 클립 목록 조회·감사 로그 열람도 다른 감사 액션과 동일하게 백엔드로 미러링된다.

## 3. 감사 파일 회전(rotation) 정책

> **상태:** 아래 JSONL `audit.jsonl` / `AuditLogStore.append()` /
> `API_AUDIT_LOG_MAX_BYTES` / `API_AUDIT_ARCHIVE_RETENTION_DAYS` 서술은
> 삭제된 JSONL 감사 구현을 가리킨다. 현재 감사 저장은
> `backend/app/features/audit/postgres_store.py` 등 PostgreSQL 경로다.
> JSONL 회전·아카이브 prune을 현행 운영 절차로 따르지 말 것. 하한 60일
> 언급이 클립 `MIN_RETENTION_DAYS`와 맞추려던 의도였다는 점만 참고용으로 남긴다.

`#131` 이전에는 `audit.jsonl`이 무한정 누적되는 격차가 있었다. 당시
`AuditLogStore.append()`가 매 기록 전에 현재 파일 크기를 확인하고, 임계값을
넘으면 타임스탬프가 붙은 아카이브 파일로 회전(rotate)했다(아래는 그 역사적
동작 요약이다).

### 3-1. 회전 임계값

- 기본값 `DEFAULT_AUDIT_LOG_MAX_BYTES = 10 MiB` (`10 * 1024 * 1024` 바이트).
- 환경변수 `API_AUDIT_LOG_MAX_BYTES`로 재정의 가능 (바이트 단위 정수). 값이
  없거나 0 이하/파싱 불가면 기본값으로 폴백한다.
- 근거: 감사 로그 한 줄은 `ts`/`actor`/`action`/`clip_id` 네 필드로 약
  120~160바이트다. 10 MiB면 회전 사이에 약 7만~9만 건을 담을 수 있어, 단일
  시설의 재생/라벨링/목록조회/감사열람 트래픽 대비 충분히 여유롭다.
- 라이브 파일이 존재하지 않는 상태(최초 기동)에서는 회전 검사가 그냥
  스킵된다 — 회전은 "이미 쓰여진 파일을 다시 쓸 때"만 의미가 있다.

### 3-2. 회전 방식

- `os.replace(self.path, archive_path)`로 라이브 파일(`audit.jsonl`)을
  아카이브 파일로 **원자적으로 rename**한다. 같은 파일시스템 내 rename은
  POSIX에서 원자적이므로, 동시에 읽는 프로세스는 회전 전 전체 파일(구 경로)
  또는 이미 rename된 아카이브(신 경로) 둘 중 하나만 보게 되며 절반만 쓰인
  파일을 보는 경우는 없다.
- 아카이브 파일명은 `audit-<UTC 타임스탬프, 마이크로초까지>.jsonl` 형식이다
  (예: `audit-20260803T120000123456Z.jsonl`). 같은 마이크로초에 충돌이 나면
  `-1`, `-2` 같은 접미사를 붙여 유일한 이름을 만든다.
- rename 직후 다음 `append()` 호출이 같은 경로(`audit.jsonl`)에 새 파일을
  만들어 이어서 기록한다 — 즉 회전은 기록 흐름을 끊지 않는다.
- rename이 `OSError`로 실패하면(예: 권한 문제) 회전을 건너뛰고 stderr에
  로그만 남긴다 — 회전 실패가 감사 기록 자체를 막지는 않는다(best-effort).

### 3-3. 아카이브 보관(prune) 정책

- 회전이 일어날 때마다, 같은 디렉터리의 `audit-*.jsonl` 아카이브들을 훑어
  파일의 mtime 기준으로 보관 기한이 지난 것을 삭제한다.
- 기본 보관 기간 `DEFAULT_AUDIT_ARCHIVE_RETENTION_DAYS = 60일`, 하한
  `MIN_AUDIT_ARCHIVE_RETENTION_DAYS = 60일`.
- 환경변수 `API_AUDIT_ARCHIVE_RETENTION_DAYS`로 재정의 가능하지만, **60일
  미만으로는 내려가지 않는다** (`max(60, value)`) — 클립 자체의 보관 하한
  (`worker/pipeline/output/evidence/clip_config.py`의
  `MIN_RETENTION_DAYS = 60`)과 정확히 동일한 하한을 감사 아카이브에도 걸어서,
  "클립은 사라졌는데 그 클립에 대한 감사 기록(누가 언제 재생/라벨링했는지)도
  같이 사라지는" 상황이 나지 않게 한다. 감사 아카이브는 항상 클립 자체와
  같거나 더 오래 남는다.
- 아카이브 삭제가 `OSError`로 실패하면(권한 등) 해당 파일은 건너뛰고
  stderr에 로그만 남긴다 — 다음 회전 때 다시 시도된다.

| 환경변수 | 기본값 | 하한 | 비고 |
| --- | --- | --- | --- |
| `API_AUDIT_LOG_MAX_BYTES` | 10 MiB (`10485760`) | 없음(0 이하는 무시하고 기본값 사용) | 라이브 파일 회전 임계값 |
| `API_AUDIT_ARCHIVE_RETENTION_DAYS` | 60일 | 60일 | 아카이브 보관 기간, 클립 보관 하한과 동일 |

## 4. 감사 액터 인증 경계

클립 API는 서버가 발급한 HttpOnly 대시보드 세션 쿠키만 운영자 권한으로
인정한다. 워커 relay 토큰은 `Authorization` 헤더, `X-Edge-Relay-Token`
헤더, `token` 쿼리 중 어느 형태로 보내도 클립 목록·재생·라벨링·감사 로그
열람 권한이 되지 않는다.

`dashboard_sessions()`는 영구 저장된 자격증명을 우선 사용하고, 아직 회전하지
않은 설치에서는 배포 시 명시한 `API_DASHBOARD_USERNAME` /
`API_DASHBOARD_PASSWORD` 부트스트랩 쌍을 사용한다. 두 값이 없거나 불완전하거나
저장소를 읽을 수 없으면 503으로 실패하며 내장 `admin`/`admin` 폴백은 없다.

따라서 감사 로그의 `actor` 필드는 실제 대시보드 세션 사용자명(또는 라벨링 시
명시적으로 지정된 `reviewer`)이다. 과거의 relay-token 호환 분기와
`"operator"`/`"bearer"`/`"legacy-dashboard"` 제네릭 actor는 제거되었다.

## 관련 문서

- [`docs/architecture.md`](../architecture.md) — 전체 아키텍처, 워커/API 레이어 구성.
- `worker/pipeline/output/evidence/clip_config.py` — 보관·워터마크 상수/헬퍼(자동 삭제 소비자 없음; [#595](https://github.com/SeniorAILab/SeeON-edge/issues/595)).
- `worker/pipeline/output/evidence/smart_record_actor.py`,
  `flow_clip_publication.py`, `clip_publication.py` — 클립 생성·게시.
- `backend/app/features/clips/router.py` — 클립 조회 API와 `append_governed` 감사 호출.
- `backend/app/features/audit/` (`catalog.py`, `postgres_store.py`, `router.py`) — 현재 감사 구현(과거 `backend/app/features/clips/audit_log.py` JSONL `AuditLogStore`는 없음).
- [`docs/operations/soak-test-plan.md`](soak-test-plan.md) — metric 5가 동일하게 “자동 회전 없음”을 기록.
