# 핸드오프 — 낙상 오탐 폭주 조사

**작성:** 2026-08-28 08:40 UTC (17:40 KST)
**대상 스택:** `seeon-edge-main` (요양원 라이브)
**이슈:** #455
**전제:** 이 문서의 모든 수치는 실측입니다. 가설은 가설이라고 명시했습니다.

---

## 1. 무엇이 일어나고 있나

```
GET /api/v1/clips?limit=1
  event_type_counts: {"fall": 8060, "bed-exit": 2325}   total 10385

최근 클립 100건 (GET /api/v1/clips?limit=100)
  구간          2026-08-28T08:19:17Z .. 08:29:47Z   (10.5분)
  환산 빈도     시간당 572건
  이벤트 종류   fall 100 / 100
  카메라 분포   19c97354:12  31c73e96:11  f13a913e:11  632f31a9:11  0bdd3414:10
                4fc0230f:8   bd260bd4:7   9737eb85:7   79874785:7   d8bfa299:5
                62c1c28e:5   3a1f2d82:5   324400e5:1        (13대 전부)
  클립 길이     최소 55.8s  중앙값 59.1s  최대 65.0s
```

`GET /api/v1/incidents?limit=5` — 같은 속도로 알림 계층까지 도달합니다.
11초 안에 서로 다른 3대에서 `fall` 생성, 전부 `lifecycle_state: OPEN`,
`event_delivery_state: ACKED`.

**해석:** 13대가 각각 약 7분에 한 번 낙상을 선언하고, 클립 길이는 이벤트 모양이
아니라 **최대 녹화 창(약 59초)에 붙어 있습니다.** 실제 낙상이 이 빈도일 수 없으므로
**진짜 낙상은 노이즈에 묻힙니다.** 알림 채널이 목적을 잃은 상태입니다.

> 이것은 "정확도가 조금 낮다"가 아닙니다. 장면·시간·카메라와 무관하게 균일하다는 것은
> 탐지기가 사실상 자유 발진(free-running)한다는 뜻입니다.

---

## 2. 측정 전에 반드시 고정할 것 (§5 규율)

이전 세션에서 재시작·백로그·잘못된 마운트가 측정을 오염시킨 전례가 있습니다.
**어떤 수치를 보고하기 전에 아래 4개를 먼저 출력하십시오.**

```bash
docker inspect seeon-edge-main-ml-worker-1 \
  --format '{{.Config.Image}} {{.Image}} restarts={{.RestartCount}} started={{.State.StartedAt}}'
docker inspect seeon-edge-main-ml-worker-1 \
  --format '{{range .Mounts}}{{if eq .Destination "/var/lib/clip-store"}}{{.Source}}{{end}}{{end}}'
docker exec seeon-edge-main-ml-worker-1 ls /var/lib/seeon-state/delivery-queue | wc -l
curl -s -b <cookiejar> http://127.0.0.1:8000/api/v1/status | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["cameras"]))'
```

**현재 기준선 (2026-08-28 08:30 UTC):**

| 항목 | 값 |
|---|---|
| ml-worker | `seeon-edge-final:a681ddc-epochfix` / `sha256:3c290c9e…` / restarts=0 / started 2026-08-27T17:48:31Z |
| ml-api | `seeon-edge-api:4befda5-cliplist2` (문서 작성 후 `*-headfix`로 교체 예정, #452) |
| clip-store 마운트 | `/home/seniorsailab/.local/share/eldercare/clip-store-main` |
| 배달 큐 | 0 |
| 카메라 | 13/13 online |
| 클립 영상 부착률 | **314/314 = 100%** (#429 수정 후 1시간 측정) |

마운트가 `clip-store-main`이 아니면 다른 디렉터리를 재고 있는 것입니다.
(그 사고를 낸 stale `clip-store` 디렉터리는 2026-08-28에 삭제했습니다.)

대시보드 인증은 `.env.seeon-edge-main`의 `API_DASHBOARD_USERNAME/PASSWORD`로
`POST /api/v1/auth/session` → `ml_dashboard_session` 쿠키.

---

## 3. 확실한 것과 불확실한 것

### 확실한 것 (측정됨)

- 빈도·분포·클립 길이 (§1)
- incident까지 도달함 (§1)
- 미디어 경로는 정상. 클립 영상 부착률 100%, `/clips` 응답 0.3초
- 파이썬은 스트림을 받지도 디코드하지도 CNN을 돌리지도 않음.
  RTSP → 네이티브 C++ 자식(`rtspsrc → nvv4l2decoder → nvstreammux → TensorRT`)이 전부 처리하고,
  파이썬은 AF_UNIX로 **인코딩 AU / NVJPEG 미리보기 / 숫자 메타데이터**만 받음
  (`worker/adapters/decode/native_au_receiver.py`, `native_preview_receiver.py`, `NativePolicyPump`)
- 낙상 LSTM은 **CPU**에서 돌고 있고(`worker/adapters/model/torch_lstm_fall.py:92` `device="cpu"`),
  스레드 클램프가 없어 워커 CPU의 52%를 OpenMP 워커가 먹고 있음 (#449)

### 불확실한 것 (가설)

- 왜 발진하는지. 아래 §4의 세 가설 중 어느 것도 아직 검증되지 않았습니다.
- ~100초마다 30초 반속 구간이 이 문제와 관련 있는지 (#449 미해결 항목)

### 믿지 말아야 할 것

- **incident 레코드는 판정 근거를 하나도 담고 있지 않습니다.**
  ```json
  {"event_type":"fall","detected_at":"…","lifecycle_state":"OPEN",
   "decision_trace_id":null,"policy_qualified_id":null,"module_qualified_id":null,
   "primary_clip_id":null,"snapshot_artifact_state":"UNAVAILABLE"}
  ```
  확률값·임계값·추적 id·정책 id가 전부 없고, 그 이벤트로 녹화된 클립과의 링크조차 없습니다
  (클립은 `event_ref`를 갖지만 incident는 `primary_clip_id`가 null).
  **저장된 데이터로는 "무슨 점수로 왜 울렸나"를 물어볼 수 없습니다.**
  이것이 이 문제가 8,060건 쌓일 때까지 아무도 분류하지 못한 이유입니다.

---

## 4. 가설 (우선순위 순)

### ① 분류기 입력 계약 불일치 — 가장 유력

`worker/domains/registry.py:62`

```python
"fall-classifier": "legacy-coco17-xyc-frame-normalized-zero-fill-v1"
```

분류기는 **COCO-17 키포인트, (x, y, confidence), 프레임 정규화, zero-fill**을 기대합니다.
이 계약은 호스트 어댑터 시절에 쓰인 것이고, 지금 포즈는 네이티브 TensorRT
(`yolo26n-pose`, `worker/native/deepstream/src/trt_perception.cpp`)에서 옵니다.

키포인트 **순서**, **정규화 기준**(프레임 크기 vs 박스 크기 vs 0~1 vs 픽셀),
**미검출 관절의 zero-fill 규약** 중 하나만 달라도 LSTM은 매 윈도우 분포 밖 입력을 받고
출력이 포화됩니다. 지금의 균일한 발진 패턴이 정확히 그 모양입니다.

관련 이력: #424 (PerceptionFrameV1이 `NativePolicyPump`에 도달하지 않아 판정 0건이던 문제)
— 그 배관이 뚫린 뒤 이 폭주가 보이기 시작했는지 타임라인을 확인할 가치가 있습니다.

### ② `operating_threshold`가 런타임에 적용되지 않음

`worker/adapters/model/torch_lstm_fall.py:99,120-153` — #217에서 런타임 구성 값으로
바뀌었습니다. 구성된 값이 실제로 분류기에 도달하는지, 아니면 매니페스트 기본값이
쓰이는지 확인해야 합니다.

### ③ 래치 / 상승엣지 재무장

`worker/domains/fall/detector.py:41-72`, `FallEventLatch`.
카메라당 약 7분 간격은 **상승엣지라기보다 쿨다운 주기처럼** 보입니다.
상태가 계속 참인데 쿨다운마다 재발화하는 것이라면 래치 문제입니다.

### ④ 모델 자체의 품질

체크포인트는 `889075695884742475b9713e3b86ba67085bb96979b64c51756ea3fd715ab57a`로 고정돼
있습니다(`worker/domains/registry.py:54`). **①~③을 배제한 다음에만** 라벨 세트로
오프라인 평가하십시오. 배관 문제를 모델 탓으로 돌리면 재학습에 시간을 버립니다.

---

## 5. 조사 절차

### 단계 1 — 확률 분포를 본다 (가장 싸고 가장 판별력 높음)

카메라별로 낙상 확률의 **히스토그램**을 10분간 수집합니다. 프레임마다 로그를 뿌리지 말고
버킷 카운터로 집계하십시오(액터 스레드에서 동기 I/O를 하면 §6의 함정에 빠집니다).

판별:

| 관측 | 결론 |
|---|---|
| 확률이 1.0 근처에 포화 | **가설 ①** — 입력 계약 불일치 |
| 임계값 바로 위에서 진동 | **가설 ②/③** — 임계값 또는 래치 |
| 확률은 낮은데 이벤트가 남 | 판정 이후 단계(래치/발행) 문제 |

읽기 전용으로 지금 당장 할 수 있는 근사치: `docker logs` 에 확률이 남지 않으므로
**코드 변경 없이는 볼 수 없습니다.** 임시 진단 로그를 넣는다면 별도 브랜치·별도 이미지로
띄우고, 라이브 교체 전에 이 문서 §2의 기준선을 다시 고정하십시오.

### 단계 2 — 키포인트를 직접 대조한다

한 카메라에서 네이티브 포즈 출력을 캡처해 `legacy-coco17-xyc-frame-normalized-zero-fill-v1`
계약과 항목별로 비교합니다. 확인할 것:

- 관절 순서(COCO-17 인덱스)와 좌우 정의
- x, y의 정규화 기준과 범위
- confidence 스케일 (0~1 vs 0~100)
- 미검출 관절의 값 (0 채움 vs 이전 값 유지 vs NaN)
- 입력 윈도우 길이·스트라이드가 학습 시와 같은지

### 단계 3 — 근거를 저장하게 만든다 (이 문제와 별개로 필요)

incident에 `decision_trace_id`, `policy_qualified_id`, 확률값, `primary_clip_id`를 채웁니다.
지금은 사후 분석이 원천적으로 불가능합니다. 이 작업이 되면 이후 모든 조사가 쉬워집니다.
(#299가 "이벤트 설명 / 오탐 귀속"을 다루고 있으니 중복 설계를 피하려면 먼저 읽으십시오.
이번 세션에서 #299는 손대지 않았습니다.)

### 단계 4 — 모델 오프라인 평가

①~③이 배제된 뒤에만. 라벨된 클립 세트 대비 ROC/PR과 운영 임계값 재산정.

---

## 6. 함정 (이전 세션에서 실제로 빠진 것들)

1. **재시작이 상태를 리셋해 "고쳤다"는 착각을 만듭니다.** 클립 잠금 수정이 동작한다고
   보고했으나 실제로는 재시작이 플래그를 리셋한 것이었습니다.
2. **백로그 배출을 신규 이벤트로 오독했습니다.** "8건 낙상이 방금 발생"이라 보고했으나
   배달 큐 백로그였습니다.
3. **잘못된 마운트를 재고 "클립 0건"이라 보고했습니다.** §2를 먼저 고정하십시오.
4. **성공은 로그를 남기지 않습니다.** `clip published without video`는 실패에만 찍힙니다.
   성공 건수는 매니페스트에서 세십시오.
5. **셸 환경변수가 `--env-file`을 덮어씁니다.** 재배포는 반드시
   `env -u COMPOSE_PROJECT_NAME -u CLIP_STORE_HOST_DIR -u ML_WORKER_IMAGE -u ML_API_IMAGE` 로.
6. **`ClipMaintenance.rotate`는 액터 스레드에서 동기 glob·JSON 파싱을 합니다.**
   진단 코드를 이 경로에 넣으면 증거 트래픽이 유실됩니다.

---

## 7. 오늘(2026-08-28) 라이브에 반영된 것

| 변경 | 커밋/PR | 효과 |
|---|---|---|
| #429 epoch 4중 원인 수정 | #427 `a681ddc` | 클립 영상 부착률 3~75% → **100%** (314/314) |
| clips 목록 카탈로그 우선 조회 | #439, #442 | `GET /clips` **300초 → 0.3초** (대시보드가 비어 보이던 원인) |
| relay 로컬 수락 명명 | #433 | 재시도 루프 273회 → 0, 큐 3,691 → 0 |
| 스키마 18 단일화 | #437 | 마이그레이션 원장 −15.9k 줄 |
| CI 병렬화 | #443, #444 | 19분37초 ×2 → **7분47초 ×1**, 액션 SHA 고정 |
| 모델 프로비저닝 | #441 | HF/ultralytics 해시 검증 `edge-model-fetch` |

미배포/진행 중: #452 (미디어 라우트 HEAD 404) 수정 및 ml-api 재배포.

---

## 8. 관련 이슈

| 이슈 | 내용 |
|---|---|
| **#455** | 본 문제 — 시간당 572건 낙상 발진 |
| #299 | 이벤트 설명 / 오탐 귀속 (열려 있음, 이번 세션 미개입) |
| #424 | PerceptionFrameV1이 pump에 도달하지 않던 문제 (타임라인 대조용) |
| #449 | 워커 398 스레드 / CPU 52%가 batch-1 LSTM의 OpenMP 워커 |
| #448 | 낙상 모델 GPU 이전 검토 — 먼저 `OMP_NUM_THREADS=1` |
| #430 / #428 | 추론 컨텍스트 직렬화 / 페이싱 ~11fps |
| #450 / #451 / #452 | MJPEG 무제한 스레드, 네이티브 restart 잠재 누수, HEAD 404 |

---

## 9. 한 줄 요약

미디어 경로(클립 영상·조회 속도)는 오늘 복구됐습니다. 남은 것은 **판정 품질**이며,
지금 상태로는 알림 채널이 사용 불가입니다. 조사는 **모델이 아니라 배관부터** —
입력 계약(①) → 임계값(②) → 래치(③) → 모델(④) 순서로, 그리고 그 전에
**판정 근거를 저장하게** 만드십시오.
