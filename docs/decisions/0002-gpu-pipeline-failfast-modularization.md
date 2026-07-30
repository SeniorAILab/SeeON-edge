# 2. GPU Inference Pipeline Fail-Fast Modularization

- Status: Accepted
- Date: 2026-07-20
- Deciders: product owner + agent (deep-interview → ralplan consensus)
- Sources: `.gjc/_session-019f7fb9-5e69-7000-a7f8-682734d93ae9/specs/deep-interview-gpu-pipeline-failfast.md`, ralplan `pending-approval.md`

## Context

Docker 환경에서 GPU 추론이 CPU로 silent fallback되고(NVDEC 초기화 실패 후 OpenCV로 fallback), warmup 이후 worker가 멈추는 문제가 관찰됐다. 근본 원인은 `edge/runners/device.py:select_device()`가 CUDA 프로브 실패 시 조용히 `"cpu"`를 반환하고, `edge/sources/rtsp.py`의 `"auto"` 디코드 백엔드가 NVDEC→OpenCV로 조용히 강등되며, `edge/runners/warmup.py`가 실질 no-op이라 첫 실추론이 lazy CUDA 컨텍스트 초기화에서 블록되는 구조에 있다.

최종 목표는 최대 50대 카메라 실시간 추론 서빙이며, 초기 실증은 2대(640×360@30fps, 단일 RTX 5070 Ti) 규모다.

## Decision

1. **Fail-fast 경계 (init fail-fast, runtime resilient)**: 필수 인프라(GPU/CUDA 검증, inference backend init(모델 로드 포함), warmup) 실패 시 silent CPU/OpenCV fallback을 하지 않고 명확한 오류와 함께 프로세스를 non-zero로 즉시 종료한다. 카메라별 단계(NVDEC init, worker loop) 실패는 해당 카메라만 DEGRADED로 격리하고 나머지는 유지한다. 이 원칙은 NVENC 클립 인코딩 경로에도 적용된다(CPU 암묵 폴백 금지).

2. **6단계 2계층 staged bootstrap**: 신규 `edge/runtime/pipeline_bootstrap.py`가 전역 1회 [GPU/CUDA 검증 → inference backend init(serving_client runner 번들) → warmup(실제 forward + `torch.cuda.synchronize()` + readiness 게이트)] → 카메라별 N회 [NVDEC init → worker loop] → runtime health(전역/카메라 관통) 순서로 실행한다. fail-fast의 `sys.exit`는 오케스트레이터에만 두고 `device.py`는 프로브/정책 반환만 한다.

3. **decode `auto` = fail-loud**: `decode_backend="auto"` 토큰은 유지하되 NVDEC 실패 시 OpenCV로 조용히 내려가지 않고 명확한 오류로 처리한다(전역이면 프로세스, 카메라별이면 DEGRADED). 명시적 `decode_backend=cpu|opencv`, `device=cpu` opt-in 경로는 보존한다(크게 로깅).

4. **serving_client 심 진화 계약 (50대 경로)**: runner/serving 계약을 배치-입력(프레임 리스트) 수용 형태로 정의한다(단일-프레임 하위호환). 향후 networked 배칭 추론 서비스로 재작성 없이 교체 가능하도록 교체지점을 ADR-lite로 남긴다. 배칭 백엔드는 지금 구현하지 않는다.

5. **decisions-first (init-first)**: 이 결정을 AGENTS.md/ADR에 먼저 고정한 뒤 코드를 시작한다. AGENTS.md가 모든 후속 에이전트 작업을 지속적으로 규율하므로 스테일 서술을 방지한다.

6. **무회귀 허용목록**: acceptance는 기존 pytest green 유지가 기본이나, 의도적으로 변경되는 계약은 예외로 명시한다 — (i) `select_device`의 "cpu 반환" 기대 테스트, (ii) `"auto"` silent OpenCV fallback 기대 테스트, (iii) 변경 시 import-linter 계약. import-linter 무회귀는 하드 게이트에서 제외한다.

## Consequences

- `select_device` / `rtsp.py auto` / `warmup` 동작이 변경되어 일부 기존 테스트를 의도적으로 갱신한다.
- 신규 `pipeline_bootstrap.py`는 `edge/runtime`(최상위 계층)에 위치하여 하위 심(serving_client/runners/sources)을 import하므로 import-linter 계층 계약을 위반하지 않는다.
- 8월 컷오버는 이 호스트에서 `docker build` + `docker compose` 스왑으로 기존 `ml-worker`/`ml-api` 스택을 대체하며, 기존 인스턴스의 CCTV 접근 정보(`.env.edge.prod`/secrets)를 재사용하고 롤백(기존 이미지 태그 보존)을 포함한다.

## Deferrals (트리거 기반)

- **50대 배칭 추론 백엔드** (Triton류, serving_client 심 뒤): 8월엔 설계만. 실제 용량은 실측으로 확정. 베이스라인 = 단일 RTX 5070 Ti(NVENC 동시 12세션), 크로스-카메라 마이크로배칭 전제.
- **NVENC 12세션 포화 처리**: 범위 제외. 명시 트리거 = 12세션 포화 발생 시 NVIDIA 프레임워크(DeepStream/Triton) 채택 또는 전용 최적화. 2대에선 미도달.
- **2대째 카메라**: 연결정보 도착 시 확정.

## Root-Cause Diagnosis (verified 2026-07-20, in `ml-worker` container)

증상("GPU 추론이 CPU로 fallback")의 실제 원점은 코드 로직이 아니라 **torch 의존성 핀**이다.

- 환경: RTX 5070 Ti(Blackwell, driver 580.159.03, 16GB), `nvidia-smi` 정상, 컨테이너 `torch.cuda.device_count()==1`(NVML로 GPU 보임).
- 그러나 컨테이너 `torch 2.12.0+cu130`에서 `torch.cuda.is_available()==False`, `torch.cuda.get_arch_list()==[]`, `torch.cuda.init()` → `RuntimeError: No CUDA GPUs are available`.
- 즉 설치된 torch wheel에 **컴파일된 CUDA 아키텍처 커널이 하나도 없다**(빈 arch list) → Blackwell `sm_120` 커널 부재 → CUDA 사용 불가 → `select_device()`가 조용히 `"cpu"` 반환.
- 원인 체인: `pyproject.toml`의 `torch>=2.3`가 CUDA-아키텍처 인식 소스 없이 PyPI 기본 registry(`https://pypi.org/simple`)에서 해석됨(`uv.lock` torch source = pypi.org/simple, wheel `torch-2.12.0-cp311-...-manylinux_2_28_x86_64.whl`).

### 처방(Fix)

1. `pyproject.toml`에 Blackwell(`sm_120`) 커널을 포함하는 CUDA torch 빌드를 명시 핀한다 — `[[tool.uv.index]]`/`[tool.uv.sources]`로 PyTorch CUDA wheel 인덱스(cu128+)를 지정, 또는 `sm_120`을 포함하는 torch 버전으로 고정.
2. `uv lock` 재생성 → `Dockerfile.edge`(`uv sync --frozen ... --group worker`) 재빌드.
3. 컨테이너에서 `torch.cuda.is_available() is True`이고 `torch.cuda.get_arch_list()`가 `sm_120`/`compute_120`을 포함함을 검증(= acceptance 2의 실CUDA 근거).
4. 이후 fail-fast bootstrap이 정상 통과하며, 환경이 재차 깨지면 조용한 CPU 강등 대신 **크게 실패**한다 — 이것이 이 재설계의 핵심 가치.

### Fix applied + verification status (2026-07-20)

- **1차 원인 수정·검증 완료**: `pyproject.toml`에 `[[tool.uv.index]] pytorch-cu130`(explicit) + `[tool.uv.sources] torch/torchvision` 추가 → `uv lock`로 torch **2.12.0(PyPI, 빈 arch_list)** → **2.13.0+cu130 (source: download.pytorch.org/whl/cu130)** 로 교체. `uv sync` 후 `torch.cuda.get_arch_list()`가 **`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`** — Blackwell `sm_120` 포함 확인. (빈 arch_list 근본원인 해소)
- **2차 이슈 발견(미해결)**: 수정된 torch에서도 호스트 `torch.cuda.is_available()==False`, `torch.cuda.init()` → `RuntimeError: No CUDA GPUs are available`. 이때 `device_count()==1`(NVML로 GPU 보임), driver 580.159.03/CUDA 13.0, `/dev/nvidia*` 존재, `CUDA_VISIBLE_DEVICES` 미설정 — 즉 CUDA **컨텍스트 초기화** 단계 실패. 이는 호스트 샌드박스가 NVML 열거는 허용하되 CUDA 컴퓨트 컨텍스트를 막는 아티팩트일 수 있고, 실제 `ml-worker` 컨테이너(정상 nvidia 런타임)에서는 다를 수 있음.
- **다음 검증**: 수정된 lock으로 `ml-worker` 이미지 재빌드 후 컨테이너에서 `torch.cuda.is_available()`를 확인해야 2차 이슈가 호스트-한정인지 실환경 문제인지 판별된다(= acceptance 2의 최종 근거). 컨테이너에서도 실패하면 driver/CUDA-13/Blackwell 호환(드라이버 업그레이드 등) 시스템 진단이 추가로 필요하다.

### Driver / CUDA pairing — 2차 원인 확정 (G006, 2026-07-20)

2차 이슈("No CUDA GPUs are available")를 드라이버 레벨에서 확정했다. 이건 샌드박스 아티팩트도 torch 문제도 아니다.

- **ctypes 프로브(호스트)**: `libcuda.so.1`의 `cuInit(0)` → **`100` (CUDA_ERROR_NO_DEVICE)**, `cuDeviceGetCount` → `3` (NOT_INITIALIZED), `cuDriverGetVersion` → `13000` (CUDA 13.0). `nvidia_uvm` 커널 모듈 로드됨, `/dev/nvidia0`·`/dev/nvidia-uvm` 권한 `crw-rw-rw-`. 즉 커널 모듈·디바이스 노드·권한은 정상인데 **드라이버 API `cuInit`이 실패**한다.
- **원인**: **NVIDIA 드라이버 580.x + CUDA 13.0 페어링의 알려진 breakage**. 커뮤니티 재현(level1techs #236558, 동일 driver 580.65.06/CUDA 13.0/A100)에서 동일 증상(`cuInit(0)=100`, `nvidia-smi`는 정상)이 보고됐고, 결론은 "CUDA 13.0은 지금 매우 broken(특히 container toolkit); CUDA 12.9 + driver 575.57.08 같은 known-good 페어링이면 동작"이다.

**해결(OPS 스텝 — 호스트 root/재부팅 필요, 자율 불가)**:
1. 호스트 NVIDIA 드라이버/CUDA를 **검증된 페어링으로 정렬**한다. 커뮤니티 확인 조합: **CUDA 12.9 + driver 575.57.08**(+ `nvidia-open`, cuDNN 9.x). 프로덕션 노드는 낙상감지 공백을 피하려 롤백 계획하에 저위험 시간대에 수행.
2. 드라이버가 CUDA 12.x로 내려가면 **torch를 그 CUDA에 맞는 cu12x wheel로 repin**하되 `get_arch_list()`에 `sm_120`이 포함되는 버전을 골라야 한다. (주의: 이 시점 `cu128` 인덱스의 최신 torch 2.11.0은 arch_list가 비어 sm_120이 없었다 — 정확한 torch 버전을 검증해야 함. 현재 repo는 sm_120을 담은 `cu130`/2.13.0으로 고정.)
3. **검증 게이트**: 실 노드에서 `python -c "import ctypes; print(ctypes.CDLL('libcuda.so.1').cuInit(0))"` → `0`, 그리고 `torch.cuda.is_available() is True` + `get_arch_list()`에 `sm_120` 포함. 이게 통과해야 acceptance 2·3·8 검증 가능.

**핵심**: 1차(torch wheel arch_list)는 코드/config로 해결됨. 2차(cuInit)는 **드라이버-CUDA 버전 정렬(ops)**이 유일한 해결책이며, 그 전엔 이 프로젝트의 fail-fast가 정확히 이 상황을 **CPU로 조용히 덮지 않고 기동 시 크게 실패**시킨다(설계 의도대로 동작).
