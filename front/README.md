빌드: `pnpm install && pnpm build`

## ML API 연결

대시보드는 기본적으로 같은 origin의 `/api/v1`을 호출합니다. Docker 이미지나
reverse proxy 뒤에서는 이 기본값을 그대로 둡니다.

Vite dev server에서 `/api/v1` 프록시 대상만 바꿀 때는 서버 환경 변수
`ML_API_PROXY_TARGET`을 사용합니다.

```bash
ML_API_PROXY_TARGET=http://127.0.0.1:8000 pnpm dev
```

브라우저가 직접 다른 ML API origin을 호출해야 하는 edge hot-reload QA에서는
클라이언트 환경 변수 `VITE_ML_API_BASE_URL`을 명시합니다. 이 값은 fetch 요청,
MJPEG stream URL, clip video URL에 모두 동일하게 적용됩니다.

```bash
VITE_ML_API_BASE_URL=http://nursinghome:8000/api/v1 pnpm dev -- --host 0.0.0.0
```

## `/api/v1/system` 확장 제안

시스템 화면은 클립 스토어 사용량을 게이지로 표시할 수 있도록 다음 선택 필드를 읽습니다.

```json
{
  "storage": {
    "clips_used_bytes": 2147483648,
    "clips_limit_bytes": 10737418240
  },
  "update_history": [
    { "id": "deploy-20260706", "version": "2026.07.06", "created_at": "2026-07-06T00:00:00.000Z", "status": "applied" }
  ],
  "rollback_history": []
}
```

필드가 없으면 대시보드는 사용량을 추정하지 않고 안내문을 표시합니다.
