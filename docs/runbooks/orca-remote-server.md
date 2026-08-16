# 원격 Orca 서버 설치와 운영 (`happy-nursing-home`)

이 런북은 엣지 호스트 `happy-nursing-home`에 Orca IDE 원격 서버를 올리고,
클라이언트 Mac에서 붙는 절차를 손으로 재현하기 위한 것이다. 서버가
프로젝트, 워크트리, 터미널, 에이전트 프로세스를 소유한다. 클라이언트 Mac은
UI만 그린다. 노트북을 닫아도 에이전트는 호스트에서 계속 돈다.

호스트는 Ubuntu 24.04.4 LTS, x86_64다. 공유 리눅스 계정은 `seniorsailab`이다.
Tailscale IPv4는 `100.96.162.127`이다.

페어링 코드, 토큰, API 키, sudo 비밀번호는 이 문서에 적지 않는다. 저널이나
티켓에도 붙이지 말 것.

> [!WARNING]
> **이름 함정: `/usr/bin/orca`는 Orca IDE가 아니다.**
> 그 바이너리는 GNOME 스크린 리더(실측 `46.1`)다. Orca IDE의 CLI는
> **`orca-ide`**다. `command -v orca`로 설치를 확인하지 말 것.
> `/usr/bin/orca`를 심링크하거나 가리지 말 것. 스크린 리더를 깨뜨린다.

## 1. 설치

패키지는 stably-ai/orca GitHub releases의
`orca-ide_1.4.183_amd64.deb`다. 설치 **전에** SHA-256을 확인한다.

```bash
sha256sum orca-ide_1.4.183_amd64.deb
```

기대 해시:

```text
35241a12007dd3a8999ab2fbeb5e9d15cc82e59bacf3fdaf1141db4c0eab9edd
```

해시가 다르면 중단한다. 맞으면:

```bash
sudo dpkg -i orca-ide_1.4.183_amd64.deb
```

의존성이 비면 이어서 고친다. 끌어오는 패키지는 `xdotool`, `xclip`, `xvfb`,
`libxdo3`다.

```bash
sudo apt-get -f install
```

패키지 `postinst`가 `/usr/bin/orca-ide`를
`/opt/Orca/resources/bin/orca-ide`로 심링크하고,
`/opt/Orca/chrome-sandbox`에 `chmod 4755`를 건다. AppImage 경로는 쓰지
않는다. 이 호스트에는 `libfuse2`가 없고, AppImage는 위 두 단계를 건너뛴다.

`orca-ide serve`는 `ELECTRON_RUN_AS_NODE=1`인 순수 Node 프로세스다.
`DISPLAY`가 필요 없고, xvfb 우회도 필요 없다.

### 버전 확인 함정

`orca-ide --version`은 버전을 찍지 않는다. CLI에 버전 핸들러가 없어서
알 수 없는 argv는 help를 찍고 종료 코드 0으로 나온다. 패키지 버전은
다음으로 본다.

```bash
dpkg-query -W -f='${Version}' orca-ide
```

실측값: `1.4.183`.

스크린 리더가 살아 있는지도 같이 본다.

```bash
orca --version
```

실측값: `46.1`. 이 출력이 사라졌거나 `orca-ide`로 바뀌면 이름 함정에
걸린 것이다. 설치를 되돌리고 `/usr/bin/orca`를 복구한다.

## 2. systemd 사용자 유닛

유닛 파일은 `~/.config/systemd/user/orca-ide.service`다. **user 유닛**이다.
system 스코프로 올리면 재시작과 로그 조회마다 sudo가 필요하다.

실측된 키:

```ini
[Service]
ExecStart=/usr/bin/orca-ide serve --port 6768 --pairing-address 100.96.162.127
Restart=on-failure
RestartSec=5
MemoryHigh=3G
MemoryMax=4G
Environment=PATH=%h/.local/bin:%h/.bun/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin

[Install]
WantedBy=default.target
```

`Environment=PATH=...`는 빠져서는 안 된다. 이 호스트의 non-interactive PATH에
`~/.local/bin`이 없어서, 없으면 서버가 `claude`, `codex`, `omo`를 찾지 못한다.

linger가 켜져 있어야 로그인 없이 부팅 시 유닛이 뜬다.

```bash
loginctl show-user seniorsailab
```

실측: `Linger=yes`.

관리 명령은 전부 `--user`다.

```bash
systemctl --user status orca-ide.service
systemctl --user restart orca-ide.service
systemctl --user disable orca-ide.service
journalctl --user -n 50
```

재시작은 정확히 다음이다.

```bash
systemctl --user restart orca-ide.service
```

## 3. 페어링

페어링 URL은 저널에 찍힌다. 형태는 `orca://pair?code=<REDACTED>`다.
재시작하면 새 URL이 나온다.

> [!WARNING]
> 이 줄은 `app-orca-*.scope`로 귀속된다. `journalctl --user -u orca-ide`는
> 놓칠 수 있다. 아래 둘 중 하나를 쓴다.

```bash
journalctl --user --no-pager -n 200
journalctl --user --no-pager -n 200 SYSLOG_IDENTIFIER=orca-ide
```

문서, 커밋, 로그, 티켓에 페어링 코드를 붙이지 말 것. 보여줄 때는
항상 이렇게 가린다.

```text
orca://pair?code=<REDACTED>
```

클라이언트 Mac에서:

```bash
orca environment add --name happy-nursing-home --pairing-code "<url>"
orca environment list
```

실측: 클라이언트 `1.4.168`과 서버 `1.4.183`은 프로토콜이 맞았다. 버전을
맞출 필요는 없었다. 벤더 문서는 목록에 incompatible protocol version이
뜨면 양쪽을 같이 올리라고 한다.

회수:

- 서버: Settings → Remote Orca Servers → Shared Server Access에서 해당
  클라이언트의 grant를 revoke. 페어링된 클라이언트마다 토큰이 따로 있다.
- 클라이언트: `orca environment rm`

## 4. 왕복 확인

원격 환경에 터미널을 열고 아래를 읽는다.

```bash
hostname
uname -sm
```

합격:

```text
happy-nursing-home
Linux x86_64
```

Mac 호스트 이름이나 `Darwin`이 나오면 명령이 로컬에서 실행된 것이다.
페어링이 서버로 일을 넘기지 않은 것이므로 3절로 돌아간다.

## 5. 네트워크 실측

`orca-ide serve`는 `0.0.0.0:6768`에 바인드한다. 호스트 주소는 세 갈래다.

| 경로 | 주소 | 인터페이스 |
| --- | --- | --- |
| 공인 | `222.120.14.49` | `enx701988522baf` (이 NIC에는 NAT 없음) |
| LAN | `10.10.117.113` | 사설 |
| tailnet | `100.96.162.127` | Tailscale |

인터넷 경로에서 실측하면 6768은 공인으로 열리지 않는다. 열려 있는 것은
SSH(22)뿐이다. 런타임은 사실상 tailnet 전용이다.

이 차단은 호스트 방화벽이 아니다. `ufw`는 설치돼 있으나
`/etc/ufw/ufw.conf`의 `ENABLED=no`다. firewalld와 nftables도 비활성이다.
필터는 **업스트림**에 있다. 업스트림 규칙이 바뀌면 6768이 그대로 노출된다.
네트워크 변경 뒤에는 바깥에서 다시 본다.

```bash
nc -z -w 5 222.120.14.49 6768
```

실패(닫힘)가 기대값이다. 성공하면 공인 노출이므로 즉시 업스트림을 확인하고
서비스를 내린다.

Tailscale 경로는 DIRECT다. DERP 릴레이가 아니다. RTT는 약 16 ms에서 82 ms다.

## 6. 이 작업 이후의 호스트 배치

활성 정본 클론:

```text
~/beomsukoh/SeeON/SeeON-Front
~/beomsukoh/SeeON/SeeON-Backend
~/beomsukoh/SeeON/SeeON-edge
~/beomsukoh/eldercare-dataset-ops
```

보존 전용. 지우지 말 것:

```text
~/beomsukoh/archive/
```

여기에는 폐기 저장소, 예전 에이전트 상태 `_gjc-husks/`
(`lanes.service` / `lanes-up` 포함), 그리고
`_preserved/edge-deploy-config-20260816/`가 있다. 후자는 손으로 고친
라이브 배포 설정의 **유일한** 복사본이다. 58개 파일이고 해시 매니페스트가
있다.

`archive/eldercare-fall-ml-v2`는 일부러 DETACHED HEAD로 남겨 두었다.
커밋되지 않은 파일 6개가 있다. 그 dirty 상태 자체가 보존 산출물이다.
정리하지 말 것.

예전 tmux `lanes.service` 자동 시작은 폐기했다. 가리키던 디렉터리가
옮겨졌다.

공유 계정에 들어오는 사람은 호스트의 `~/AGENTS.md`를 먼저 읽는다.
위 경계가 거기에 적혀 있다.

## 7. 건드리면 안 되는 경계

> [!WARNING]
> **이 호스트는 공유 기기이자 현장 장비다. 아래 두 영역은 읽기만 한다.**

`~/happy_admin/**`는 Junho Park
(`github.com/parkjunho12/happy-nursing-home-jsx`)의 작업 트리이고,
커밋되지 않은 작업이 들어 있다. 다른 사람은 읽기만 한다.

`broadcast-agent.service`는 요양원 전관 방송 시스템이다. 전용 사용자
`broadcast`로 돌고 `Restart=always`다. 관련 경로:

```text
/opt/broadcast-agent
/var/lib/broadcast-agent
```

이 유닛을 멈추거나, 재시작하거나, 수정하지 말 것. 장애는 거주자 대면
사고로 취급한다.

## 8. 철거 / 롤백

서버만 내릴 때:

```bash
systemctl --user disable --now orca-ide.service
```

페어링된 클라이언트의 grant를 서버 Settings에서 revoke한다. 클라이언트는
`orca environment rm`으로 환경을 지운다.

패키지까지 제거할 때:

```bash
sudo apt-get remove orca-ide
```

이 명령은 GNOME 스크린 리더(`orca` 패키지, `/usr/bin/orca`)를 건드리면
안 된다. 제거 후 `orca --version`이 여전히 `46.1`인지 확인한다.
`/usr/bin/orca`가 사라졌거나 `orca-ide`를 가리키면 복구가 먼저다.
