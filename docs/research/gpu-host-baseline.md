# happy-nursing-home GPU host baseline

- Measured: 2026-07-30
- Host: `happy-nursing-home`
- Scope: narrow, read-only inventory probes only; no install, reset, restart, remediation, image pull, or third-party host access was performed.
- Tailnet profile during measurement: `seniorsailab@gmail.com`

## Tailnet and host identity

### Active tailnet profile

Value: `seniorsailab@gmail.com` was active before the GPU probes.

Command:

```console
$ tailscale switch --list
ID    Tailnet                 Account
e048  kren.kr                 gobeumsu@gmail.com
0f67  gobeumsu@gmail.com      gobeumsu@gmail.com
1051  seniorsailab@gmail.com  seniorsailab@gmail.com*
```

Exit status: `0`.

### Hostname

Value: `happy-nursing-home`.

Command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'hostname'
happy-nursing-home
```

Exit status: `0`.

## Required GPU fields

### GPU identity, driver, compute capability, and memory

Values:

- GPU name: `NVIDIA GeForce RTX 5070 Ti`
- Driver version: `595.84`
- Compute capability: `12.0`
- Total memory: `16303 MiB`

Command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv'
name, driver_version, compute_cap, memory.total [MiB]
NVIDIA GeForce RTX 5070 Ti, 595.84, 12.0, 16303 MiB
```

Exit status: `0`.

### GPU power-limit fields

Values:

- GPU name: `NVIDIA GeForce RTX 5070 Ti`
- Driver version: `595.84`
- Compute capability: `12.0`
- Total memory: `16303 MiB`
- Power limit: `[N/A]`
- Default power limit: `[N/A]`

The host returned both power-limit fields as `[N/A]`; no values were inferred or substituted.

Command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,power.limit,power.default_limit --format=csv'
name, driver_version, compute_cap, memory.total [MiB], power.limit [W], power.default_limit [W]
NVIDIA GeForce RTX 5070 Ti, 595.84, 12.0, 16303 MiB, [N/A], [N/A]
```

Exit status: `0`.

### Kernel

Value: `7.0.0-28-generic`.

Command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'uname -r'
7.0.0-28-generic
```

Exit status: `0`.

### NVIDIA kernel-module license

Value: `Dual MIT/GPL`, consistent with the open NVIDIA kernel module required for this Blackwell GPU class.

Command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'modinfo nvidia | grep -i license'
license:        Dual MIT/GPL
```

Exit status: `0`.

### GSP firmware state

Values:

- `EnableGpuFirmware: 18`
- `EnableGpuFirmwareLogs: 2`

Command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'cat /proc/driver/nvidia/params | grep -i firmware'
EnableGpuFirmware: 18
EnableGpuFirmwareLogs: 2
```

Exit status: `0`.

### CUDA runtime/toolkit

Value: **FAILED / NOT CAPTURED**. `nvcc` is absent, and the read-only NVIDIA container CLI fallback failed because NVML reported that the GPU requires reset. No CUDA runtime version is inferred from the driver or toolkit CLI version.

Primary command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'nvcc --version'
bash: line 1: nvcc: command not found
```

Exit status: `127`. The message above was stderr; stdout was empty.

Fallback command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'nvidia-container-cli info'
nvidia-container-cli: detection error: nvml error: gpu requires reset
```

Exit status: `1`. The message above was stderr; stdout was empty. No reset was attempted.

### NVIDIA Container Toolkit

Value: NVIDIA Container Toolkit CLI `1.19.1`, commit `09ceee5dde66ba9ce25c7cc69b1ebd5e6e3266fa`.

Command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'nvidia-ctk --version'
NVIDIA Container Toolkit CLI version 1.19.1
commit: 09ceee5dde66ba9ce25c7cc69b1ebd5e6e3266fa
```

Exit status: `0`.

### Docker GPU reachability

Value: **FAILED / UNVERIFIED**. The pinned image tag used for the attempt was `nvidia/cuda:12.8.1-base-ubuntu24.04`. The remote Docker daemon API was unavailable, so the cache and any digest could not be inspected and the container could not start. `--pull=never` prohibited a state-changing image pull. This result does not establish a GPU pass or GPU failure.

Cache command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'docker image ls --digests nvidia/cuda'
failed to connect to the docker API at unix:///var/run/docker.sock; check if the path is correct and if the daemon is running: dial unix:///var/run/docker.sock: connect: no such file or directory
```

Exit status: `1`. The displayed text was stderr; stdout was empty.

Pinned-image inspection command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'docker image inspect nvidia/cuda:12.8.1-base-ubuntu24.04'
[]
failed to connect to the docker API at unix:///var/run/docker.sock; check if the path is correct and if the daemon is running: dial unix:///var/run/docker.sock: connect: no such file or directory
```

Exit status: `1`; the Docker error was stderr.

Required reachability command:

```console
$ ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no happy-nursing-home 'docker run --rm --pull=never --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L'
failed to connect to the docker API at unix:///var/run/docker.sock; check if the path is correct and if the daemon is running: dial unix:///var/run/docker.sock: connect: no such file or directory
```

Exit status: `1`. The displayed text was stderr; stdout was empty.

## Driver-assumption comparison

The two independent narrow GPU queries both measured driver `595.84`. This **MATCHES** the draft's assumed `595.84` figure exactly; there is no mismatch to adopt silently. Had the measured value differed, this report would flag the mismatch explicitly and retain both the measured and assumed values.

For context only, `595.84` is above the documented CUDA 13.2 GA driver floor of `595.45.04`. That comparison does not establish an installed CUDA runtime: the runtime probe failed as recorded above. CUDA initialization and an `sm_120` kernel execution were not tested by this inventory task.
