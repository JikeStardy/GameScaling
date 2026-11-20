# Windows 客户端安装与使用教程

适用于本 fork 的 `codex/v100-transformers` 分支，以及已经运行的 Linux GameScaling API。Windows 只做客户端，不需要下载模型、安装 CUDA、PyTorch、vLLM、WSL 或 Docker。

Windows 客户端可以发送一张图片，也可以持续截取指定的前台游戏窗口，把最近的画面和动作发给服务，再执行模型返回的键鼠动作。先用观察模式验证窗口和模型输出，再在训练场开启执行。

## 1. 安装 Windows 依赖

使用普通 PowerShell，无需以管理员身份运行客户端。

- 安装 [Python 官方 Windows 发行版](https://www.python.org/downloads/windows/)，选择适合本机的64位版本。已有 Python 3.12 可以复用；首次安装选择官网当前受支持的发行版，不要为复现客户端特意安装旧补丁版本。
- 安装 [Git for Windows](https://git-scm.com/install/windows)。
- 安装后重新打开 PowerShell，检查：

```powershell
py --version
git --version
```

以下 `py -m venv` 使用默认 Python。Linux/本地客户端验证使用 Python 3.12；Windows 的具体 Python 和依赖版本需要以实际安装及本教程的测试结果为准。

## 2. 下载 fork 并创建客户端环境

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\GameScalingWork" | Out-Null
Set-Location "$env:USERPROFILE\GameScalingWork"
git clone --branch codex/v100-transformers --single-branch https://github.com/JikeStardy/GameScaling.git
Set-Location .\GameScaling
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe examples\run_example.py --offline
```

最后一步应打印参考动作并显示 `parsed=True`，它不连接服务器，也不操作游戏。

本教程始终显式调用 `.venv\Scripts\python.exe`，不用执行 `Activate.ps1`，也不用修改 PowerShell ExecutionPolicy。若已 clone，可进入仓库后执行 `git pull --ff-only`，不要重复 clone 到同一路径。

## 3. 选择一种连接方式

### A. 与服务器在同一局域网

将 `SERVER_LAN_IP` 替换为推理服务器的局域网地址。下面12388是本次部署复用的现有外部端口；容器中服务监听8888。

```powershell
$env:GAMESCALING_BASE_URL = "http://SERVER_LAN_IP:12388/v1"
Test-NetConnection SERVER_LAN_IP -Port 12388
Invoke-RestMethod "http://SERVER_LAN_IP:12388/health"
```

预期 `TcpTestSucceeded : True`，health 返回 `status=ready`。`busy=True` 仅表示已有推理请求在执行。

### B. 通过现有容器 SSH 入口转发

如果不在局域网，可使用能到达服务器的既有 VPN/Tailscale 网络和容器 SSH 入口。Windows 必须具备自己的 SSH 登录凭据；Mac 上已经能登录，不会自动让 Windows 获得相同凭据。

先检查 `ssh -V`。没有该命令时，按 [Microsoft OpenSSH 客户端安装说明](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)安装客户端。

在一个**独立 PowerShell 窗口**运行，将 `TAILSCALE_SERVER_IP` 替换为可达地址：

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18888:127.0.0.1:8888 -p 12322 root@TAILSCALE_SERVER_IP
```

这是登录 **ai-devel 容器的12322入口**，不使用宿主机管理入口。核对首次连接的主机指纹后认证；成功时不进入命令行、持续等待属于正常现象。保持该窗口打开，Ctrl+C 会关闭隧道。本命令只在 Windows 本机127.0.0.1监听18888，不增加宿主机端口映射。

回到执行 Python 的 PowerShell 窗口：

```powershell
$env:GAMESCALING_BASE_URL = "http://127.0.0.1:18888/v1"
Invoke-RestMethod "http://127.0.0.1:18888/health"
```

如果18888已被本机其他程序占用，可更换 `-L` 左侧端口，并同步修改 Base URL。不要将当前仅绑定局域网的12388直接改成外网监听来解决连接问题。

## 4. 设置 API key 并验证鉴权

继续在运行 Python 的同一个 PowerShell 窗口中操作。

API key 已存在于 ai-devel 的 `/workspace/services/gamescaling-secrets/api-key`。从你已有的受信任连接读取或安全传给 Windows，**沿用原 key，不重新生成或覆盖它**。不要把 key 写进仓库文件、截图、命令参数或教程。

通过不回显的交互输入，将 key 临时放入当前 PowerShell 及其子进程环境：

```powershell
$gsSecret = Read-Host "粘贴现有 GameScaling API key" -AsSecureString
$env:GAMESCALING_API_KEY = [System.Net.NetworkCredential]::new("", $gsSecret).Password.Trim()
Remove-Variable gsSecret
$gsHeaders = @{ Authorization = "Bearer $env:GAMESCALING_API_KEY" }
Invoke-RestMethod "$env:GAMESCALING_BASE_URL/models" -Headers $gsHeaders
```

预期模型名为 `gamescaling`，metadata 中 `parameter_devices` 为 `cuda:0`、dtype为 `torch.float16`，且CPU/disk offload均为false。没有鉴权的 `/health` 成功，不代表带鉴权的 `/v1/models` 一定成功。

窗口关闭后，需要在新窗口重新设置 Base URL 和 key。本教程不将 key 持久化到系统环境。

## 5. 用示例图片检查推理

```powershell
.\.venv\Scripts\python.exe -m gamescaling.windows_client --image .\examples\doom_battle1.jpg --prompt .\prompts\example_doom_battle1.txt
```

成功时会显示 `Observed`、请求耗时和动作。图片在上传前缩至 854×480。这个入口只打印动作。

## 6. 观察游戏窗口

手动打开《绝区零》，进入训练场。使用窗口或无边框模式，确保游戏窗口完整显示。用 PowerShell 查实际窗口标题和进程名。

```powershell
Get-Process | Where-Object MainWindowTitle | Select-Object ProcessName, MainWindowTitle
Copy-Item .\prompts\system_prompt_template.txt .\prompts\my_game.txt
notepad .\prompts\my_game.txt
```

把 `{GAME_NAME}`、`{GAME_RULES}`、`{KEY_BINDINGS}` 换成游戏名、当前场景与目标，以及从游戏设置核对的键位。保留其余动作格式文本。比如训练场移动可以先描述 `W`、`A`、`S`、`D` 的作用。把下列命令中的 `YOUR_WINDOW_TITLE` 替换为实际窗口标题或可唯一匹配的片段。

```powershell
.\.venv\Scripts\python.exe -m gamescaling.windows_client --window "YOUR_WINDOW_TITLE" --prompt .\prompts\my_game.txt
```

看到 `Switch to the game window` 后，切回游戏。客户端只截取绑定窗口的内容。窗口消失、最小化、被明显遮挡或截图全黑时，客户端会退出并给出原因。可加 `--process` 指定可执行文件名（包含 `.exe`），避免与其他窗口重名。

## 7. 开启动作执行

观察模式识别的目标窗口和动作均符合预期后，在训练场运行以下短测试。只启用已核实且适合当前目标的键位。

```powershell
.\.venv\Scripts\python.exe -m gamescaling.windows_client --window "YOUR_WINDOW_TITLE" --prompt .\prompts\my_game.txt --execute --keys "W,A,S,D,LB" --mouse-look --turns 8 --seconds 15
```

- 按 **F12** 即可停止，也可以切离绑定窗口。客户端会释放当前按住的键和鼠标键。F12 保留给停止功能。PowerShell 在前台时也可用 Ctrl+C 停止。
- `--keys` 填可执行的游戏键和鼠标键，例如 `LB` 是左键。`--mouse-look` 允许相对鼠标移动。需要滚轮时显式加 `--wheel`。超出启用范围或格式异常的动作会结束当前会话。
- 每次只处理一帧，执行约 200 ms 的动作后再截图。默认保留最近两轮已执行动作和对应画面，可用 `--history 0` 降低延迟。截图到动作启动默认最多 1500 ms，可用 `--max-age-ms` 调整。
- 游戏输入由 Windows `SendInput` 发送，目标窗口必须在前台。若游戏不接受系统合成输入，观察模式仍可用于检查推理结果。 [Microsoft SendInput 说明](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput)
- 日志中的 `sent` 表示 Windows 输入 API 接收了事件，`no-op` 表示此轮动作为空，`observed` 表示观察模式。若要确认游戏接受了按键，请观察角色是否实际移动。

游戏没有响应时，可在训练场的开阔处，用独立测试发送一次 `S`，约 200 ms 后释放。该测试不请求推理服务。先用 `scan-code`，再用 `virtual-key` 做同条件对照。

```powershell
.\.venv\Scripts\python.exe -m gamescaling.windows_client --window "YOUR_WINDOW_TITLE" --execute --probe-key S --keyboard-mode scan-code
.\.venv\Scripts\python.exe -m gamescaling.windows_client --window "YOUR_WINDOW_TITLE" --execute --probe-key S --keyboard-mode virtual-key
```

`scan-code` 是默认模式。两种模式都使用 Windows `SendInput`，支持前台检查和 F12 停止。

## 8. 常见问题

| 现象 | 排查方法 |
|---|---|
| health 成功，models 返回 401 | 确认同一个 PowerShell 窗口中使用了现有 API key |
| 连接超时 | 检查局域网映射或 SSH 隧道是否可达 |
| 绑定窗口失败 | 用 `Get-Process` 核对标题和进程名，并恢复窗口 |
| 截图全黑或被遮挡 | 使用窗口或无边框模式，关闭挡在窗口上的浮层，再前台运行 |
| 提示帧过期 | 先用 `--history 0` 测试网络和推理时延 |
| `Model requested keys outside the enabled set` | 核对实际键位和任务提示词，再决定是否允许对应输入 |
| `SendInput failed` | 先验证观察模式，检查游戏是否接收普通系统合成输入 |

用完可清除当前 PowerShell 会话内的 key。

```powershell
Remove-Item Env:\GAMESCALING_API_KEY
Remove-Variable gsHeaders -ErrorAction SilentlyContinue
```

安装与网络命令见 [官方来源核验](windows-client-sources.md)。推理服务的环境和性能数据见 [V100 部署](v100-transformers.md)和 [V100 性能报告](v100-performance.md)。
