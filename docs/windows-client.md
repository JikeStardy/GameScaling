# Windows 客户端安装与使用教程

适用于本 fork 的 `codex/v100-transformers` 分支，以及已经运行的 Linux GameScaling API。Windows 只做客户端，不需要下载模型、安装 CUDA、PyTorch、vLLM、WSL 或 Docker。

**当前能完成：** Windows 连接远端服务，发送示例图片或手动截取的游戏画面，接收动作文本。

**当前尚未实现：** Windows 自动截图循环、游戏窗口绑定、键鼠动作执行、任务启停和停止时释放按键。因此本文不会让《绝区零》自动操作；命令中的 Python 程序只打印模型建议，不发送任何键鼠输入。Linux 服务已通过 GPU 验收。教程中的 Python 单帧程序已从 macOS 经 SSH 隧道调用现有 V100 服务验证通过；PowerShell 和 Windows 步骤尚未在你的游戏机上实机验证。

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

## 5. 创建单帧推理程序

在仓库根目录运行以下整段 PowerShell，创建 `windows_infer_once.py`。这是本教程新建的本地示例文件，不是声称仓库已经带有的启动器。

```powershell
$env:PYTHONUTF8 = "1"
@'
import argparse
import os
from pathlib import Path
import sys

from PIL import Image
from gamescaling import GameScalingPolicy, PolicyConfig

parser = argparse.ArgumentParser(description="Send one image; print actions without executing input")
parser.add_argument("--image", default="examples/doom_battle1.jpg")
parser.add_argument("--prompt", default="prompts/example_doom_battle1.txt")
args = parser.parse_args()
base_url = os.environ.get("GAMESCALING_BASE_URL", "").strip().rstrip("/")
api_key = os.environ.get("GAMESCALING_API_KEY", "").strip()
if not base_url or not api_key or api_key == "EMPTY":
    parser.error("Set GAMESCALING_BASE_URL and GAMESCALING_API_KEY in this terminal")

cfg = PolicyConfig(
    base_url=base_url,
    model="gamescaling",
    api_key=api_key,
    image_size=(854, 480),
    history_len=2,
    reasoning=False,
    max_tokens=512,
    timeout=120.0,
)
prompt = Path(args.prompt).read_text(encoding="utf-8-sig").rstrip("\n")
policy = GameScalingPolicy(prompt, cfg)
with Image.open(args.image) as source:
    decision = policy.act(source.convert("RGB"))
if decision.error:
    print("Request failed:", decision.error, file=sys.stderr)
    raise SystemExit(1)
print("latency_ms:", round(decision.latency_ms, 1))
print("completion_tokens:", decision.completion_tokens)
print("model_text:", decision.text)
print("parsed:", decision.action.parsed)
print("mouse:", decision.action.mouse)
print("steps:", decision.action.steps)
print("No keyboard or mouse input was executed.")
if not decision.action.parsed or decision.action.truncated or decision.action.dropped:
    raise SystemExit("Action parsing failed, was truncated or dropped keys")
'@ | Set-Content -Encoding UTF8 .\windows_infer_once.py

.\.venv\Scripts\python.exe .\windows_infer_once.py
```

预期输出包含 `parsed: True`、鼠标与六组按键状态。温度保持客户端默认1.0，动作内容不必与参考文件一致。`parsed=True` 只说明可解析，不代表游戏任务成功。首次请求可能需要额外预热时间。

此程序每次启动只有一次请求，即使配置 `history_len=2`，该请求的实际历史仍为0。只有复用同一个 `policy` 连续调用 `act()` 才会逐渐积累历史。

仓库原 `examples/run_example.py` 没有API key参数，并默认使用 `EMPTY`。它的 `--offline` 可直接运行；不要把其未鉴权在线命令当成当前服务的可用调用方式，也不要添加不存在的 `--api-key` 参数。

## 6. 用自己的游戏画面做一次观察测试

1. 手动打开游戏，在窗口或无边框窗口中进入训练场。
2. 用 Windows 截图工具只截取游戏画面，保存为仓库下的 `game.png`。尽量保持16:9；本示例会直接缩放到854×480，超宽画面会被拉伸。游戏本身可以继续以4K运行。
3. 复制官方提示词模板：

```powershell
Copy-Item .\prompts\system_prompt_template.txt .\prompts\my_game.txt
notepad .\prompts\my_game.txt
```

替换全部三个占位符，保存为UTF-8：

- `{GAME_NAME}`：例如 `Zenless Zone Zero`。
- `{GAME_RULES}`：当前场景、规则与具体目标，例如训练场接近目标并普通攻击。
- `{KEY_BINDINGS}`：逐行填写你在游戏设置中实际核实的键位，使用官方训练词表，例如 `- W -> Move forward`。不要凭其他游戏的键位猜测。

保留模板的Output Format、Action标签和其余协议文本。然后执行：

```powershell
.\.venv\Scripts\python.exe .\windows_infer_once.py --image .\game.png --prompt .\prompts\my_game.txt
```

程序会把原图缩至854×480后发送，打印模型建议，仍不执行操作。当前服务限制单张输入不超过400万像素，3840×2160原图不能直接原尺寸上传；本示例的客户端缩放已处理这点。

## 7. 当前历史与调度设置

- `image_size=(854,480)`：客户端上传前缩放，游戏显示分辨率不随之改变。
- `history_len=2`：连续运行时保留过去两轮截图和动作，外加当前截图，共最多3张图。库默认5，本示例显式选2；尚未证明哪一项在《绝区零》中效果最佳。
- `reasoning=False, max_tokens=512`：与已验证的无思考服务调用一致。
- 当前服务最多接收6张图片，对应最多5轮历史；上下文总预算8192 tokens。超限会报错，不会自动缩减历史。

单帧程序没有连续运行调度。未来完整Windows控制器需要维护“只保留最新截图”的缓冲区，按上一轮推理及动作执行完成触发下一轮，并把实际执行的动作记录进历史。截图可5Hz，但不能将推理请求固定按5Hz堆积：当前V100固定负载推理约450–720ms，加动作执行约200ms后，串行闭环只能先按约1–1.5Hz规划，还未计网络、截图等开销。

## 8. 常见问题

| 现象 | 排查方法 |
|---|---|
| `py` / `git` 找不到 | 安装后重新打开PowerShell；检查安装器是否提供launcher及命令路径 |
| `.venv\Scripts\python.exe` 不存在 | 确认当前目录是仓库根目录，并且创建venv命令成功 |
| 连接超时/拒绝 | 检查LAN地址与12388；跨网先确认VPN及容器SSH12322可达，或隧道窗口仍在运行 |
| health成功，models返回401 | 检查同一个PowerShell窗口的key变量；确认沿用现有key且无多余空格 |
| 404 | Base URL包含一次 `/v1`；模型列表路径是 `/v1/models` |
| 413 `image_too_large` | 确认使用本示例的缩放设置，且没有直接上传4K原始图 |
| 422 | 请求字段或消息结构不符合当前服务；优先使用本fork的客户端 |
| 503 `model_busy` | 已有一次推理执行中，等它结束；不要启动并行循环反复提交 |
| `context_length_exceeded` | 减少输入图片尺寸/历史或提示词长度；当前服务不自动裁剪 |
| 模型返回动作但游戏不动 | 本教程只做推理观察；Windows输入执行器尚未实现 |

使用结束可在同一个PowerShell窗口清除临时key：

```powershell
Remove-Item Env:\GAMESCALING_API_KEY
Remove-Variable gsHeaders -ErrorAction SilentlyContinue
```

## 9. 到哪一步才算完成

本教程的验收是：离线解析成功 → health可达 → 鉴权models成功 → 示例图片动作可解析 → 自己的游戏截图能返回动作。

完成后，Windows侧的网络、Python和推理调用已跑通。让游戏自主行动仍需实现并验证：窗口/DPI正确截图、任务目标入口、按真实窗口尺寸换算鼠标、六个33ms按键状态执行、焦点检查、停止与释放按键、过期响应丢弃及执行后历史提交。这些功能不在当前教程中冒充已经存在。

官方来源与Windows命令依据见 [windows-client-sources.md](windows-client-sources.md)。服务安装与API限制见 [v100-transformers.md](v100-transformers.md)，GPU速度实测见 [v100-performance.md](v100-performance.md)。
