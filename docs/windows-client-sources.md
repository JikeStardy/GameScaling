# Windows 客户端教程：官方来源核验

核验日期：2026-11-14。本文是安装与网络命令的来源记录，不是 Windows 实机验收。

## Python 与虚拟环境

Python 的 Windows launcher 支持用 `py -3.12` 选择已安装的 3.12，`py --list` 可查看安装版本。因此已有 3.12 时可以使用：

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Windows 虚拟环境的解释器位于 `Scripts` 目录。官方明确允许直接调用解释器，无须执行 `Activate.ps1`，因此本教程无需修改 PowerShell 执行策略。[Python 3.12 Windows launcher](https://docs.python.org/3.12/using/windows.html#python-launcher-for-windows)、[venv 文档](https://docs.python.org/3.12/library/venv.html#how-venvs-work)

首次安装时需留意版本：3.12 已进入仅安全修复阶段，后续官方发行版只有源码，3.12.10 是最后提供二进制安装器的 3.12 版本；不要将它称为最新版。新 Windows 安装应从官方 Windows 下载页选择仍提供安装器的受支持版本，再相应调整 `py -3.x`。客户端环境只安装 `requirements.txt`，与远端模型的 Python/CUDA 环境分开；不同 Python 版本的 Windows 运行仍应实际验证。[Python 3.12.16 发行说明](https://www.python.org/downloads/release/python-31216/#no-installers)、[Windows 官方下载页](https://www.python.org/downloads/windows/)

## Git for Windows

可从 Git 官方 Windows 安装页下载 x64 安装器。官方也列出以下 PowerShell 命令，但说明 winget 的 `Git.Git` 包并非 Git for Windows 项目维护，可能与其最新版本不同：

```powershell
winget install --id Git.Git -e --source winget
```

安装后重新打开 PowerShell，执行 `git --version` 验证。[Git 官方 Windows 安装页](https://git-scm.com/install/windows)、[Git for Windows](https://gitforwindows.org/)

## OpenSSH 与可选隧道

Windows 10/11 可通过系统可选功能提供 OpenSSH Client。先执行 `Get-Command ssh` 或 `ssh -V` 检查；缺失时只需安装 **OpenSSH Client**，无需在 Windows 上安装/启动 SSH Server 或开放入站 22 端口。[Microsoft：Windows Terminal SSH](https://learn.microsoft.com/en-us/windows/terminal/tutorials/ssh)、[Microsoft：OpenSSH 安装说明](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse#install-openssh-server--client)

局域网直连可继续使用已部署的 `http://SERVER_LAN_IP:12388/v1`。如 Windows 能访问现有容器 SSH 入口，并具备其登录凭据，可在单独 PowerShell 窗口保留以下隧道：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18888:127.0.0.1:8888 -p 12322 root@TAILSCALE_SERVER_IP
```

随后客户端 Base URL 使用 `http://127.0.0.1:18888/v1`。`-N` 不执行远端命令，`-p` 指定 SSH 端口，`-L` 将本机 18888 的连接经 SSH 转到远端视角的 `127.0.0.1:8888`。由于 SSH 的终点是 ai-devel 容器，这里的远端 localhost 指容器内部。此命令利用现有 SSH 入口，不新增 Docker 端口、不修改宿主机。[OpenSSH ssh 手册](https://man.openbsd.org/ssh)

前提是该地址从 Windows 可达、SSH 登录成功且容器 SSH 允许 TCP forwarding；`TAILSCALE_SERVER_IP` 的可达性取决于已有网络/VPN连接，不能把它当作任意公网都能直连的地址。首次连接需核对服务器指纹，不使用跳过 host-key 检查的参数。隧道窗口关闭后，本机 18888 转发随之结束；模型服务继续独立运行。局域网直连与隧道两种方式都仍需原 API Bearer key。
