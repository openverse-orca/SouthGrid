# 随附运行时组件

本目录包含 SouthGrid 运行所需的固定版本组件，由仓库根目录的安装脚本统一安装。

| 目录 | 用途 |
| --- | --- |
| `lerobot/` | 数据集读写与视频处理 |
| `televuer/` | OrcaLab 7.3 环境使用的 XR 运行时兼容包 |
| `openpi-client/` | 与 OpenPI 策略服务通信的客户端 |

当前宇树 G1 流程使用 PicoJoystick；TeleVuer 作为兼容组件保留，不是默认遥操作入口。

策略服务、模型和 checkpoint 不包含在本仓库中。
