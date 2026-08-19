# 策略服务部署

本说明适用于智元 G1 OmniPicker 的在线推理。SouthGrid 负责采集、场景交互和策略客户端；策略训练与策略服务运行在独立的 [OpenPI](https://github.com/Physical-Intelligence/openpi) 环境中。

## 硬件要求

OpenPI 官方给出的单卡起始显存要求如下。实际需求会随所用模型、训练配置和并发量变化，部署前应完成目标任务验证。

| 场景 | 官方起始显存要求 | 参考硬件 |
| --- | --- | --- |
| 推理 | 大于 8 GB | RTX 4090（24 GB） |
| LoRA 微调 | 大于 22.5 GB | RTX 4090（24 GB） |
| 全量微调 | 大于 70 GB | A100 80 GB / H100 |

以上要求和 Linux 支持范围以 [OpenPI 官方 README](https://github.com/Physical-Intelligence/openpi) 为准。

## 安装策略服务环境

策略服务与 SouthGrid 运行环境彼此独立。请在 GPU 服务器上按 OpenPI 官方安装说明创建环境：

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv sync
```

SouthGrid 客户端通过标准 WebSocket 接口连接策略服务。无需将 SouthGrid 的 Python 环境或第三方目录复制到 OpenPI 环境中。

## 启动策略服务

请使用与交付任务匹配的 OpenPI 策略配置和 checkpoint。在 OpenPI 环境中启动服务：

```bash
uv run scripts/serve_policy.py \
    --port 8010 \
    policy:checkpoint \
    --policy.config=<策略配置名> \
    --policy.dir=<checkpoint目录>
```

服务启动后，在 SouthGrid 的推理命令中填写策略服务器地址和端口。按钮与工具任务的客户端操作见 [智元 G1 OmniPicker 在线推理](g1_omnipicker_inference.md)。

## 远程部署

策略服务可运行在独立 GPU 服务器上。请确保运行 OrcaLab 的主机可访问策略服务器的 `8010` 端口；无法直接连通时，可由部署人员配置受控网络或 SSH 隧道。

## 支持范围

任务专用的数据变换、训练配置、checkpoint 和性能调优属于交付集成内容。请勿修改 OpenPI 源码或替换策略服务依赖；如需新增任务或训练自定义策略，请联系交付支持团队。
