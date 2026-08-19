# SouthGrid

SouthGrid 为南方电网竞赛场景提供人形机器人数据采集、数据回放与在线推理工具。

| 平台 | 交付功能 | 使用说明 |
| --- | --- | --- |
| 智元 G1 OmniPicker | 四色按钮与工具整理的数据采集、回放和在线推理 | [数据采集](docs/g1_omnipicker_collection.md) · [在线推理](docs/g1_omnipicker_inference.md) |
| 宇树 G1 | 工具抓取数据采集 | [数据采集](docs/unitree_g1_collection.md) |

如需基于采集数据训练策略或部署智元 G1 OmniPicker 在线推理服务，请阅读 [策略服务部署](docs/openpi_deployment.md)。

## 开始前准备

### 系统与硬件

- Ubuntu 22.04、Conda 与 NVIDIA 驱动。
- 采集视频需要支持 AV1 NVENC 的 NVIDIA Ada-Lovelace 架构或更新架构 GPU。
- Pico 遥操作需要 Android Platform Tools（`adb`）。在 Debian/Ubuntu 上可通过 `sudo apt install adb` 安装。
- 在 OrcaLab 资产库中订阅下列场景资产：
  - `SouthGrid_Competition_2026`
  - `G1_omnipicker`
  - `G1_Picker_SouthGrid`

### 获取代码

```bash
git clone https://github.com/openverse-orca/SouthGrid.git
cd SouthGrid
```

## 安装运行环境

请在新建的 Conda 环境中执行安装：

```bash
conda env create -f environment-unitree.yml
conda activate orcalab_lerobot
bash scripts/install_runtime.sh
```

安装完成后启动 OrcaLab：

```bash
conda activate orcalab_lerobot
orcalab
```

可使用以下命令检查运行环境：

```bash
python scripts/verify_environment.py
```

## 使用流程

1. 在 OrcaLab 中加载相应任务的布局文件并启动仿真。
2. 按平台文档完成相机与遥操作准备。
3. 运行对应的采集、回放或推理命令。

所有相机连接参数均以随任务布局提供的 JSON 配置为准。加载布局后请保持其中的相机配置不变。

## 数据输出

采集结果采用 LeRobot v2.1 数据集格式，包含帧数据、任务信息和视频文件。相机、帧率及数据字段由所用采集任务自动写入数据集元信息；训练或集成时请读取数据集内的 `meta/info.json`。

## 目录概览

```text
SouthGrid/
├── docs/                    # 平台使用与部署说明
├── scripts/                 # 环境安装与检查脚本
├── src/examples/
│   ├── dataCollection/      # 数据采集和回放入口
│   └── inference/           # 在线推理入口
└── third_party/             # 随交付环境安装的运行时组件
```

## 支持范围

本仓库仅包含交付场景所需的运行入口和配置。未在上述使用说明中列出的内容不属于客户使用流程；如需扩展任务、训练配置或部署支持，请联系交付支持团队。
