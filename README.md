# SouthGrid 南方电网大赛

SouthGrid 为南方电网竞赛场景提供人形机器人数据采集、数据回放与在线推理工具。

| 平台 | 交付功能 | 使用说明 |
| --- | --- | --- |
| 智元 G1 OmniPicker | 四色按钮与工具整理的数据采集、回放和在线推理 | [数据采集](docs/g1_omnipicker_collection.md) · [在线推理](docs/g1_omnipicker_inference.md) |
| 宇树 G1 | 工具抓取数据采集 | [数据采集](docs/unitree_g1_collection.md) |

如需基于采集数据训练策略或部署智元 G1 OmniPicker 在线推理服务，请阅读 [策略服务部署](docs/openpi_deployment.md)。

## 开始前准备

### 系统与硬件

- Ubuntu 22.04 或 24.04、Conda 与 NVIDIA 驱动（570及以上）。
- 采集视频需要支持 AV1 NVENC 的 NVIDIA  40系及以上 GPU。

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

安装完成后，可使用以下命令检查运行环境：

```bash
python scripts/verify_environment.py
```

## 首次启动与资产订阅

激活运行环境并启动 OrcaLab：

```bash
conda activate orcalab_lerobot
orcalab
```

> [!IMPORTANT]
> 首次运行任务前，必须在 OrcaLab 资产库中订阅以下资产：
>
> - `SouthGrid_Competition_2026`
> - `g1_omnipicker`
> - `g1_pick`

资产订阅流程：

1. 启动 OrcaLab。OrcaLab 会自动导航到资产库；如果没有自动打开，请点击“打开资产库”。
2. 在资产库网站中依次搜索上述资产名称。
3. 打开资产详情并点击“订阅”。
4. 完成订阅后关闭并重新启动 OrcaLab，等待资产同步完成。

## Pico 遥操作准备

只有使用 Pico 进行遥操作时才需要完成本节。请先按照 Openverse Orca 官方的 [VR 遥操作与数据采集操作指南](https://github.com/openverse-orca/OrcaDocs/blob/main/%E6%93%8D%E4%BD%9C%E6%8C%87%E5%8D%97/%E6%95%B0%E6%8D%AE%E9%87%87%E9%9B%86%E4%B8%8E%E5%90%88%E6%88%90/VR%E9%81%A5%E6%93%8D%E4%BD%9C%E4%B8%8E%E6%95%B0%E6%8D%AE%E9%87%87%E9%9B%86%E6%93%8D%E4%BD%9C%E6%8C%87%E5%8D%97.md) 完成 Pico 应用安装、开发者模式和设备连接。

Pico 遥操作还需要 Android Platform Tools（`adb`）。在 Ubuntu 上可执行：

```bash
sudo apt install adb
```

## 使用流程

完成环境安装和资产订阅后，按以下顺序运行任务：

1. 根据目标平台选择[智元 G1 OmniPicker 数据采集](docs/g1_omnipicker_collection.md)、[智元 G1 OmniPicker 在线推理](docs/g1_omnipicker_inference.md)或[宇树 G1 数据采集](docs/unitree_g1_collection.md)。
2. 在 OrcaLab 中加载平台文档指定的任务布局。
3. 按平台文档检查相机配置；使用 Pico 遥操作时，同时完成设备连接和端口映射。
4. 在 OrcaLab 中启动仿真。
5. 在终端中运行所选任务的采集、回放或推理命令。

所有相机连接参数均以随任务布局提供的 JSON 配置为准。加载布局后请保持其中的相机配置不变。

不同工作流之间的依赖关系如下：

- 遥操作采集可以直接创建演示数据。
- 脚本化采集需要先准备对应任务的路点或候选位姿。
- 数据回放需要已有采集数据，并使用与采集时相同的任务布局。
- 在线推理需要先部署策略服务，再使用与策略任务对应的布局启动推理客户端。

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


