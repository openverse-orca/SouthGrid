# G1 OSC 关节剥离采集

本目录是可复现包：编译期删掉下肢 / 腰 / 左臂 / 左爪关节，只保留右臂和右爪参与 `mj_step`。左臂侧平举在删关节前烘进 body，渲染和相机挂点不变。

| 文件 | 作用 |
| --- | --- |
| `g1_pick_osc_collection_tele_lerobot_strip.py` | 数采脚本（相对原版增加 `--joint_strip`） |
| `mj_joint_strip.py` | XML 注入：删关节、烘姿态、qpos 补全、安全检查 |
| `uni_osc.json` | 当前实测场景布局 |
| `example.yaml` | 任务配置（agent 前缀 `g1_pick_southgrid_usda_1_`） |
| `run_strip.sh` | 推荐启动命令 |

机器人名称必须是 **`g1_pick_southgrid_usda_1`**。完整说明见仓库 `docs/unitree_g1_collection.md`。

---

## 1. 环境

按仓库根目录 `README.md` 安装。OrcaLab / OrcaGym 为 **26.7.1（7.1）**。

```bash
git clone https://github.com/openverse-orca/SouthGrid.git
cd SouthGrid
conda activate orcalab_lerobot
```

---

## 2. 加载布局

1. 启动 OrcaLab 7.1。
2. 打开：

```
src/examples/dataCollection/unitree_g1/joint_strip/uni_osc.json
```

3. 确认机器人名为 `g1_pick_southgrid_usda_1`。
4. 相机端口：头 **7090**，右腕 **7080**。
5. 运行仿真，`localhost:50051`。

---

## 3. Pico

```bash
adb reverse tcp:8001 tcp:8001
adb reverse --list
```

---

## 4. 采集

```bash
cd src/examples/dataCollection/unitree_g1/joint_strip
bash run_strip.sh /path/to/output_dataset
```

仅遥操：命令末尾加 `--teleop_only`，并去掉 `--lerobot_out`。
