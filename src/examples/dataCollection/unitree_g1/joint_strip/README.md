# G1 OSC 关节剥离采集

本目录是可复现包：编译期删掉下肢 / 腰 / 左臂 / 左爪关节，只保留右臂和右爪参与 `mj_step`。左臂自然下垂在删关节前烘进 body，渲染和相机挂点不变。

| 文件 | 作用 |
| --- | --- |
| `g1_pick_osc_collection_tele_lerobot_strip.py` | 遥操数采脚本（相对原版增加 `--joint_strip`） |
| `record_waypoints.py` | Pico 录右臂路点，写出 YAML |
| `g1_pick_osc_collection_scripted_lerobot_strip.py` | 回放路点 YAML 的自动采集脚本（不接 Pico） |
| `g1_pick_osc_replay_lerobot_strip.py` | 回放脚本化采集写出的 LeRobot parquet |
| `mj_joint_strip.py` | XML 注入：删关节、烘姿态、qpos 补全、安全检查 |
| `uni_osc.json` | 当前实测场景布局 |
| `example.yaml` | 任务配置（agent 前缀 `g1_pick_southgrid_usda_1_`） |
| `run_strip.sh` | 遥操采集启动命令 |

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

---

## 5. 录路点

参数与数采相同。双手 Grip 记一点，右 Squeeze 丢弃上一点并重置场景，Ctrl+C 写出 YAML。

```bash
cd src/examples/dataCollection/unitree_g1/joint_strip
adb reverse tcp:8001 tcp:8001
conda activate orcalab_lerobot
OMP_NUM_THREADS=1 python record_waypoints.py \
    --task_config example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --lerobot_out $HOME/southgrid_datasets/g1_osc \
    --repo_id local/g1_pick_osc_strip \
    --task "按红色按钮" \
    --fps 20 --clock wall \
    --cameras head,wrist_r --camera_source websocket \
    --dls_lambda 0.23 \
    --joint_strip on --strip_col off \
    --time_step 0.001 --frame_skip 5 \
    --output my_waypoint_tool1.yaml
```

---

## 6. 路点回放自动采集

`g1_pick_osc_collection_scripted_lerobot_strip.py` 复用遥操脚本的控制链路（右臂 OSC + 2F85 反向夹爪 + 关节剥离），只是不接 Pico：把录好的 YAML 离线插值成整条轨迹，逐控制步写进 OSC 控制器，轨迹首尾自动置任务状态 RUNNING / END，每集跑完强制写盘。

空跑（不开相机、不写盘）：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/unitree_g1/joint_strip
OMP_NUM_THREADS=1 python g1_pick_osc_collection_scripted_lerobot_strip.py \
    --task_config example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --waypoint_files my_waypoint_tool1.yaml \
    --joint_strip on --strip_col off \
    --dry_run --track_log_every 100 \
    --dls_lambda 0.23 --dls_sigma_th 0.12 --null_kp 10 \
    --kp 0 --action_repeat 1 \
    --time_step 0.001 --frame_skip 5 \
    --track_ki 0 --track_clamp 0.08
```

采集：

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/unitree_g1/joint_strip
OMP_NUM_THREADS=1 python g1_pick_osc_collection_scripted_lerobot_strip.py \
    --task_config example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --waypoint_files my_waypoint_tool1.yaml \
    --task "按红色按钮" \
    --lerobot_out $HOME/southgrid_datasets/g1_osc_scripted \
    --repo_id local/g1_pick_osc_scripted \
    --num_episodes 10 \
    --fps 20 --clock sim \
    --cameras head,wrist_r --camera_source websocket \
    --joint_strip on --strip_col off \
    --time_step 0.001 --frame_skip 5 \
    --dls_lambda 0.23 --dls_sigma_th 0.12 --null_kp 10 \
    --kp 0 --action_repeat 1 \
    --track_ki 0 --track_clamp 0.08
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--waypoint_files` | 路点 YAML，逗号分隔可传多个，按顺序在同一集内依次执行 |
| `--num_episodes` | 采集集数（默认 1） |
| `--speed` | 轨迹提速倍率，各段步数除以它（默认 1.0） |
| `--steps` | 覆盖每段控制步数（默认 0 = 用 YAML 里的 `steps`） |
| `--hold_steps` | 末尾保持段步数，给最后一次开/合爪留沉降时间（默认 100） |
| `--kp` | OSC 刚度（默认 0 = 沿用 `osc_pose` 配置；>0 时覆盖并设 kd=2√kp） |
| `--dls_lambda` | 阻尼最小二乘 λ_max（默认 0.23，与遥操一致；0 = 原始 pinv） |
| `--dls_sigma_th` | 变λ触发阈值（默认 0.12；0 = 固定 λ） |
| `--null_kp` | 零空间复原增益（默认 10） |
| `--track_ki` | 末端外环积分增益（默认 0.02；0 = 关）。只在每个采样第一拍积分 |
| `--track_clamp` | 积分偏置限幅，米（默认 0.08） |
| `--action_repeat` | 每个轨迹采样重复控制步数（默认 1） |
| `--dry_run` | 只跑轨迹不保存数据 |

一段 300 步 ≈ 1.5 s（`env.dt = time_step × frame_skip = 5 ms`），6 点路点加末尾保持约 9.5 s / 190 帧（`--fps 20 --clock sim`）。

本脚本不做随机化，多集轨迹几乎相同；要多样性就多录几份路点，用 `--waypoint_files a.yaml,b.yaml` 串起来。

---

## 7. 回放已采集的 LeRobot 数据

不开相机、不写盘。控制链路与脚本化采集相同，数据源换成 parquet 的 18 维 `action`（只驱动右臂 / 右爪）。

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/unitree_g1/joint_strip
OMP_NUM_THREADS=1 python g1_pick_osc_replay_lerobot_strip.py \
    --dataset_dir $HOME/southgrid_datasets/g1_osc_scripted \
    --task_config example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --joint_strip on --strip_col off \
    --time_step 0.001 --frame_skip 5 \
    --steps_per_frame 10 \
    --dls_lambda 0.23 --dls_sigma_th 0.12 --null_kp 10 \
    --kp 0 --track_ki 0 --track_clamp 0.08
```

只播第 1 集加 `--episode 1`，循环加 `--loop`。
