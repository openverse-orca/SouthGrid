# 宇树 G1 · 数据采集（OSC）

本文对应本仓库当前代码：宇树 G1 已改为 **OSC 力矩控制 + OmniPicker 2F85 夹爪**，通过 **Pico 手柄（端口 8001）** 遥操作。旧的 TeleVuer / 关节角 IK / 脚本化按按钮链路已删除。

在线推理见 [unitree_g1_inference.md](unitree_g1_inference.md)。

---

## 环境

请先在仓库根目录按 README 创建环境并执行：

```bash
conda activate orcalab_lerobot
bash scripts/install_runtime.sh
```

需要 **OrcaLab / OrcaGym 26.7.1**（即 OrcaLab 7.1）。不要单独 `pip install -r requirements.txt`。

---

## 推荐路径：关节剥离采集

关节剥离会在模型编译前删掉下肢 / 腰 / 左臂 / 左爪的 `<joint>`，只让右臂和右爪参与 `mj_step`。渲染、相机挂点、body 树保留。这是当前实测能稳定跑的版本。

### 1. 启动 OrcaLab 并加载布局

1. `conda activate orcalab_lerobot && orcalab`
2. 加载布局：

```
src/examples/dataCollection/unitree_g1/joint_strip/uni_osc.json
```

3. 确认机器人名称为 **`g1_pick_southgrid_usda_1`**。
4. 按下面端口配置两路相机（脚本按这个读，和智元默认一致）：

| 相机 | Color Port |
|------|------------|
| 头部 `camera_head_color` / `head_cam` | **7090** |
| 右腕 `camera_wrist_r_color` | **7080** |

每路勾选 **UseNvEnc**、**Color Camera**、**Enable**。采集时再勾选 **Recording**。仅遥操（`--teleop_only`）可以不勾 Recording。

5. 点击运行，等到 OrcaGym 就绪：`localhost:50051`。

布局文件里可能已写入不同端口。以本表为准，加载后请再对一下，避免头/腕画面对调或超时。

### 2. Pico 端口转发

宇树 OSC 走 **PicoJoystick，端口 8001**（不是 TeleVuer 8012）。

```bash
adb devices
adb reverse tcp:8001 tcp:8001
adb reverse --list
```

应看到 `tcp:8001 tcp:8001`。头显侧按平时智元 Pico 遥操作的方式连接本机 8001。

第一次用某台 Pico 时，如进入 VR 被房间标定挡住：

```bash
adb shell pm disable-user --user 0 com.pvr.roomcapture
```

### 3. 启动采集

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/unitree_g1/joint_strip
bash run_strip.sh ~/southgrid_datasets/g1_osc
```

等价命令：

```bash
cd src/examples/dataCollection/unitree_g1/joint_strip
adb reverse tcp:8001 tcp:8001
OMP_NUM_THREADS=1 python g1_pick_osc_collection_tele_lerobot_strip.py \
    --task_config example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --lerobot_out ~/southgrid_datasets/g1_osc \
    --repo_id local/g1_pick_osc_strip \
    --task "按红色按钮" \
    --fps 20 --clock wall \
    --cameras head,wrist_r --camera_source websocket \
    --dls_lambda 0.2 \
    --joint_strip on --strip_col off \
    --time_step 0.001 --frame_skip 5
```

启动日志应出现类似：

```
[STRIP] ✓ 已启用
  规模: nq 113→76  nv 104→68  nu 45→15
[OSC] 变λ阻尼已启用
```

安全检查不过会自动回退原始 XML，采集仍能跑，只是没有加速。

仅遥操、不写盘：

```bash
OMP_NUM_THREADS=1 python g1_pick_osc_collection_tele_lerobot_strip.py \
    --task_config example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --teleop_only \
    --dls_lambda 0.2 \
    --joint_strip on --strip_col off \
    --time_step 0.001 --frame_skip 5
```

断点续采在采集命令上加 `--resume`。

---

## 备选：不剥离的原版 OSC

布局可用同目录上一级的 `g1_pick_tools.json` / `g1_pick_buttons.json`，或仍用 `joint_strip/uni_osc.json`。机器人名必须和场景一致，当前布局是 `g1_pick_southgrid_usda_1`。

```bash
conda activate orcalab_lerobot
cd src/examples/dataCollection/unitree_g1
adb reverse tcp:8001 tcp:8001
python g1_pick_osc_collection_tele_lerobot.py \
    --task_config ../common/example.yaml \
    --agent_name g1_pick_southgrid_usda_1 \
    --lerobot_out ~/southgrid_datasets/g1_osc \
    --repo_id local/g1_pick_osc \
    --fps 20 --clock wall \
    --cameras head,wrist_r \
    --dls_lambda 0.23 --dls_sigma_th 0.12
```

仅遥操加 `--teleop_only`，可省略 `--lerobot_out`。

---

## 按键

左臂在代码里锁定为侧平举（`shoulder_roll=π/2`，`elbow≈80°`），不跟左手柄位姿。

| 功能 | 操作 |
|------|------|
| 右臂末端跟随 | 右手柄 6DOF（开始采集后才生效） |
| 右爪 | A / B 或右扳机 |
| 左爪 | X / Y 或左扳机（剥离开启时左爪控制器会跳过） |
| 开始当前集 | 轻按左 squeeze |
| 结束并保存 | 再按左 squeeze |
| 丢弃本集 | 右 squeeze |
| 结束全部 | 左右 squeeze 同时按，或终端 `Ctrl+C` |

未按左 squeeze 开始前，手臂保持静止。

---

## 数据集格式

LeRobot v2.1。`observation.state` / `action` 均为 **18 维**：

| 维度 | 含义 |
|------|------|
| `[0:3]` | 左臂末端位置 |
| `[3:7]` | 左臂末端四元数 xyzw |
| `[7:10]` | 右臂末端位置 |
| `[10:14]` | 右臂末端四元数 xyzw |
| `[14:16]` | 左爪 inner/outer，归一化到 [0, 1] |
| `[16:18]` | 右爪 inner/outer，归一化到 [0, 1] |

默认相机：`cam_head`、`cam_wrist_r`，480×640，20 FPS。

---

## 故障排查

**现象**：`adb reverse` 后手柄无输入。**处理**：确认转发的是 **8001** 不是 8012；`adb reverse --list` 能看到映射；先开仿真再开脚本。

**现象**：关节名 / actuator 找不到。**处理**：`--agent_name` 必须等于布局里的机器人名（当前为 `g1_pick_southgrid_usda_1`）。

**现象**：相机超时或头腕对调。**处理**：把 Color Port 改成头 7090、右腕 7080，并勾选 Recording（采集模式）。

**现象**：按左 squeeze 没反应。**处理**：`fuser -k 8001/tcp` 后重开脚本；确认 Pico 已连上 8001。

**现象**：手臂发沉、仿真卡。**处理**：改用推荐的 `--joint_strip on`。
