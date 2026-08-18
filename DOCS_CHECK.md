# SouthGrid 文档核对 / 验收记录

日期：2026-08-18。按「贴合最新代码改文档、无关处不动、测试能直接跑」做完。

---

## 本轮改了什么

### 环境（从 g1_lerobot 对齐，本机已核对）

- `scripts/install_runtime.sh` / `scripts/verify_environment.py` / `requirements.*` / `constraints.txt`：安装 **orca-gym==26.7.1、orca-lab==26.7.1**
- 本机 conda `orcalab_lerobot` 实测：`orca-gym=26.7.1`，`orca-lab=26.7.1`
- `pyproject.toml` 依赖由 26.6.3 改为 26.7.1
- README 前置改为：OrcaLab 7.1 随 Conda + `install_runtime.sh` 安装，不再写「去官网装 6.3」

### 宇树（和代码改动直接相关）

- 采集文档改成 OSC + Pico **8001** + 推荐 `joint_strip`
- 推理文档改为「入口已删，暂无替换」
- `joint_strip/README.md`、`teleop.sh` 路径从 `unitree_g1_osc` / OrcaManipulation 改到 SouthGrid
- `run_strip.sh` 默认输出改为 `~/southgrid_datasets/g1_osc`（去掉个人仓库路径）

### 智元

- 只改了一处过时版本号：`docs/g1_omnipicker_collection.md` 里 `orca-gym 26.6.3` → `26.7.1`
- 采集/推理步骤、相机端口、按键说明未改

---

## 模拟跑过的检查（无仿真窗口）

- 文档里的脚本 / 布局 / yaml / URDF 路径都存在
- 旧 IK 脚本 `g1_pick_collection_tele_lerobot.py`、`eval_g1_pick_lerobot.py` 已不在仓库
- `py_compile` 通过
- `g1_pick_osc_collection_tele_lerobot_strip.py --help` 能出来，缺 `--lerobot_out` 会提示加 `--teleop_only`
- `uni_osc.json` 含 `g1_pick_southgrid_usda_1`；`example.yaml` 前缀匹配
- README / docs 里不再出现可执行的旧 IK 命令

没跑：OrcaLab 开场景、真机 Pico、完整 `mj_step` 采集。需要你本地按文档走一遍。

测试最短路径：

```bash
conda activate orcalab_lerobot
orcalab   # 加载 unitree_g1/joint_strip/uni_osc.json，头 7090 / 右腕 7080，运行仿真
adb reverse tcp:8001 tcp:8001
cd src/examples/dataCollection/unitree_g1/joint_strip
bash run_strip.sh ~/southgrid_datasets/g1_osc
```

---

## 跳过 / 疑问（未改代码，等你验收）

1. **宇树推理**：没有 OSC eval，文档只写了「暂无」。要不要补脚本？
2. **`uni_osc.json` 相机端口**：布局里 7090 / 7080 各出现一次，和脚本 `DEFAULT_CAMERA_MAP`（头 7090、右腕 7080）可能对不齐。测试若头腕对调，按文档手动改端口，还是改 JSON？
3. **左臂姿态**：代码仍是侧平举锁定。之前口头要过自然下垂，这次没改。
4. **`verify_environment.py` 仍校验 IK URDF**（`assets/g1/` + Pinocchio）。OSC 不用它，但自检会失败如果删了该目录。保持不动。
5. **智元文档流程**没动（按你说的与本次代码无关）。若智元侧也要改命令，另开一轮。
6. **Pico 8001 vs TeleVuer 8012**：OSC 走 PicoJoystick 8001。头显如果还按旧 TeleVuer 网页连 8012，会没输入。
7. **`--dls_lambda`**：strip 脚本 / `run_strip.sh` 用 0.2；原版默认 0.23 + `dls_sigma_th=0.12`。未统一。
8. **原版 OSC 脚本默认 `--agent_name unitree_humanoid_robot_1`**，当前布局是 `g1_pick_southgrid_usda_1`。文档命令已写对，但直接裸跑原版脚本会错名字。
