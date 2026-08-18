# SouthGrid 待办事项

## 文档与环境配置（暂未更新）

### 1. README 更新
- `g1_lerobot` 分支有更新的 README（OrcaLab 7.1 环境安装步骤），需要合并进来
  - 提交：`f6e7a26 Install OrcaLab 7.1 with the delivery Conda stack`
  - 提交：`bd5b6ac docs: refresh g1_lerobot README and user guides`
- 当前 README 描述的是 OrcaLab 6.3，实际应使用 7.1

### 2. 推理脚本（已删除 IK 版本）
- `src/examples/inference/unitree_g1/eval_g1_pick_lerobot.py` 已删除（IK 位置控制版本）
- 需要新增基于 OSC 状态空间（18 维末端位姿）的推理脚本
- 状态格式：`[l_pos(3), l_quat(4), r_pos(3), r_quat(4), l_grip(2), r_grip(2)]`
- 参考：`src/dataStorage/g1_pick_osc_data_storage.py` 的 `G1PickOscLeRobotStorage`

### 3. 文档目录
- `docs/unitree_g1_inference.md` 仍引用旧 IK 推理脚本，需要更新
- `docs/` 下其他文档中的命令行示例需检查是否还与 OSC 脚本匹配

### 4. 环境依赖
- `requirements.txt` / Conda 环境文件中的 OrcaLab 版本需确认为 7.1
- OSC 控制器依赖 `scipy`（`from scipy.spatial.transform import Rotation`），需确认已在依赖中列出

## 已合并的两分支主要内容
- 基础：`g1_lerobot_osc`（OSC 控制链路）
- 取 `g1_lerobot` 的 SIGINT 禁用修复（`against_sea@163.com` 提交）
- 移除全部 IK 相关文件（`g1_pick_conf`、IK controller、IK data_storage、IK devices）
