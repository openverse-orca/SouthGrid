# 宇树 G1 · 在线推理

旧的 `eval_g1_pick_lerobot.py`（28 维关节角 IK）已从本仓库删除，**当前没有可运行的宇树推理入口**。

采集侧已换成 OSC 18 维末端位姿，见 [unitree_g1_collection.md](unitree_g1_collection.md) 与 `src/dataStorage/g1_pick_osc_data_storage.py`：

```
[l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
 l_grip_inner_norm, l_grip_outer_norm, r_grip_inner_norm, r_grip_outer_norm]
```

用旧 IK 数据集或旧评估脚本对接现在的场景会直接对不上执行器（现在是 `_mctrl` 力矩 + 夹爪 position）。

智元推理仍可用，见 [g1_omnipicker_inference.md](g1_omnipicker_inference.md)。
