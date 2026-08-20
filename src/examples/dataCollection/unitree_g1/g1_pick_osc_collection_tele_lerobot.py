import argparse
import os
import sys
import threading
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from yaml import Loader, load

import mj_joint_strip

from conf import g1_pick_osc_conf
from controllers import controllers
from controllers.controller_task import TaskStatus
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.g1_pick_osc_data_storage import G1PickOscLeRobotStorage
from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_pick_osc_lerobot_stream"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1PickOscLeRobot",
    log_file="g1_pick_osc_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# 双臂初始姿态；左臂使用预设停靠位姿，右臂从零位开始。
_L_INIT_JOINT_VALUES = [0.0, 0.127, 0.0, 1.5708, 0.0, 0.0, 0.0]
_R_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class JointHoldController:
    """将指定关节锁定在给定位置，每 episode 重置时重新读取 qpos。"""

    def __init__(
        self,
        env,
        ctrl_name: list[str],
        init_positions: np.ndarray,
        joint_names: list[str],
    ):
        self.env = env
        self.ctrl_name = ctrl_name
        self._joint_names = joint_names
        self._joint_ids = [env.joint(n) for n in joint_names]
        self.ctrl_index = self.init_ctrl_index()
        self.hold_positions = np.asarray(
            [self._as_scalar(v) for v in np.asarray(init_positions).reshape(-1)],
            dtype=np.float32,
        )
        self.init_ctrl = self._build_init_ctrl()

    @staticmethod
    def _as_scalar(v) -> float:
        return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])

    def _build_init_ctrl(self) -> dict[int, float]:
        return {
            self.env.model.actuator_name2id(n): self._as_scalar(self.hold_positions[i])
            for i, n in enumerate(self.ctrl_name)
        }

    def init_ctrl_index(self) -> list[int]:
        return [self.env.model.actuator_name2id(n) for n in self.ctrl_name]

    def get_init_ctrl(self) -> dict[int, float]:
        return self._build_init_ctrl()

    def reset(self):
        """每 episode 重新读取当前 qpos 并更新 hold 值。"""
        qpos = self.env.query_joint_qpos(self._joint_ids)
        self.hold_positions = np.array(
            [self._as_scalar(qpos[j]) for j in self._joint_ids], dtype=np.float32
        )
        self.init_ctrl = self._build_init_ctrl()

    def run_controller(self) -> dict[int, float]:
        return {
            self.ctrl_index[i]: self._as_scalar(self.hold_positions[i])
            for i in range(len(self.ctrl_index))
        }


def lock_waist_joints(manager: DataCollectionManager, env):
    """注册腰部姿态保持控制器，使腰部维持 episode 初始姿态。"""
    joint_names = g1_pick_osc_conf.locked_waist_joints
    ctrl_name = [env.actuator(n) for n in joint_names]
    joint_ids = [env.joint(n) for n in joint_names]
    qpos = env.query_joint_qpos(joint_ids)
    init_positions = np.array(
        [
            float(np.asarray(qpos[j], dtype=np.float64).reshape(-1)[0])
            for j in joint_ids
        ],
        dtype=np.float32,
    )

    holder = JointHoldController(env, ctrl_name, init_positions, joint_names)
    manager.add_controller(holder)
    return holder


def pin_waist_joints(env, agent_name: str) -> bool:
    """在仿真步进期间保持腰部参考姿态。"""
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[CONSTRAINT] 腰部姿态约束初始化失败：模型状态不可用")
        return False

    mj, md = gym._mjModel, gym._mjData
    joint_names = g1_pick_osc_conf.locked_waist_joints
    qadrs: list[int] = []
    dadrs: list[int] = []
    q0s: list[float] = []
    for short in joint_names:
        full = f"{agent_name}_{short}"
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, full)
        if jid < 0:
            orca_logger.warning("[CONSTRAINT] 腰部姿态约束初始化失败：模型不兼容")
            return False
        qadr = int(mj.jnt_qposadr[jid])
        dadr = int(mj.jnt_dofadr[jid])
        qadrs.append(qadr)
        dadrs.append(dadr)
        q0s.append(float(md.qpos[qadr]))

    _orig_mj_step = gym.mj_step

    def _mj_step_waist_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        for _ in range(max(n, 1)):
            for qadr, q0 in zip(qadrs, q0s):
                md.qpos[qadr] = q0
            for dadr in dadrs:
                md.qvel[dadr] = 0.0
            _orig_mj_step(1)
            for qadr, q0 in zip(qadrs, q0s):
                md.qpos[qadr] = q0
            for dadr in dadrs:
                md.qvel[dadr] = 0.0
        mujoco.mj_forward(mj, md)

    gym.mj_step = _mj_step_waist_pinned
    orca_logger.info("[CONSTRAINT] 腰部姿态约束已启用")
    return True


def pin_floating_base(env, agent_name: str) -> bool:
    """在仿真步进期间保持浮动基座的参考位姿。"""
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[CONSTRAINT] 基座姿态约束初始化失败：模型状态不可用")
        return False

    mj, md = gym._mjModel, gym._mjData
    jname = f"{agent_name}_floating_base_joint"
    jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
        orca_logger.warning("[CONSTRAINT] 基座姿态约束初始化失败：模型不兼容")
        return False

    qadr = int(mj.jnt_qposadr[jid])
    dadr = int(mj.jnt_dofadr[jid])
    q0 = np.array(md.qpos[qadr : qadr + 7], dtype=np.float64, copy=True)
    _orig_mj_step = gym.mj_step

    def _mj_step_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        for _ in range(max(n, 1)):
            md.qpos[qadr : qadr + 7] = q0
            md.qvel[dadr : dadr + 6] = 0.0
            _orig_mj_step(1)
            md.qpos[qadr : qadr + 7] = q0
            md.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(mj, md)

    gym.mj_step = _mj_step_pinned
    return True


def pin_left_arm_joints(env, agent_name: str) -> bool:
    """在仿真步进期间保持左臂的预设停靠姿态。"""
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[CONSTRAINT] 左臂姿态约束初始化失败：模型状态不可用")
        return False

    mj, md = gym._mjModel, gym._mjData
    joint_names = g1_pick_osc_conf.l_arm["joint_names"]
    motor_names = g1_pick_osc_conf.l_arm["motors_names"]
    target_qpos = list(g1_pick_osc_conf.l_arm["neutral_joint_values"])

    qadrs: list[int] = []
    dadrs: list[int] = []
    for short in joint_names:
        full = f"{agent_name}_{short}"
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, full)
        if jid < 0:
            orca_logger.warning("[CONSTRAINT] 左臂姿态约束初始化失败：模型不兼容")
            return False
        qadrs.append(int(mj.jnt_qposadr[jid]))
        dadrs.append(int(mj.jnt_dofadr[jid]))

    act_ids: list[int] = []
    for short in motor_names:
        full = f"{agent_name}_{short}"
        aid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
        if aid < 0:
            orca_logger.warning("[CONSTRAINT] 左臂姿态约束初始化失败：执行器不兼容")
            return False
        act_ids.append(aid)

    # 初始化左臂参考姿态
    for qadr, q0 in zip(qadrs, target_qpos):
        md.qpos[qadr] = float(q0)
    for dadr in dadrs:
        md.qvel[dadr] = 0.0
    for aid in act_ids:
        md.ctrl[aid] = 0.0
    mujoco.mj_forward(mj, md)

    _orig_mj_step = gym.mj_step

    def _mj_step_larm_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        for _ in range(max(n, 1)):
            for qadr, q0 in zip(qadrs, target_qpos):
                md.qpos[qadr] = float(q0)
            for dadr in dadrs:
                md.qvel[dadr] = 0.0
            for aid in act_ids:
                md.ctrl[aid] = 0.0
            _orig_mj_step(1)
            for qadr, q0 in zip(qadrs, target_qpos):
                md.qpos[qadr] = float(q0)
            for dadr in dadrs:
                md.qvel[dadr] = 0.0
            for aid in act_ids:
                md.ctrl[aid] = 0.0
        mujoco.mj_forward(mj, md)

    gym.mj_step = _mj_step_larm_pinned
    orca_logger.info("[CONSTRAINT] 左臂停靠姿态约束已启用")
    return True


def pin_all_joints(env, agent_name: str) -> bool:
    """统一应用基座、腰部和左臂的参考姿态约束。"""
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[CONSTRAINT] 姿态约束初始化失败：模型状态不可用")
        return False

    mj, md = gym._mjModel, gym._mjData

    def _joint_qadr_dadr(short_name: str) -> tuple[int, int]:
        full = f"{agent_name}_{short_name}"
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, full)
        if jid < 0:
            raise ValueError("姿态约束初始化失败：模型关节不兼容")
        return int(mj.jnt_qposadr[jid]), int(mj.jnt_dofadr[jid])

    def _actuator_id(short_name: str) -> int:
        full = f"{agent_name}_{short_name}"
        aid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
        if aid < 0:
            raise ValueError("姿态约束初始化失败：模型执行器不兼容")
        return aid

    # ── 收集参考姿态 ──────────────────────────────────────────────────────
    # 基座参考位姿
    base_qadr, base_dadr = _joint_qadr_dadr("floating_base_joint")
    base_q0 = np.array(md.qpos[base_qadr : base_qadr + 7], dtype=np.float64, copy=True)

    # 腰部参考姿态
    waist_names = g1_pick_osc_conf.locked_waist_joints
    waist_qadrs: list[int] = []
    waist_dadrs: list[int] = []
    waist_q0s: list[float] = []
    for short in waist_names:
        qa, da = _joint_qadr_dadr(short)
        waist_qadrs.append(qa)
        waist_dadrs.append(da)
        waist_q0s.append(float(md.qpos[qa]))

    # 左臂停靠姿态
    l_arm_joint_names = g1_pick_osc_conf.l_arm["joint_names"]
    l_arm_motor_names = g1_pick_osc_conf.l_arm["motors_names"]
    l_arm_target = list(g1_pick_osc_conf.l_arm["neutral_joint_values"])
    larm_qadrs: list[int] = []
    larm_dadrs: list[int] = []
    for short in l_arm_joint_names:
        qa, da = _joint_qadr_dadr(short)
        larm_qadrs.append(qa)
        larm_dadrs.append(da)
    larm_act_ids: list[int] = [_actuator_id(short) for short in l_arm_motor_names]

    # ── 初始化参考姿态 ───────────────────────────────────────────────────
    md.qpos[base_qadr : base_qadr + 7] = base_q0
    md.qvel[base_dadr : base_dadr + 6] = 0.0
    for qa, q0 in zip(waist_qadrs, waist_q0s):
        md.qpos[qa] = q0
    for da in waist_dadrs:
        md.qvel[da] = 0.0
    for qa, q0 in zip(larm_qadrs, l_arm_target):
        md.qpos[qa] = float(q0)
    for da in larm_dadrs:
        md.qvel[da] = 0.0
    for aid in larm_act_ids:
        md.ctrl[aid] = 0.0
    mujoco.mj_forward(mj, md)

    # ── 单层 mj_step 包装 ───────────────────────────────────────────────
    _orig_mj_step = gym.mj_step

    def _mj_step_all_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        for _ in range(max(n, 1)):
            # 前置写回
            md.qpos[base_qadr : base_qadr + 7] = base_q0
            md.qvel[base_dadr : base_dadr + 6] = 0.0
            for qa, q0 in zip(waist_qadrs, waist_q0s):
                md.qpos[qa] = q0
            for da in waist_dadrs:
                md.qvel[da] = 0.0
            for qa, q0 in zip(larm_qadrs, l_arm_target):
                md.qpos[qa] = float(q0)
            for da in larm_dadrs:
                md.qvel[da] = 0.0
            for aid in larm_act_ids:
                md.ctrl[aid] = 0.0

            _orig_mj_step(1)

            # 后置写回
            md.qpos[base_qadr : base_qadr + 7] = base_q0
            md.qvel[base_dadr : base_dadr + 6] = 0.0
            for qa, q0 in zip(waist_qadrs, waist_q0s):
                md.qpos[qa] = q0
            for da in waist_dadrs:
                md.qvel[da] = 0.0
            for qa, q0 in zip(larm_qadrs, l_arm_target):
                md.qpos[qa] = float(q0)
            for da in larm_dadrs:
                md.qvel[da] = 0.0
            for aid in larm_act_ids:
                md.ctrl[aid] = 0.0

        # 同步约束状态
        mujoco.mj_forward(mj, md)

    gym.mj_step = _mj_step_all_pinned
    orca_logger.info("[CONSTRAINT] 任务姿态约束已启用")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 Pick OSC VR 遥操作采集 → LeRobot v2.1 格式"
    )
    parser.add_argument(
        "--level", type=str, default="default", help="场景的名称（默认 default）"
    )
    parser.add_argument(
        "--task_config", default="example.yaml", help="场景配置 YAML 文件名"
    )
    parser.add_argument(
        "--lerobot_out",
        default=None,
        help="LeRobot 数据集输出根目录（--teleop_only 时可省略）",
    )
    parser.add_argument(
        "--repo_id",
        default="local/g1_pick_osc",
        help="LeRobot repo_id（默认 local/g1_pick_osc）",
    )
    parser.add_argument(
        "--task",
        default="g1 pick osc teleoperation",
        help="任务语言描述（写入 LeRobot 元数据）",
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="采集帧率（默认 20，wall 遥操作推荐）"
    )
    parser.add_argument(
        "--clock",
        choices=("sim", "wall"),
        default="wall",
        help="采帧时钟源：wall 使用系统时钟，sim 使用仿真时钟",
    )
    parser.add_argument(
        "--resume", action="store_true", help="追加到已有数据集（断点续采）"
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--agent_name",
        default="g1_pick",
        help="OrcaStudio 场景中的 agent 名称",
    )
    parser.add_argument(
        "--cameras",
        default="head,wrist_r",
        help="启用的相机列表（逗号分隔，可选 head/wrist_r）；"
        "设为 none/off/空 或配合 --teleop_only 可关闭相机。",
    )
    parser.add_argument(
        "--cam_resolution",
        default="480x640",
        help="采集帧 resize 目标分辨率 HxW（默认 480x640）。",
    )
    parser.add_argument(
        "--camera_source",
        choices=("websocket", "mp4"),
        default="websocket",
        help="相机数据来源。websocket（默认）：内存流流式写盘；mp4：集末从服务端 MP4 提取帧。",
    )
    parser.add_argument(
        "--teleop_only",
        action="store_true",
        help="仅遥操、不保存数据；关闭相机也可正常运行（跳过相机推流与 LeRobot 写盘）。",
    )
    parser.add_argument(
        "--dls_lambda", type=float, default=0.23,
        help=(
            "OSC 阻尼最小二乘最大系数 λ_max（0 表示使用标准伪逆）。"
            "当 --dls_sigma_th 大于 0 时使用自适应阻尼，否则使用固定阻尼。"
        ),
    )
    parser.add_argument(
        "--dls_sigma_th", type=float, default=0.12,
        help=(
            "自适应阻尼触发阈值 σ_th（默认 0.12）。"
            "最小奇异值低于该阈值时逐步增加阻尼；设为 0 时使用固定阻尼。"
        ),
    )
    parser.add_argument(
        "--null_kp", type=float, default=10.0,
        help="零空间关节复原增益 kp（默认 10；临界阻尼 kd=2√kp 自动计算）。",
    )
    parser.add_argument(
        "--joint_strip", choices=["off", "on"], default="off",
        help=(
            "选择任务模型配置：on 使用采集任务配置，off 使用完整模型配置。"
            "配置校验未通过时使用默认模型。"
        ),
    )
    parser.add_argument(
        "--strip_col", choices=["off", "keep"], default="off",
        help=(
            "任务模型的碰撞配置：off 使用采集碰撞配置，keep 保留完整碰撞配置。"
        ),
    )
    parser.add_argument(
        "--time_step", type=float, default=0.001, help="MuJoCo 物理步长（秒）。")
    parser.add_argument(
        "--frame_skip", type=int, default=5, help="每控制周期的物理子步数。")
    args = parser.parse_args()

    teleop_only = bool(args.teleop_only)
    if not teleop_only and not args.lerobot_out:
        parser.error("采集模式必须指定 --lerobot_out；仅遥操请加 --teleop_only")

    lerobot_out = (
        os.path.abspath(os.path.expanduser(args.lerobot_out))
        if args.lerobot_out
        else None
    )

    # ── OSC 数值策略（在控制器创建前配置）────────────────────────────────
    from controllers.controllers import install_osc_patches
    install_osc_patches(
        dls_lambda=args.dls_lambda,
        dls_sigma_th=args.dls_sigma_th,
        null_kp=args.null_kp,
    )
    if args.dls_lambda > 0.0:
        if args.dls_sigma_th > 0.0:
            orca_logger.info(
                "[CONTROL] OSC 自适应阻尼已启用"
            )
        else:
            orca_logger.info(
                "[CONTROL] OSC 固定阻尼已启用"
            )
    else:
        orca_logger.info("[CONTROL] OSC 标准逆解已启用")
    if args.null_kp != 10.0:
        orca_logger.info("[CONTROL] OSC 零空间参数已加载")

    # ── 相机路数 / 分辨率 ────────────────────────────────────────────────────
    _CAM_KEY_MAP = {
        "head": "camera_head_color",
        "wrist_r": "camera_wrist_r_color",
    }
    _cam_raw = (args.cameras or "").strip().lower()
    _cameras_disabled = teleop_only or _cam_raw in ("", "none", "off", "0", "false")
    if _cameras_disabled:
        camera_map = {}
        orca_logger.info("相机已关闭（teleop_only / --cameras none）")
    else:
        _enabled = {k.strip() for k in args.cameras.split(",") if k.strip()}
        camera_map = {
            env_name: (key, port)
            for env_name, (key, port) in DEFAULT_CAMERA_MAP.items()
            if any(env_name == _CAM_KEY_MAP.get(k) for k in _enabled)
        }
        if not camera_map:
            orca_logger.warning("--cameras 参数未匹配到任何已知相机，回退全路")
            camera_map = DEFAULT_CAMERA_MAP

    try:
        _h, _w = (int(x) for x in args.cam_resolution.lower().split("x"))
        cam_hw_override = (_h, _w)
    except Exception:
        orca_logger.warning(
            f"--cam_resolution 格式错误 '{args.cam_resolution}'，使用默认 {DEFAULT_HW}"
        )
        cam_hw_override = DEFAULT_HW

    # ── 关节初值（双臂全零 = 站立位）─────────────────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(g1_pick_osc_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(g1_pick_osc_conf.r_arm["joint_names"], _R_INIT_JOINT_VALUES):
        default_joint_values[jn] = v

    # ── VR 设备 ───────────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  G1 Pick OSC LeRobot 数采启动中...", flush=True)
    print(f"  场景: {args.level}  fps: {args.fps}  clock: {args.clock}", flush=True)
    if teleop_only:
        print("  模式: 仅遥操（不保存数据）", flush=True)
        print("  相机: 已关闭", flush=True)
    else:
        print(f"  相机: {args.cameras}  分辨率: {args.cam_resolution}", flush=True)
        print(f"  输出目录: {lerobot_out}", flush=True)
    print("  等待 Pico 连接...", flush=True)
    print("=" * 60, flush=True)
    orca_logger.info("Creating VR device")
    pico_device = PicoJoystickDevice(PicoJoystick())

    # ── 场景管理 ──────────────────────────────────────────────────────────────
    orca_logger.info("Creating scene manager")
    with open(
        os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8"
    ) as f:
        scene_config = load(f, Loader=Loader)
    # 覆盖 agent_joint_prefix 以匹配 G1 humanoid agent 名称
    if "data_collection" in scene_config:
        scene_config["data_collection"]["agent_joint_prefix"] = f"{args.agent_name}_"
    else:
        scene_config["data_collection"] = {"agent_joint_prefix": f"{args.agent_name}_"}
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = (
        os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    )
    scene_manager.show_ui_message(
        1, "开始仿真程序，请按左右遥杆进行操作", "0xffff00", showtime=10
    )
    scene_manager.get_scene_data(script_name, "beginscene")

    # ── Storage ───────────────────────────────────────────────────────────────
    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_pick_osc", args.level)
    storage = G1PickOscLeRobotStorage(dataset_path=scratch_dir)

    _n_motor = len(g1_pick_osc_conf.gripper_l["actuator_names"]) + len(
        g1_pick_osc_conf.gripper_r["actuator_names"]
    )

    def _name_in_dict(d, name: str) -> bool:
        return bool(d) and name in d

    def _obs_callback_safe(env):
        """返回与数据集定义一致的固定观测字段。"""
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
                "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
            }
        if args.joint_strip != "on":
            return storage.obs_callback(env)

        jdict = env.model.get_joint_dict() or {}
        adict = getattr(env.model, "_actuator_dict", None) or {}

        def _joint_q(short_names):
            full = [env.joint(n) for n in short_names]
            alive = [n for n in full if _name_in_dict(jdict, n)]
            q = env.query_joint_qpos(alive) if alive else {}
            out = []
            for n in full:
                v = q.get(n, 0.0) if alive else 0.0
                out.append(np.asarray(v, dtype=np.float32).reshape(-1)[0])
            return np.asarray(out, dtype=np.float32)

        def _act_ctrl(short_names):
            full = [env.actuator(n) for n in short_names]
            out = []
            for n in full:
                if not _name_in_dict(adict, n):
                    out.append(0.0)
                    continue
                aid = env.model.actuator_name2id(n)
                out.append(float(env.ctrl[aid]) if aid < len(env.ctrl) else 0.0)
            return np.asarray(out, dtype=np.float32)

        joint_q = _joint_q(
            g1_pick_osc_conf.l_arm["joint_names"] + g1_pick_osc_conf.r_arm["joint_names"]
        )
        hand_q = _joint_q(
            g1_pick_osc_conf.gripper_l["joint_names"]
            + g1_pick_osc_conf.gripper_r["joint_names"]
        )
        hand_m = _act_ctrl(
            g1_pick_osc_conf.gripper_l["actuator_names"]
            + g1_pick_osc_conf.gripper_r["actuator_names"]
        )
        arm_m = _act_ctrl(
            g1_pick_osc_conf.l_arm["motors_names"] + g1_pick_osc_conf.r_arm["motors_names"]
        )

        ee_sites = [
            env.site(g1_pick_osc_conf.l_arm["ee_site_name"]),
            env.site(g1_pick_osc_conf.r_arm["ee_site_name"]),
        ]
        ee = env.query_site_pos_and_quat_B(
            ee_sites, [env.body(g1_pick_osc_conf.base_body)]
        )
        return {
            "/action/joint/position": joint_q,
            "/action/joint/motor": arm_m,
            "/action/effector/position": hand_q,
            "/action/effector/motor": hand_m,
            "/action/end/position": np.array(
                [ee[s]["xpos"] for s in ee_sites], dtype=np.float32
            ),
            "/action/end/orientation": np.array(
                [ee[s]["xquat"][[1, 2, 3, 0]] for s in ee_sites], dtype=np.float32
            ),
            "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
        }

    # ── 任务模型配置（在环境创建前注册）───────────────────────────────────
    strip = None
    if args.joint_strip == "on":
        # 使用当前任务的控制链配置。
        keep = mj_joint_strip.KEEP_DEFAULT + tuple(g1_pick_osc_conf.l_arm["joint_names"])
        strip = mj_joint_strip.install(
            None, args.agent_name,
            keep=keep,
            kill_collision=(args.strip_col == "off"),
            required_cameras=tuple(camera_map.keys()) or ("cam_head",),
            log=lambda m: (orca_logger.info(m), print(m, flush=True)),
        )

    # ── DataCollectionManager ─────────────────────────────────────────────────
    orca_logger.info("Creating DataCollectionManager")
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=pico_device,
        scene_manager=scene_manager,
        # 仅遥操模式不写入数据集
        data_storage=None if teleop_only else storage,
        frame_skip=args.frame_skip,
        time_step=args.time_step,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    # 按当前模型过滤初始关节状态
    stripped = bool(strip is not None and strip.applied)
    if stripped:
        alive = set(env.model.get_joint_dict() or {})
        dropped = [j for j in default_joint_values if env.joint(j) not in alive]
        for j in dropped:
            default_joint_values.pop(j)
        orca_logger.info(
            f"[MODEL] 初始状态配置完成（{len(default_joint_values)} 个关节）"
        )
        print(
            "[MODEL] 任务模型配置已加载",
            flush=True,
        )

    # ── 场景就绪后初始化控制器 + 相机 ─────────────────────────────────────────
    cameras: dict = {}
    cam_hw = cam_hw_override
    video_started = False

    try:
        env.reset()
        time.sleep(0.1)
        if strip is not None:
            mj_joint_strip.finish_install(
                env, strip, args.agent_name,
                log=lambda m: (orca_logger.info(m), print(m, flush=True)),
            )
        if manager.update_scene():
            env.set_default_joint_values(default_joint_values)

            # 夹爪控制器
            if stripped:
                orca_logger.info("[MODEL] 当前任务配置不启用左夹爪控制器")
            else:
                orca_logger.info("Adding left gripper controller")
                controllers.add_gripper_2f85_reverse_pico_controller(
                    manager,
                    env,
                    g1_pick_osc_conf.gripper_l,
                    g1_pick_osc_conf.base_body,
                    pico_device,
                    [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
                )

            orca_logger.info("Adding right gripper controller")
            controllers.add_gripper_2f85_reverse_pico_controller(
                manager,
                env,
                g1_pick_osc_conf.gripper_r,
                g1_pick_osc_conf.base_body,
                pico_device,
                [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
            )

            # Pico 位姿经 Unity→MuJoCo 坐标转换后更新右臂目标。
            orca_logger.info("Adding right arm OSC controller")
            controllers.add_arm_osc_pico_controller(
                manager,
                env,
                g1_pick_osc_conf.r_arm,
                g1_pick_osc_conf.base_body,
                pico_device,
                PicoJoystickKey.R_TRANSFORM,
            )

            # 应用任务所需的姿态约束。
            if stripped:
                # 任务模型已包含对应姿态约束。
                orca_logger.info("[MODEL] 任务模型配置已应用，姿态约束初始化完成")
            else:
                orca_logger.info(
                    "[CONSTRAINT] 正在初始化任务姿态约束"
                )
                pin_all_joints(env, args.agent_name)

            # Task + task status controller
            orca_logger.info("Setting task and task status controller")
            manager.set_task(EmptyTask(env))
            controllers.add_task_status_pico_controller(
                manager, env, pico_device, g1_pick_osc_conf.base_body
            )

            # ── 相机（teleop_only / 关闭相机时跳过）────────────────────────────
            if _cameras_disabled:
                orca_logger.info("跳过相机推流（仅遥操 / 相机已关闭）")
                print("[场景] 机器人已就绪，相机已关闭", flush=True)
            else:
                orca_logger.info(f"启用相机: {list(camera_map.keys())}")
                print(
                    "[场景] 机器人已就绪，正在连接相机...",
                    flush=True,
                )
                if args.camera_source == "websocket":
                    os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
                    env.begin_save_video(STREAM_TRIGGER_PATH)
                    video_started = True
                    orca_logger.info("相机数据流已启动")
                    cameras = bring_up_cameras(camera_map)
                    camera_map = {n: v for n, v in camera_map.items() if n in cameras}
                    if cameras:
                        cam_hw = probe_camera_hw(
                            cameras, camera_map, default_hw=cam_hw_override
                        )
                else:
                    orca_logger.info(
                        "mp4 模式：跳过 WebSocket 相机连接，每集 begin_save_video 按集触发"
                    )

    except KeyboardInterrupt:
        orca_logger.info("初始化阶段收到 Ctrl+C，正在释放相机推流会话...")
    except Exception as e:
        orca_logger.error(f"初始化失败: {e}")

    def _release_and_close():
        if video_started:
            try:
                env.stop_save_video()
                orca_logger.info("已停止相机推流")
            except Exception as stop_err:
                orca_logger.warning("相机数据流停止时遇到错误")
        close_cameras(cameras)
        try:
            scene_manager.show_ui_message(1, "", showtime=0)
            env.render()
        except Exception as ui_err:
            orca_logger.warning("界面状态清理未完成")
        try:
            env.close()
        except Exception:
            pass

    if not teleop_only and not cameras and args.camera_source != "mp4":
        orca_logger.error("没有可用相机，退出（仅遥操请加 --teleop_only）")
        _release_and_close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])
    if teleop_only or _cameras_disabled:
        orca_logger.info("仅遥操模式：不初始化相机 / 不写 LeRobot")
    elif cameras:
        orca_logger.info(
            f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}，路数={len(cameras)}"
        )
    else:
        orca_logger.info(
            f"mp4 模式，帧分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}，"
            f"相机路数={len(camera_map)}"
        )

    # ── 后台状态监控线程 ───────────────────────────────────────────────────────
    _monitor_stop = threading.Event()
    _discard_episode_event = threading.Event()  # 右Grip单按：丢弃本集并重置场景
    _first_connect_notified = {"done": False}  # 首次连接手柄提示（一次性）
    _POLL_DT = 0.02
    _STATUS_EVERY = 2.0
    _GRIP_DEBOUNCE = 0.3  # 右Grip / 双Grip 触发防抖时间（秒）

    def _hand_btn_sig(h: dict) -> tuple:
        jp = h.get("joystickPosition") or [0.0, 0.0]
        return (
            bool(h.get("gripButtonPressed")),
            bool(h.get("primaryButtonPressed")),
            bool(h.get("secondaryButtonPressed")),
            bool(h.get("joystickPressed")),
            round(float(h.get("triggerValue", 0.0)), 1),
            round(float(jp[0]), 1),
            round(float(jp[1]), 1),
        )

    def _fmt_hand_sig(sig: tuple) -> str:
        grip, prim, sec, jpr, trig, jx, jy = sig
        return (
            f"Grip={int(grip)} 扳机={trig:.1f} 主键={int(prim)} "
            f"副键={int(sec)} 摇杆=({jx:.1f},{jy:.1f})"
        )

    def _status_monitor():
        last_wall = time.perf_counter()
        try:
            last_sim = float(env.data.time)
        except Exception:
            last_sim = 0.0
        _prev_sig = None
        _last_status = last_wall
        _r_grip_only_prev = False
        _both_grip_prev = False
        _grip_debounce_t = 0.0

        while not _monitor_stop.wait(_POLL_DT):
            try:
                pj = pico_device.pico_joystick
                n_clients = len(pj.clients)
                raw_key = pj.current_key_state
                now = time.perf_counter()

                if n_clients > 0 and not _first_connect_notified["done"]:
                    _first_connect_notified["done"] = True
                    orca_logger.info("[Pico] 手柄已连接，可以开始采集")
                    try:
                        scene_manager.show_ui_message(
                            1, "已连接手柄，可以开始采集", "0x00ff00", showtime=5
                        )
                    except Exception:
                        pass

                if n_clients > 0 and raw_key is not None:
                    l_sig = _hand_btn_sig(raw_key.get("leftHand", {}) or {})
                    r_sig = _hand_btn_sig(raw_key.get("rightHand", {}) or {})
                    sig = (l_sig, r_sig)
                    if sig != _prev_sig:
                        orca_logger.info(
                            "[Pico] 输入状态已更新"
                        )
                        _prev_sig = sig

                    l_grip = l_sig[0]
                    r_grip = r_sig[0]
                    both_grip = l_grip and r_grip
                    r_grip_only = r_grip and not l_grip

                    if now - _grip_debounce_t >= _GRIP_DEBOUNCE:
                        if both_grip and not _both_grip_prev:
                            orca_logger.info(
                                "[Grip] 左右Grip同按 → 终止"
                                + (
                                    "遥操"
                                    if teleop_only
                                    else "全部采集，等待编码完成后退出"
                                )
                            )
                            try:
                                scene_manager.show_ui_message(
                                    1,
                                    "遥操终止..."
                                    if teleop_only
                                    else "采集终止，等待保存...",
                                    "0xff4400",
                                    showtime=0,
                                )
                            except Exception:
                                pass
                            manager._shutdown_requested = True  # noqa: SLF001
                            _grip_debounce_t = now
                        elif r_grip_only and not _r_grip_only_prev:
                            orca_logger.info(
                                "[Grip] 右Grip单按 → "
                                + ("重置场景" if teleop_only else "丢弃本集，重置场景")
                            )
                            try:
                                scene_manager.show_ui_message(
                                    1,
                                    "重置场景..."
                                    if teleop_only
                                    else "已丢弃，重置场景...",
                                    "0xff0000",
                                    showtime=2,
                                )
                            except Exception:
                                pass
                            _discard_episode_event.set()
                            manager._shutdown_requested = True  # noqa: SLF001
                            _grip_debounce_t = now

                    _r_grip_only_prev = r_grip_only
                    _both_grip_prev = both_grip

                if now - _last_status < _STATUS_EVERY:
                    continue
                _last_status = now
                sim_now = float(env.data.time)
                d_wall = now - last_wall
                d_sim = sim_now - last_sim
                last_wall, last_sim = now, sim_now

                if n_clients == 0:
                    orca_logger.info("[Pico] 无客户端连接（请确认 Pico 端 App 已启动）")
                else:
                    orca_logger.info("[Pico] 手柄连接正常")

                if d_sim < 0:
                    orca_logger.info("[监控] 仿真已重置，等待下一集...")
                    continue
                rt = (d_sim / d_wall) if d_wall > 0 else 0.0
                ctrl_dt = float(env.dt) if float(env.dt) > 0 else 1.0
                loop_hz = ((d_sim / ctrl_dt) / d_wall) if d_wall > 0 else 0.0
                orca_logger.info("[监控] 系统运行正常")
            except Exception:
                pass

    _monitor = threading.Thread(target=_status_monitor, daemon=True)
    _monitor.start()

    # ── 采集前输入门控 ──────────────────────────────────────────────────────
    # 场景重置后、按左Grip开始采集前，机械臂/夹爪不响应手柄（保持静止）；
    # 仅放行 L_GRIPBUTTON（任务状态：开始/保存）。开始采集(RUNNING)后放行除锁定外按键。
    # 左臂位姿(L_TRANSFORM)全程保持预设停靠姿态，不响应手柄。
    _LOCKED_KEYS: set = {PicoJoystickKey.L_TRANSFORM}
    _all_pico_keys = [k for k in pico_device.keys if k not in _LOCKED_KEYS]
    _pre_start_keys = [k for k in _all_pico_keys if k == PicoJoystickKey.L_GRIPBUTTON]

    def _gated_pico_update():
        tsc = manager.task_status_controller
        if tsc is not None and tsc.current_status == TaskStatus.RUNNING:
            pico_device.pico_joystick.update(_all_pico_keys)
        else:
            pico_device.pico_joystick.update(_pre_start_keys)

    pico_device.update = _gated_pico_update

    print("", flush=True)
    print("=" * 60, flush=True)
    if teleop_only:
        print("  ✓ 场景加载完成，进入仅遥操主循环", flush=True)
        print(f"  任务: {args.task}", flush=True)
        print("  数据保存: 关闭", flush=True)
        print("  相机: 关闭", flush=True)
    else:
        print("  ✓ 场景加载完成，进入采集主循环", flush=True)
        print(f"  任务: {args.task}", flush=True)
        print(f"  数据输出: {lerobot_out}", flush=True)
    print("-" * 60, flush=True)
    print("  【操作按键】", flush=True)
    print("  左臂移动    保持预设停靠姿态", flush=True)
    print("  右臂移动    右手柄位姿 (持握激活)", flush=True)
    print("  左夹爪      X / Y 键 或 左扳机", flush=True)
    print("  右夹爪      A / B 键 或 右扳机", flush=True)
    print("  浮动基座    保持参考位姿", flush=True)
    print("  腰部关节    保持参考姿态", flush=True)
    print("-" * 60, flush=True)
    if teleop_only:
        print("  【遥操流程（不保存）】", flush=True)
        print(
            "  注意：开始前机械臂保持静止，不响应手柄；开始后才随手柄运动", flush=True
        )
        print("  开始遥操  →  轻按一下【左手柄 Grip 侧握键】", flush=True)
        print("  结束本轮  →  再轻按一下【左手柄 Grip 侧握键】（不写盘）", flush=True)
        print("  重置场景  →  轻按【右手柄 Grip 侧握键】", flush=True)
        print("  退出      →  【左右 Grip 同时按下】或 Ctrl+C", flush=True)
        ui_msg = "左Grip×1=开始 左Grip×2=结束 右Grip=重置 左右Grip同按=退出（不保存）"
    else:
        print("  【采集流程（强制保存）】", flush=True)
        print(
            "  注意：开始采集前机械臂保持静止，不响应手柄；开始后才随手柄运动",
            flush=True,
        )
        print("  第1步 开始采集  →  轻按一下【左手柄 Grip 侧握键】", flush=True)
        print("  第2步 遥操作完成后", flush=True)
        print(
            "         保 存   →  再轻按一下【左手柄 Grip 侧握键】（无论成功与否均保存）",
            flush=True,
        )
        print(
            "  放弃本集        →  轻按【右手柄 Grip 侧握键】（丢弃数据，重置场景，继续采集）",
            flush=True,
        )
        print(
            "  终止全部采集    →  【左右 Grip 同时按下】（等待编码保存后自动退出）",
            flush=True,
        )
        print("  强制退出        →  终端按 Ctrl+C", flush=True)
        ui_msg = "左Grip×1=开始 左Grip×2=保存 右Grip=丢弃重置 左右Grip同按=退出"
    print("=" * 60, flush=True)
    print("", flush=True)

    try:
        scene_manager.show_ui_message(
            1,
            ui_msg,
            "0x00ff00",
            showtime=0,
        )
    except Exception as ui_err:
        orca_logger.warning("界面提示暂不可用")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    if teleop_only:
        orca_logger.info("开始仅遥操（不保存数据）")
    else:
        orca_logger.info(f"开始采集，LeRobot 输出: {lerobot_out}")
    writer = None
    try:
        if teleop_only:
            _ep_idx = 0
            while not manager._shutdown_requested:  # noqa: SLF001
                _ep_idx += 1
                env.reset()
                time.sleep(0.1)
                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止遥操")
                    break
                env.set_default_joint_values(default_joint_values)

                orca_logger.info(f"========== 遥操第 {_ep_idx} 轮 ==========")
                print(
                    f"\n>>> 遥操第 {_ep_idx} 轮（按左Grip开始，再按左Grip结束；不保存）",
                    flush=True,
                )

                _ep_t0 = time.perf_counter()
                manager.run_episode()
                _ep_dur = time.perf_counter() - _ep_t0

                if _discard_episode_event.is_set():
                    _discard_episode_event.clear()
                    manager._shutdown_requested = False  # noqa: SLF001
                    orca_logger.info(f"[EP {_ep_idx}] 已重置场景（右Grip）")
                    continue

                if manager._shutdown_requested:  # noqa: SLF001
                    orca_logger.info("结束遥操（左右Grip/Ctrl+C）")
                    print("\n[结束] 已停止遥操", flush=True)
                    break

                orca_logger.info(
                    f"[EP {_ep_idx}] 本轮结束，时长 {_ep_dur:.1f}s（未保存）"
                )
                print(f">>> 本轮结束（未保存），时长 {_ep_dur:.1f}s", flush=True)
        else:
            writer = LeRobotDatasetWriter.create(
                repo_id=args.repo_id,
                root=lerobot_out,
                fps=args.fps,
                camera_map=camera_map,
                state_dim=storage.state_dim,
                state_names=storage.state_names,
                cam_shape=cam_shape,
                resume=args.resume,
                robot_type="g1_pick_osc",
            )
            storage.configure_lerobot(
                fps=args.fps,
                cameras=cameras,
                camera_map=camera_map,
                target_hw=cam_hw,
                writer=writer,
                task=args.task,
                clock=args.clock,
                camera_source=args.camera_source,
            )
            with writer:
                _ep_idx = 0
                while not manager._shutdown_requested:  # noqa: SLF001
                    _ep_idx += 1
                    env.reset()
                    time.sleep(0.1)
                    if not manager.update_scene():
                        orca_logger.info("update_scene 失败，停止采集")
                        break
                    env.set_default_joint_values(default_joint_values)

                    # mp4 模式：每集开录
                    ep_dir: str | None = None
                    ep_start_wall: float | None = None
                    if args.camera_source == "mp4":
                        ep_dir = os.path.join(scratch_dir, "mp4", f"ep_{_ep_idx:06d}")
                        os.makedirs(os.path.join(ep_dir, "video"), exist_ok=True)
                        ep_start_wall = time.perf_counter()
                        env.begin_save_video(ep_dir)
                        video_started = True

                    _collecting_ep_no = writer.num_episodes + 1
                    orca_logger.info(
                        f"========== 正在采集第 {_collecting_ep_no} 集 =========="
                    )
                    print(
                        f"\n>>> 正在采集第 {_collecting_ep_no} 集（按左Grip开始，再按左Grip保存）",
                        flush=True,
                    )

                    _ep_t0 = time.perf_counter()
                    _task_is_success, _rec_start, _rec_end, _init_qpos = (
                        manager.run_episode()
                    )
                    _ep_dur = time.perf_counter() - _ep_t0

                    _ep_frames = storage.buffered_frame_count

                    if args.camera_source == "mp4" and video_started:
                        try:
                            env.stop_save_video()
                        except Exception as _stop_e:
                            orca_logger.warning(
                                f"stop_save_video 失败（可忽略）: {_stop_e}"
                            )
                        video_started = False

                    # 右Grip单按：丢弃本集并继续下一集
                    if _discard_episode_event.is_set():
                        _discard_episode_event.clear()
                        manager._shutdown_requested = False  # noqa: SLF001
                        storage.clear_data()
                        orca_logger.info(
                            f"[EP {_ep_idx}] 已丢弃本集（右Grip），重置场景"
                        )
                        continue

                    # Ctrl+C 或 左右Grip同按：终止全部采集
                    if manager._shutdown_requested:  # noqa: SLF001
                        orca_logger.info(
                            "结束采集（左右Grip/Ctrl+C），丢弃当前未保存集"
                        )
                        print("\n[结束采集] 已停止采集，丢弃当前未保存集", flush=True)
                        storage.clear_data()
                        break

                    _cap_fps = (_ep_frames / _ep_dur) if _ep_dur > 0 else 0.0
                    orca_logger.info(
                        f"[EP {_ep_idx}] 时长 {_ep_dur:.1f}s / 捕获 {_ep_frames} 帧 / "
                        f"capture-fps {_cap_fps:.1f}（目标 {args.fps}）"
                    )
                    if args.clock == "wall" and _cap_fps < 0.9 * args.fps:
                        orca_logger.warning(
                            f"[EP {_ep_idx}] 墙钟模式欠采：有效 capture-fps {_cap_fps:.1f} < "
                            f"目标 {args.fps} 的 90%，建议降低 --fps。"
                        )

                    # 强制保存
                    orca_logger.info(
                        f"[EP {_ep_idx}] 正在保存本集数据"
                    )
                    storage.save_data(
                        task_info=manager.task.get_task_info(),
                        scene_info=scene_manager.get_scene_info(),
                        task_description=manager.task.get_task_description(),
                        episode_video_dir=ep_dir,
                        ep_start_wall=ep_start_wall,
                    )
                    orca_logger.info(
                        f"✓ 本集已保存，当前共采集 {writer.num_episodes} 集 / {writer.num_frames} 帧"
                    )
                    print(
                        f">>> ✓ 本集已保存，当前共采集 {writer.num_episodes} 集",
                        flush=True,
                    )

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}")
    finally:
        _monitor_stop.set()
        if writer is not None:
            try:
                orca_logger.info("正在等待所有视频编码完成，请勿关闭程序...")
                print("\n[退出] 正在等待所有视频编码完成，请勿关闭程序...", flush=True)
                writer.close()
                orca_logger.info("✓ 所有视频编码已完成")
                print("[退出] ✓ 所有视频编码已完成", flush=True)
            except Exception:
                pass
        if video_started:
            try:
                env.stop_save_video()
                orca_logger.info("已停止相机推流")
            except Exception as stop_err:
                orca_logger.warning("相机数据流停止时遇到错误")
        close_cameras(cameras)
        try:
            env.close()
        except Exception:
            pass
        if teleop_only:
            summary = "遥操结束（未保存数据）"
        elif writer is not None:
            summary = f"采集结束，共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        else:
            summary = "采集结束（未成功创建数据集）"
        orca_logger.info(summary)
        print(f"\n{'=' * 60}", flush=True)
        print(f"  {summary}", flush=True)
        if lerobot_out is not None:
            print(f"  数据位于: {lerobot_out}", flush=True)
        print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("已收到中断请求")
    except Exception as e:
        OrcaLog.get_instance().error(f"程序异常: {e}")
    finally:
        orca_logger.info("程序已退出")
        os._exit(0)
