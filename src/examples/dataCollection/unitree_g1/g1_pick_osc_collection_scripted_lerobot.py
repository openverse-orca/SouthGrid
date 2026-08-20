"""基于 record_waypoints.py 路点文件的 G1 Pick OSC 自动采集。

脚本将路点 YAML 插值为右臂与右夹爪轨迹，并在每集结束后保存数据。
YAML 的 segments 包含 r_target_b、r_quat_b、gripper_r 和 steps；steps 按
控制步计（env.dt = time_step × frame_skip）。多个文件用逗号分隔，并按给定
顺序在同一集内依次执行。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
from yaml import Loader, load, safe_load

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
_common_dir = os.path.abspath(os.path.join(base_dir, "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

import data_collection_scripted as scripted
import mj_joint_strip
import g1_pick_osc_collection_tele_lerobot as tele

from conf import g1_pick_osc_conf
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_task import TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
    install_osc_patches,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_pick_osc_data_storage import G1PickOscLeRobotStorage
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_pick_osc_lerobot_stream"
_L_INIT_JOINT_VALUES = [0.0, 0.127, 0.0, 1.5708, 0.0, 0.0, 0.0]

log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1PickOscScripted",
    log_file="g1_pick_osc_scripted.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def _load_waypoint_segments(path: str) -> tuple[float, float, list[dict]]:
    """读 record_waypoints.py 输出的 YAML，转成 build_segmented_trajectory 的段列表。"""
    with open(path, "r", encoding="utf-8") as f:
        spec = safe_load(f) or {}

    segs = spec.get("segments") or []
    if not segs:
        raise ValueError(f"{path}: segments 为空，无法构建轨迹")

    r_range = g1_pick_osc_conf.gripper_r["actuator_ranges"][0]
    g_open = float(spec.get("gripper_open", r_range[0]))
    g_close = float(spec.get("gripper_close", r_range[1]))

    out: list[dict] = []
    for i, seg in enumerate(segs):
        if not isinstance(seg, dict) or "r_target_b" not in seg or "r_quat_b" not in seg:
            raise ValueError(f"{path}: segments[{i}] 缺少 r_target_b / r_quat_b")
        pos = [float(v) for v in seg["r_target_b"]]
        quat = [float(v) for v in seg["r_quat_b"]]
        if len(pos) != 3 or len(quat) != 4:
            raise ValueError(
                f"{path}: segments[{i}] 维度错误（r_target_b 需 3 维，r_quat_b 需 4 维 xyzw）"
            )
        out.append(
            {
                "steps": int(seg.get("steps", 300)),
                # 左臂保持预设停靠姿态
                "l_hold": True,
                "r_target_b": pos,
                "r_quat_b": quat,
                "gripper_r": str(seg.get("gripper_r", "open")).strip().lower(),
            }
        )
    return g_open, g_close, out


class G1OscScriptedDevice(AbstractDevice):
    """把离线算好的 B 系末端位姿逐步写进右臂 OSC 与右爪控制器。

    轨迹第一步把任务状态推到 RUNNING（开始录帧），最后一步推到 END（结束本集）。
    """

    def __init__(
        self,
        env,
        r_arm,
        r_grip: Controller2F85Reverse,
        task_status: TaskStatusController,
        r_pos: np.ndarray,
        r_quat_xyzw: np.ndarray,
        r_grip_motor: np.ndarray,
        waypoint_marks: dict[int, str] | None = None,
        track_log_every: int = 0,
        track_ki: float = 0.02,
        track_clamp: float = 0.08,
        action_repeat: int = 1,
    ):
        super().__init__()
        n = len(r_pos)
        if not (len(r_quat_xyzw) == n and len(r_grip_motor) == n):
            raise ValueError("轨迹各通道长度不一致")
        self.env = env
        self.r_arm = r_arm
        self.r_grip = r_grip
        self.task_status = task_status
        self.r_pos = r_pos
        self.r_quat_xyzw = r_quat_xyzw
        self.r_grip_motor = r_grip_motor
        self.waypoint_marks = waypoint_marks or {}
        self.track_log_every = max(0, int(track_log_every))
        self.track_ki = max(0.0, float(track_ki))
        self.track_clamp = abs(float(track_clamp))
        self.action_repeat = max(1, int(action_repeat))
        self._icorr = np.zeros(3, dtype=np.float64)
        self.n = n
        self.t = 0
        self._hold = 0
        self._ee_site = env.site(g1_pick_osc_conf.r_arm["ee_site_name"])
        self._base_body = env.body(g1_pick_osc_conf.base_body)

    @property
    def finished(self) -> bool:
        return self.t >= self.n

    def _tracking_err(self, target) -> tuple[float, np.ndarray]:
        """当前右末端相对该步目标的 B 系误差：(距离 mm, 分轴 mm)。"""
        res = self.env.query_site_pos_and_quat_B([self._ee_site], [self._base_body])
        cur = np.asarray(res[self._ee_site]["xpos"], dtype=np.float64).reshape(3)
        d = (cur - np.asarray(target, dtype=np.float64)) * 1000.0
        return float(np.linalg.norm(d)), d

    def _fmt_err(self, target) -> str:
        norm, d = self._tracking_err(target)
        return f"{norm:.0f}mm (dx={d[0]:+.0f} dy={d[1]:+.0f} dz={d[2]:+.0f})"

    def update(self):
        if self.t >= self.n:
            return
        first_hold = self._hold == 0
        if self.t == 0 and first_hold:
            self.task_status.update_task_status(True)

        mark = self.waypoint_marks.get(self.t) if first_hold else None
        if mark is not None:
            if self.t == 0:
                orca_logger.info(f"[轨迹] {mark}")
            else:
                # 报告上一段的跟踪状态
                orca_logger.info(
                    f"[轨迹] {mark}"
                )
        elif first_hold and self.track_log_every and self.t % self.track_log_every == 0:
            orca_logger.info(
                f"[跟踪] 轨迹进度 {self.t}/{self.n}"
            )

        target = np.asarray(self.r_pos[self.t], dtype=np.float64)
        # 每个轨迹采样仅更新一次积分项。
        if self.track_ki > 0.0:
            if first_hold:
                _, d_mm = self._tracking_err(target)
                self._icorr = np.clip(
                    self._icorr - self.track_ki * (d_mm / 1000.0),
                    -self.track_clamp,
                    self.track_clamp,
                )
            target = target + self._icorr

        self.r_arm.update_action_position(target)
        self.r_arm.update_action_axisangle(self.r_quat_xyzw[self.t])
        self.r_grip.update_ctrl(
            np.full(len(self.r_grip.ctrl_index), self.r_grip_motor[self.t], dtype=np.float32)
        )

        self._hold += 1
        if self._hold < self.action_repeat:
            return
        self._hold = 0
        if self.t == self.n - 1:
            orca_logger.info("[轨迹] 已到达末尾")
            self.task_status.update_task_status(True)
        self.t += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 Pick OSC 脚本化路点回放采集 → LeRobot v2.1 格式"
    )
    parser.add_argument("--level", default="default", help="场景的名称（默认 default）")
    parser.add_argument(
        "--task_config", default="example.yaml", help="场景配置 YAML 文件名"
    )
    parser.add_argument(
        "--waypoint_files",
        default=os.path.join(base_dir, "waypoint_tool", "my_waypoint_tool1.yaml"),
        help="路点 YAML 路径，逗号分隔可传多个，按顺序在同一集内依次执行",
    )
    parser.add_argument(
        "--lerobot_out",
        default=None,
        help="LeRobot 数据集输出根目录（--dry_run 时可省略）",
    )
    parser.add_argument(
        "--repo_id",
        default="local/g1_pick_osc_scripted",
        help="LeRobot repo_id（默认 local/g1_pick_osc_scripted）",
    )
    parser.add_argument(
        "--task", default="按红色按钮", help="任务语言描述（写入 LeRobot 元数据）"
    )
    parser.add_argument("--num_episodes", type=int, default=1, help="采集集数（默认 1）")
    parser.add_argument(
        "--fps", type=int, default=20, help="采集帧率（默认 20）"
    )
    parser.add_argument(
        "--clock",
        choices=("sim", "wall"),
        default="sim",
        help="采帧时钟源：sim 使用仿真时钟，wall 使用系统时钟",
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
        help="启用的相机列表（逗号分隔，可选 head/wrist_r）；none/off/空 或 --dry_run 关闭相机。",
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
        "--track_log_every",
        type=int,
        default=0,
        help="每 N 个控制步输出一次轨迹进度（默认 0 = 只在路点处输出）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅执行轨迹，不启用相机或写入数据集。",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="轨迹整体提速倍率（默认 1.0；2.0=快一倍，各段步数除以该值）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="覆盖每个路点段的控制步数（默认 0 = 用 YAML 里的 steps）",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=200,
        help="夹爪切换前的末端稳定段控制步数（默认 200 = 1s；0 关闭）。",
    )
    parser.add_argument(
        "--hold_steps",
        type=int,
        default=100,
        help="末尾保持段的控制步数，给最后一次开/合爪留沉降时间（默认 100）",
    )
    parser.add_argument(
        "--track_ki",
        type=float,
        default=0.02,
        help="末端位置积分增益 ki（默认 0.02；0 = 关闭）；每个轨迹采样更新一次。",
    )
    parser.add_argument(
        "--track_clamp",
        type=float,
        default=0.08,
        help="积分器偏置限幅（米，默认 0.08）",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=0.0,
        help="OSC 阻抗刚度 kp（默认 0 = 沿用 osc_pose 配置，通常 150；>0 时覆盖并设 kd=2√kp）",
    )
    parser.add_argument(
        "--dls_lambda", type=float, default=0.23,
        help="OSC 阻尼最小二乘最大系数 λ_max（0 表示使用标准伪逆）。",
    )
    parser.add_argument(
        "--dls_sigma_th", type=float, default=0.12,
        help="自适应阻尼触发阈值 σ_th（0 表示使用固定阻尼）。",
    )
    parser.add_argument(
        "--null_kp", type=float, default=10.0,
        help="零空间关节复原增益 kp（默认 10；kd=2√kp 自动计算）。",
    )
    parser.add_argument(
        "--action_repeat",
        type=int,
        default=1,
        help="每个轨迹采样重复执行的控制步数（默认 1；增大可给 OSC 更多收敛时间，任务时长同步变长）。",
    )
    parser.add_argument(
        "--joint_strip", choices=["off", "on"], default="on",
        help="选择任务模型配置：on 使用采集任务配置，off 使用完整模型配置。",
    )
    parser.add_argument(
        "--strip_col", choices=["off", "keep"], default="off",
        help="任务模型的碰撞配置：off 使用采集配置，keep 保留完整配置。",
    )
    parser.add_argument(
        "--time_step", type=float, default=0.001, help="MuJoCo 物理步长（秒）。"
    )
    parser.add_argument(
        "--frame_skip", type=int, default=5, help="每控制周期的物理子步数。"
    )
    args = parser.parse_args()

    dry_run = bool(args.dry_run)
    if not dry_run and not args.lerobot_out:
        parser.error("采集模式必须指定 --lerobot_out；只想验证动作请加 --dry_run")
    if args.num_episodes < 1 or args.num_episodes > 9999:
        parser.error("--num_episodes 必须是 1–9999 的整数")
    if args.speed <= 0:
        parser.error("--speed 必须大于 0")
    if args.action_repeat < 1:
        parser.error("--action_repeat 必须 >= 1")
    if args.frame_skip < 1:
        parser.error("--frame_skip 必须 >= 1")
    if args.time_step <= 0:
        parser.error("--time_step 必须大于 0")
    if args.track_ki < 0:
        parser.error("--track_ki 必须 >= 0")
    if args.kp < 0:
        parser.error("--kp 必须 >= 0")

    lerobot_out = (
        os.path.abspath(os.path.expanduser(args.lerobot_out))
        if args.lerobot_out
        else None
    )

    # ── 路点加载与启动前校验 ──────────────────────────────────────────────
    wp_paths = [p.strip() for p in str(args.waypoint_files).split(",") if p.strip()]
    if not wp_paths:
        parser.error("--waypoint_files 不能为空")
    wp_paths = [p if os.path.isabs(p) else os.path.join(base_dir, p) for p in wp_paths]

    pairs: list[tuple[dict, str]] = []
    g_open = g_close = None
    try:
        for path in wp_paths:
            f_open, f_close, segs = _load_waypoint_segments(path)
            if g_open is None:
                g_open, g_close = f_open, f_close
            elif (f_open, f_close) != (g_open, g_close):
                orca_logger.warning(
                    f"{path}: gripper_open/close 与首个文件不一致，沿用首个文件的量程"
                )
            for i, seg in enumerate(segs):
                raw_steps = int(args.steps) if args.steps > 0 else int(seg["steps"])
                seg["steps"] = max(1, int(round(raw_steps / float(args.speed))))
                pairs.append(
                    (
                        seg,
                        f"{os.path.basename(path)} 路点 {i + 1}/{len(segs)}"
                        f" → gripper_r={seg['gripper_r']}",
                    )
                )
    except Exception as e:
        orca_logger.error(f"路点加载失败: {e}")
        print(f"[错误] 路点加载失败: {e}", flush=True)
        return

    # 在夹爪状态切换前插入末端稳定段
    settle_steps = max(0, int(args.settle_steps))
    all_pairs: list[tuple[dict, str]] = []
    prev_grip = "open"
    for seg, label in pairs:
        grip = seg["gripper_r"]
        if settle_steps > 0 and grip != prev_grip and all_pairs:
            all_pairs.append(
                (
                    {
                        "steps": settle_steps,
                        "l_hold": True,
                        "r_hold": True,
                        "gripper_r": prev_grip,
                    },
                    f"沉降 {settle_steps} 步（保持 {prev_grip}，等跟到位再动夹爪）",
                )
            )
        all_pairs.append((seg, label))
        prev_grip = grip

    if args.hold_steps > 0:
        all_pairs.append(
            (
                {
                    "steps": int(args.hold_steps),
                    "l_hold": True,
                    "r_hold": True,
                    "gripper_r": "hold",
                },
                "末尾保持（等夹爪沉降）",
            )
        )

    all_segments = [seg for seg, _ in all_pairs]

    # 段起始控制步 → 提示文本，device 跨段时打日志
    waypoint_marks: dict[int, str] = {}
    total_steps = 0
    for seg, label in all_pairs:
        waypoint_marks[total_steps] = label
        total_steps += int(seg["steps"])
    env_dt = float(args.time_step) * int(args.frame_skip)
    ctrl_steps = total_steps * int(args.action_repeat)
    duration_s = ctrl_steps * env_dt

    # ── OSC 数值策略（在控制器创建前配置）────────────────────────────────
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
            orca_logger.info("[CONTROL] OSC 固定阻尼已启用")
    else:
        orca_logger.info("[CONTROL] OSC 标准逆解已启用")
    orca_logger.info("[CONTROL] 轨迹控制参数已加载")

    # ── 相机路数 / 分辨率 ────────────────────────────────────────────────────
    _CAM_KEY_MAP = {"head": "camera_head_color", "wrist_r": "camera_wrist_r_color"}
    _cam_raw = (args.cameras or "").strip().lower()
    _cameras_disabled = dry_run or _cam_raw in ("", "none", "off", "0", "false")
    if _cameras_disabled:
        camera_map = {}
        orca_logger.info("相机已关闭（--dry_run / --cameras none）")
    else:
        _enabled = {k.strip() for k in args.cameras.split(",") if k.strip()}
        camera_map = {
            env_name: (key, port)
            for env_name, (key, port) in DEFAULT_CAMERA_MAP.items()
            if any(env_name == _CAM_KEY_MAP.get(k) for k in _enabled)
        }
        if not camera_map:
            orca_logger.warning("--cameras 未匹配已配置相机，将使用完整相机配置")
            camera_map = DEFAULT_CAMERA_MAP

    try:
        _h, _w = (int(x) for x in args.cam_resolution.lower().split("x"))
        cam_hw_override = (_h, _w)
    except Exception:
        orca_logger.warning(
            f"--cam_resolution 格式错误 '{args.cam_resolution}'，使用默认 {DEFAULT_HW}"
        )
        cam_hw_override = DEFAULT_HW

    # ── 关节初值（左臂停靠自然下垂，右臂站立位）──────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(g1_pick_osc_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(g1_pick_osc_conf.r_arm["joint_names"], tele._R_INIT_JOINT_VALUES):
        default_joint_values[jn] = v

    print("=" * 62, flush=True)
    print("  G1 Pick OSC 脚本化路点回放采集", flush=True)
    print(f"  任务: {args.task}", flush=True)
    print(f"  路点: {', '.join(os.path.basename(p) for p in wp_paths)}", flush=True)
    print(
        f"  轨迹: {len(all_segments)} 段 / {total_steps} 采样"
        f" × repeat={args.action_repeat} = {ctrl_steps} 控制步"
        f"（约 {duration_s:.1f}s，env.dt={env_dt * 1000:.0f}ms，speed={args.speed:.2f}x）",
        flush=True,
    )
    print(f"  集数: {args.num_episodes}  fps: {args.fps}  clock: {args.clock}", flush=True)
    print(
        f"  [配置] DLS λ={args.dls_lambda}  σ_th={args.dls_sigma_th}  null_kp={args.null_kp}",
        flush=True,
    )
    print(
        f"  [配置] kp={args.kp}  action_repeat={args.action_repeat}  "
        f"frame_skip={args.frame_skip}  time_step={args.time_step}",
        flush=True,
    )
    print(
        f"  [配置] 积分器 ki={args.track_ki}  clamp={args.track_clamp}m",
        flush=True,
    )
    if dry_run:
        print("  模式: 只跑轨迹（不保存数据、不开相机）", flush=True)
    else:
        print(f"  相机: {args.cameras}  分辨率: {args.cam_resolution}", flush=True)
        print(f"  输出目录: {lerobot_out}", flush=True)
    print("=" * 62, flush=True)

    if args.num_episodes > 1:
        orca_logger.warning(
            "本脚本不做随机化，多集会得到几乎相同的轨迹；"
            "需要多样性请录多份路点或后续加入随机化。"
        )

    # ── 场景管理 ──────────────────────────────────────────────────────────────
    orca_logger.info("Creating scene manager")
    with open(
        os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8"
    ) as f:
        scene_config = load(f, Loader=Loader)
    if "data_collection" in scene_config:
        scene_config["data_collection"]["agent_joint_prefix"] = f"{args.agent_name}_"
    else:
        scene_config["data_collection"] = {"agent_joint_prefix": f"{args.agent_name}_"}
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = (
        os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    )
    scene_manager.show_ui_message(
        1, "脚本控制：G1 路点回放自动采集", "0xffff00", showtime=5
    )
    scene_manager.get_scene_data(script_name, "beginscene")

    # ── Storage ───────────────────────────────────────────────────────────────
    scratch_dir = os.path.join(
        base_dir, "_lerobot_scratch", "g1_pick_osc_scripted", args.level
    )
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
        keep = mj_joint_strip.KEEP_DEFAULT + tuple(g1_pick_osc_conf.l_arm["joint_names"])
        strip = mj_joint_strip.install(
            None,
            args.agent_name,
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
        device=None,
        scene_manager=scene_manager,
        data_storage=None if dry_run else storage,
        frame_skip=args.frame_skip,
        time_step=args.time_step,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    stripped = bool(strip is not None and strip.applied)
    if stripped:
        alive = set(env.model.get_joint_dict() or {})
        dropped = [j for j in default_joint_values if env.joint(j) not in alive]
        for j in dropped:
            default_joint_values.pop(j)
        orca_logger.info(
            f"[MODEL] 初始状态配置完成（{len(default_joint_values)} 个关节）"
        )

    # ── 场景就绪后初始化控制器 + 相机 ─────────────────────────────────────────
    cameras: dict = {}
    cam_hw = cam_hw_override
    video_started = False
    r_arm = None
    r_grip = None
    task_status = None

    try:
        env.reset()
        time.sleep(0.1)
        if strip is not None:
            mj_joint_strip.finish_install(
                env, strip, args.agent_name,
                log=lambda m: (orca_logger.info(m), print(m, flush=True)),
            )
        if not manager.update_scene():
            orca_logger.error("首次 update_scene 失败，退出")
            env.close()
            return

        env.set_default_joint_values(default_joint_values)
        print(
            "[MODEL] 任务模型配置已加载",
            flush=True,
        )

        # 夹爪控制器
        if stripped:
            orca_logger.info("[MODEL] 当前任务配置不启用左夹爪控制器")
        else:
            l_gname = [
                env.actuator(n) for n in g1_pick_osc_conf.gripper_l["actuator_names"]
            ]
            l_grip = create_gripper_2f85_reverse_controller(
                env,
                g1_pick_osc_conf.gripper_l,
                g1_pick_osc_conf.base_body,
                l_gname,
                {n: v for n, v in zip(l_gname, g1_pick_osc_conf.gripper_l["init_ctrl"])},
                Controller2F85Reverse.ControllerType.DATA,
            )
            l_grip.update_ctrl(np.full(len(l_grip.ctrl_index), g_open, dtype=np.float32))
            manager.add_controller(l_grip)

        orca_logger.info("Adding right gripper controller (DATA)")
        r_gname = [
            env.actuator(n) for n in g1_pick_osc_conf.gripper_r["actuator_names"]
        ]
        r_grip = create_gripper_2f85_reverse_controller(
            env,
            g1_pick_osc_conf.gripper_r,
            g1_pick_osc_conf.base_body,
            r_gname,
            {n: v for n, v in zip(r_gname, g1_pick_osc_conf.gripper_r["init_ctrl"])},
            Controller2F85Reverse.ControllerType.DATA,
        )
        r_grip.update_ctrl(np.full(len(r_grip.ctrl_index), g_open, dtype=np.float32))
        manager.add_controller(r_grip)

        orca_logger.info("Adding right arm OSC controller")
        ctrl_r_name = [
            env.actuator(m) for m in g1_pick_osc_conf.r_arm["motors_names"]
        ]
        r_arm = create_arm_osc_controller(
            env,
            g1_pick_osc_conf.r_arm,
            g1_pick_osc_conf.base_body,
            ctrl_r_name,
            {
                n: v
                for n, v in zip(
                    ctrl_r_name, g1_pick_osc_conf.r_arm["motors_init_ctrl"]
                )
            },
        )
        manager.add_controller(r_arm)

        if args.kp > 0.0:
            kp_val = float(np.clip(args.kp, 1.0, 300.0))
            r_arm.controller.kp = np.ones(6, dtype=np.float64) * kp_val
            r_arm.controller.kd = 2.0 * np.sqrt(r_arm.controller.kp)
            orca_logger.info("[CONTROL] OSC 阻抗参数已加载")

        if stripped:
            orca_logger.info("[MODEL] 任务模型配置已应用，姿态约束初始化完成")
        else:
            orca_logger.info("[CONSTRAINT] 正在初始化任务姿态约束")
            tele.pin_all_joints(env, args.agent_name)

        orca_logger.info("Setting task and task status controller")
        manager.set_task(EmptyTask(env))
        task_status = TaskStatusController(
            env, g1_pick_osc_conf.base_body, is_controller=False
        )
        manager.set_task_status_controller(task_status)

        if _cameras_disabled:
            orca_logger.info("跳过相机推流（--dry_run / 相机已关闭）")
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
                    "MP4 相机模式已启用"
                )
    except KeyboardInterrupt:
        orca_logger.info("初始化阶段收到 Ctrl+C，正在释放相机推流会话...")
    except Exception as e:
        orca_logger.error(f"初始化失败: {e}")

    def _release_and_close():
        if video_started:
            try:
                env.stop_save_video()
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

    if r_arm is None or r_grip is None or task_status is None:
        orca_logger.error("控制器未创建成功，退出")
        _release_and_close()
        return
    if not dry_run and not cameras and args.camera_source != "mp4":
        orca_logger.error("没有可用相机，退出（只想验证动作请加 --dry_run）")
        _release_and_close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])

    # ── 主循环 ────────────────────────────────────────────────────────────────
    writer = None
    n_saved = 0
    try:
        if not dry_run:
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

        orca_logger.info(
            f"开始脚本化采集，共 {args.num_episodes} 集，任务: {args.task}"
        )

        for ep_idx in range(1, args.num_episodes + 1):
            orca_logger.info(f"========== Episode {ep_idx}/{args.num_episodes} ==========")
            print(
                f"\n>>> 正在采集第 {ep_idx}/{args.num_episodes} 集 | 任务: {args.task}",
                flush=True,
            )
            try:
                scene_manager.show_ui_message(
                    1,
                    f"采集中: {args.task}  ({ep_idx}/{args.num_episodes})",
                    "0x00ff88",
                    showtime=0,
                )
            except Exception:
                pass

            env.reset()
            time.sleep(0.1)
            if not manager.update_scene():
                orca_logger.error("update_scene 失败，停止采集")
                break
            env.set_default_joint_values(default_joint_values)

            ep_dir: str | None = None
            ep_start_wall: float | None = None
            if not dry_run and args.camera_source == "mp4":
                ep_dir = os.path.join(scratch_dir, "mp4", f"ep_{ep_idx:06d}")
                os.makedirs(os.path.join(ep_dir, "video"), exist_ok=True)
                ep_start_wall = time.perf_counter()
                env.begin_save_video(ep_dir)
                video_started = True

            # 从当前末端位姿生成右臂执行轨迹
            _, _, r_pos, r_quat, _, r_gm = scripted.build_segmented_trajectory(
                env, g1_pick_osc_conf, all_segments, g_open, g_close
            )

            device = G1OscScriptedDevice(
                env,
                r_arm,
                r_grip,
                task_status,
                r_pos,
                r_quat,
                r_gm,
                waypoint_marks,
                track_log_every=args.track_log_every,
                track_ki=args.track_ki,
                track_clamp=args.track_clamp,
                action_repeat=args.action_repeat,
            )
            manager.set_device(device)

            ep_t0 = time.perf_counter()
            manager.run_episode()
            ep_dur = time.perf_counter() - ep_t0

            if not dry_run and args.camera_source == "mp4" and video_started:
                try:
                    env.stop_save_video()
                except Exception as stop_err:
                    orca_logger.warning("相机数据流停止时遇到错误")
                video_started = False

            if not device.finished:
                orca_logger.warning(
                    f"[EP {ep_idx}] 轨迹提前结束，请检查运行状态"
                )

            if dry_run:
                orca_logger.info(
                    f"[EP {ep_idx}] 轨迹回放完毕，时长 {ep_dur:.1f}s（--dry_run 不保存）"
                )
                print(f">>> 第 {ep_idx} 集轨迹回放完毕（未保存）", flush=True)
                continue

            ep_frames = storage.buffered_frame_count
            storage.save_data(
                task_info=manager.task.get_task_info(),
                scene_info=scene_manager.get_scene_info(),
                task_description=manager.task.get_task_description(),
                episode_video_dir=ep_dir,
                ep_start_wall=ep_start_wall,
            )
            n_saved += 1
            cap_fps = (ep_frames / ep_dur) if ep_dur > 0 else 0.0
            orca_logger.info(
                f"[✓] Episode {ep_idx} 已保存：{ep_frames} 帧 / {ep_dur:.1f}s，"
                f"累计 {writer.num_episodes} 集 / {writer.num_frames} 帧"
            )
            print(
                f">>> [✓] 第 {ep_idx} 集已保存（{ep_frames} 帧），"
                f"累计 {writer.num_episodes} 集",
                flush=True,
            )

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集，丢弃当前未保存集")
        print("\n[停止] 采集已中断，丢弃当前未保存集", flush=True)
        if not dry_run:
            storage.clear_data()
    except Exception as e:
        orca_logger.error(f"采集异常: {e}")
    finally:
        if writer is not None:
            try:
                orca_logger.info("正在等待所有视频编码完成，请勿关闭程序...")
                print("\n[退出] 正在等待所有视频编码完成，请勿关闭程序...", flush=True)
                writer.close()
                orca_logger.info("✓ 所有视频编码已完成")
            except Exception as close_err:
                orca_logger.error(f"数据集收尾失败: {close_err}")
        if dry_run:
            summary = f"轨迹回放结束，共 {args.num_episodes} 集（未保存数据）"
        elif writer is not None:
            summary = (
                f"采集结束，本次保存 {n_saved} 集，"
                f"数据集共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
            )
        else:
            summary = "采集结束（未成功创建数据集）"
        orca_logger.info(summary)
        _release_and_close()
        print(f"\n{'=' * 62}", flush=True)
        print(f"  {summary}", flush=True)
        if lerobot_out is not None:
            print(f"  数据位于: {lerobot_out}", flush=True)
        print(f"{'=' * 62}", flush=True)


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
