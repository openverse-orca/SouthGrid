"""回放 g1_pick_osc 脚本化采集写出的 LeRobot v2.1 数据集。

环境与 g1_pick_osc_collection_scripted_lerobot.py 一致：
右臂 OSC + 右爪 2F85 + joint_strip。数据源换成 parquet 的 18 维 action，
只驱动右臂 / 右爪；左臂那 9 维丢掉（strip 后左臂已不存在）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
from yaml import Loader, load

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

import mj_joint_strip
import g1_pick_osc_collection_tele_lerobot as tele

from conf import g1_pick_osc_conf
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_task import TaskStatus, TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
    install_osc_patches,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
_L_INIT_JOINT_VALUES = [0.0, 0.127, 0.0, 1.5708, 0.0, 0.0, 0.0]

orca_logger = get_orca_logger(
    name="G1OscReplay",
    log_file="g1_pick_osc_replay.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=os.path.join(base_dir, "logs"),
    use_colors=True,
    force_reinit=True,
)


def _scan_parquet_files(dataset_dir: str) -> list[str]:
    data_dir = os.path.join(dataset_dir, "data")
    files: list[str] = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data 目录不存在: {data_dir}")
    for chunk in sorted(os.listdir(data_dir)):
        chunk_path = os.path.join(data_dir, chunk)
        if not os.path.isdir(chunk_path):
            continue
        for fname in sorted(os.listdir(chunk_path)):
            if fname.startswith("episode_") and fname.endswith(".parquet"):
                files.append(os.path.join(chunk_path, fname))
    return files


def _denorm(norm_col: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(norm_col, 0.0, 1.0) * (hi - lo) + lo


def _load_episode(parquet_path: str) -> dict:
    import pyarrow.parquet as pq

    actions = np.array(pq.read_table(parquet_path)["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 18:
        raise ValueError(f"{parquet_path}: action 形状异常 {actions.shape}，期望 (N, >=18)")
    r_ranges = g1_pick_osc_conf.gripper_r["actuator_ranges"]
    return {
        "r_pos_b": actions[:, 7:10],
        "r_quat_b": actions[:, 10:14],
        "r_grip": np.stack(
            [
                _denorm(actions[:, 16], *r_ranges[0]),
                _denorm(actions[:, 17], *r_ranges[1]),
            ],
            axis=1,
        ),
        "n_frames": len(actions),
    }


def _query_r_ee_b(env) -> tuple[np.ndarray, np.ndarray]:
    site = env.site(g1_pick_osc_conf.r_arm["ee_site_name"])
    base = env.body(g1_pick_osc_conf.base_body)
    res = env.query_site_pos_and_quat_B([site], [base])
    pos = np.asarray(res[site]["xpos"], dtype=np.float32).reshape(3)
    quat = np.asarray(res[site]["xquat"], dtype=np.float32)[[1, 2, 3, 0]]
    return pos, quat


class G1OscParquetReplayDevice(AbstractDevice):
    """把 parquet 右臂目标逐步写进 OSC / 右爪。"""

    def __init__(
        self,
        env,
        r_arm,
        r_grip: Controller2F85Reverse,
        task_status: TaskStatusController,
        data: dict,
        steps_per_frame: int,
        track_ki: float = 0.0,
        track_clamp: float = 0.08,
        track_log_every: int = 0,
    ):
        super().__init__()
        self.env = env
        self.r_arm = r_arm
        self.r_grip = r_grip
        self.task_status = task_status
        self.r_pos = np.asarray(data["r_pos_b"], dtype=np.float64)
        self.r_quat = np.asarray(data["r_quat_b"], dtype=np.float32)
        self.r_grip_motor = np.asarray(data["r_grip"], dtype=np.float32)
        self.n = int(data["n_frames"])
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.track_ki = max(0.0, float(track_ki))
        self.track_clamp = abs(float(track_clamp))
        self.track_log_every = max(0, int(track_log_every))
        self._icorr = np.zeros(3, dtype=np.float64)
        self._ee_site = env.site(g1_pick_osc_conf.r_arm["ee_site_name"])
        self._base_body = env.body(g1_pick_osc_conf.base_body)
        self._call_count = 0
        self._frame_idx = -1

    @property
    def finished(self) -> bool:
        return self._call_count >= self.n * self.steps_per_frame

    def _tracking_err(self, target) -> tuple[float, np.ndarray]:
        res = self.env.query_site_pos_and_quat_B([self._ee_site], [self._base_body])
        cur = np.asarray(res[self._ee_site]["xpos"], dtype=np.float64).reshape(3)
        d = (cur - np.asarray(target, dtype=np.float64)) * 1000.0
        return float(np.linalg.norm(d)), d

    def update(self):
        call = self._call_count
        self._call_count += 1
        frame = call // self.steps_per_frame
        if frame >= self.n:
            if self.task_status.current_status == TaskStatus.RUNNING:
                self.task_status.update_task_status(True)
            return
        if call == 0:
            self.task_status.update_task_status(True)

        first_hold = frame != self._frame_idx
        if first_hold:
            self._frame_idx = frame
            if self.track_log_every and frame % self.track_log_every == 0:
                norm, d = self._tracking_err(self.r_pos[frame])
                orca_logger.info(
                    f"[跟踪] frame {frame}/{self.n} 残差 {norm:.0f}mm "
                    f"(dx={d[0]:+.0f} dy={d[1]:+.0f} dz={d[2]:+.0f})"
                )

        target = np.asarray(self.r_pos[frame], dtype=np.float64)
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
        self.r_arm.update_action_axisangle(self.r_quat[frame])
        self.r_grip.update_ctrl(self.r_grip_motor[frame])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="回放 g1_pick_osc 脚本化采集的 LeRobot parquet（只开右臂 OSC）"
    )
    parser.add_argument("--dataset_dir", required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--task_config", default="example.yaml")
    parser.add_argument("--episode", type=int, default=None, help="只回放第 N 集（1 起）")
    parser.add_argument("--loop", action="store_true", help="全部播完后从头循环")
    parser.add_argument(
        "--steps_per_frame",
        type=int,
        default=10,
        help="每个 parquet 帧重复的控制步数（fps=20、dt=5ms 时为 10）",
    )
    parser.add_argument("--settle_steps", type=int, default=10, help="开播前驻留步数")
    parser.add_argument("--agent_name", default="g1_pick")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--dls_lambda", type=float, default=0.23)
    parser.add_argument("--dls_sigma_th", type=float, default=0.12)
    parser.add_argument("--null_kp", type=float, default=10.0)
    parser.add_argument(
        "--kp", type=float, default=0.0,
        help="OSC kp（默认 0 = 沿用 osc_pose；>0 时覆盖并设 kd=2√kp）",
    )
    parser.add_argument("--track_ki", type=float, default=0.0)
    parser.add_argument("--track_clamp", type=float, default=0.08)
    parser.add_argument("--track_log_every", type=int, default=20)
    parser.add_argument("--joint_strip", choices=["off", "on"], default="on")
    parser.add_argument("--strip_col", choices=["off", "keep"], default="off")
    parser.add_argument("--time_step", type=float, default=0.001)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--render_every", type=int, default=5)
    args = parser.parse_args()

    if args.steps_per_frame < 1:
        parser.error("--steps_per_frame 必须 >= 1")
    if args.frame_skip < 1 or args.time_step <= 0:
        parser.error("--time_step / --frame_skip 非法")

    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    all_files = _scan_parquet_files(dataset_dir)
    if not all_files:
        raise FileNotFoundError(f"未找到 parquet: {dataset_dir}/data/")
    if args.episode is not None:
        idx = args.episode - 1
        if not (0 <= idx < len(all_files)):
            raise ValueError(f"--episode {args.episode} 超出范围 1–{len(all_files)}")
        playlist = [all_files[idx]]
    else:
        playlist = list(all_files)

    env_dt = float(args.time_step) * int(args.frame_skip)
    install_osc_patches(
        dls_lambda=args.dls_lambda,
        dls_sigma_th=args.dls_sigma_th,
        null_kp=args.null_kp,
    )

    default_joint_values: dict = {}
    for jn, v in zip(g1_pick_osc_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(g1_pick_osc_conf.r_arm["joint_names"], tele._R_INIT_JOINT_VALUES):
        default_joint_values[jn] = v

    print("=" * 62, flush=True)
    print("  G1 Pick OSC LeRobot 回放", flush=True)
    print(f"  数据集: {dataset_dir}", flush=True)
    print(f"  共 {len(all_files)} 集，待播 {len(playlist)} 集", flush=True)
    print(
        f"  steps_per_frame={args.steps_per_frame}  dt={env_dt * 1000:.0f}ms  "
        f"DLS λ={args.dls_lambda}  kp={args.kp}  ki={args.track_ki}",
        flush=True,
    )
    print("=" * 62, flush=True)

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    if "data_collection" in scene_config:
        scene_config["data_collection"]["agent_joint_prefix"] = f"{args.agent_name}_"
    else:
        scene_config["data_collection"] = {"agent_joint_prefix": f"{args.agent_name}_"}
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    scene_manager.show_ui_message(1, "脚本控制：G1 OSC parquet 回放", "0xffff00", showtime=5)
    scene_manager.get_scene_data(os.path.basename(__file__), "beginscene")

    strip = None
    if args.joint_strip == "on":
        keep = mj_joint_strip.KEEP_DEFAULT + tuple(g1_pick_osc_conf.l_arm["joint_names"])
        strip = mj_joint_strip.install(
            None,
            args.agent_name,
            keep=keep,
            kill_collision=(args.strip_col == "off"),
            required_cameras=("cam_head",),
            log=lambda m: (orca_logger.info(m), print(m, flush=True)),
        )

    def _obs_callback(env):
        return {"replay": np.zeros(max(env.nu, 1), dtype=np.float32)}

    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=args.frame_skip,
        time_step=args.time_step,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False
    manager.mode = manager.DataCollectionMode.INFERENCE
    manager.inference_ui_message = "数据恢复中..."

    stripped = bool(strip is not None and strip.applied)
    if stripped:
        alive = set(env.model.get_joint_dict() or {})
        for j in [j for j in default_joint_values if env.joint(j) not in alive]:
            default_joint_values.pop(j)

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
            f"[STRIP] {'生效' if stripped else '未启用'} "
            f"nq={env.model.nq} nv={env.model.nv} nu={env.model.nu}"
            f"  dt={env_dt * 1000:.0f}ms",
            flush=True,
        )

        r_gname = [env.actuator(n) for n in g1_pick_osc_conf.gripper_r["actuator_names"]]
        r_grip = create_gripper_2f85_reverse_controller(
            env,
            g1_pick_osc_conf.gripper_r,
            g1_pick_osc_conf.base_body,
            r_gname,
            {n: v for n, v in zip(r_gname, g1_pick_osc_conf.gripper_r["init_ctrl"])},
            Controller2F85Reverse.ControllerType.DATA,
        )
        manager.add_controller(r_grip)

        ctrl_r_name = [env.actuator(m) for m in g1_pick_osc_conf.r_arm["motors_names"]]
        r_arm = create_arm_osc_controller(
            env,
            g1_pick_osc_conf.r_arm,
            g1_pick_osc_conf.base_body,
            ctrl_r_name,
            {n: v for n, v in zip(ctrl_r_name, g1_pick_osc_conf.r_arm["motors_init_ctrl"])},
        )
        manager.add_controller(r_arm)
        if args.kp > 0.0:
            kp_val = float(np.clip(args.kp, 1.0, 300.0))
            r_arm.controller.kp = np.ones(6, dtype=np.float64) * kp_val
            r_arm.controller.kd = 2.0 * np.sqrt(r_arm.controller.kp)
            orca_logger.info(f"OSC 阻抗刚度 kp 设为 {kp_val}（kd=2√kp 临界阻尼）")

        if stripped:
            orca_logger.info("[STRIP] base/腰/下肢/左爪已剥离焊死，左臂保留自由演化，跳过 pin_all_joints")
        else:
            tele.pin_all_joints(env, args.agent_name)

        manager.set_task(EmptyTask(env))
        task_status = TaskStatusController(
            env, g1_pick_osc_conf.base_body, is_controller=False
        )
        manager.set_task_status_controller(task_status)
    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")

    if r_arm is None or r_grip is None or task_status is None:
        orca_logger.error("控制器未创建成功，退出")
        try:
            env.close()
        except Exception:
            pass
        return

    _render_counter = [0]
    _render_every = max(0, int(args.render_every))
    _orig_render = env.render

    def _patched_render():
        if _render_every == 0:
            return None
        _render_counter[0] += 1
        if _render_counter[0] % _render_every == 0:
            return _orig_render()
        return None

    env.render = _patched_render

    try:
        ep_files = list(playlist)
        ep_total = len(ep_files)
        ep_idx = 0
        while True:
            if manager._shutdown_requested:
                break
            if ep_idx >= len(ep_files):
                if args.loop:
                    ep_files = list(playlist)
                    ep_idx = 0
                    orca_logger.info("全部播完，循环重头")
                else:
                    orca_logger.info("全部播完，退出")
                    if manager.scene_manager is not None:
                        manager.scene_manager.show_ui_message(
                            1, "回放完毕", "0x00ff00", showtime=0
                        )
                        env.render()
                        time.sleep(1.5)
                    break

            parquet_path = ep_files[ep_idx]
            ep_idx += 1
            ep_name = os.path.basename(parquet_path)
            ep_data = _load_episode(parquet_path)
            n_frames = ep_data["n_frames"]
            spf = args.steps_per_frame
            orca_logger.info(
                f"=== 回放 {ep_name} ({ep_idx}/{ep_total})  "
                f"{n_frames} 帧 × {spf} = {n_frames * spf} 控制步 ==="
            )

            env.reset()
            time.sleep(0.05)
            if not manager.update_scene():
                orca_logger.error("update_scene 失败，停止")
                break
            env.set_default_joint_values(default_joint_values)
            env.mj_forward()
            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            for controller in manager.controllers:
                controller.reset()
            env.render()
            time.sleep(0.05)

            r_pos, r_quat = _query_r_ee_b(env)
            orca_logger.info(f"  [reset] R_ee={np.round(r_pos, 4).tolist()}")
            r_arm.update_action_position(r_pos)
            r_arm.update_action_axisangle(r_quat)
            r_grip.update_ctrl(np.asarray(ep_data["r_grip"][0], dtype=np.float32))
            for _ in range(max(0, int(args.settle_steps))):
                env.step(manager.run_controllers())
                env.render()

            r_pos, _ = _query_r_ee_b(env)
            orca_logger.info(f"  [init] 开播前 R_ee={np.round(r_pos, 4).tolist()}")

            device = G1OscParquetReplayDevice(
                env,
                r_arm,
                r_grip,
                task_status,
                ep_data,
                steps_per_frame=spf,
                track_ki=args.track_ki,
                track_clamp=args.track_clamp,
                track_log_every=args.track_log_every,
            )
            manager.set_device(device)
            t0 = time.perf_counter()
            manager.run_episode()
            orca_logger.info(
                f"  播完 {ep_name}：{device._call_count} 控制步 / "
                f"{time.perf_counter() - t0:.1f}s"
                f"{'' if device.finished else '（未跑完）'}"
            )
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止回放")
    except Exception as e:
        orca_logger.error(f"回放异常: {e}\n{traceback.format_exc()}")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        OrcaLog.get_instance().error(f"Error: {e}\n{traceback.format_exc()}")
        raise
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)
