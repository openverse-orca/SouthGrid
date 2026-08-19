from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime

import numpy as np
from yaml import Loader, load

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mj_joint_strip
import g1_pick_osc_collection_tele_lerobot_strip as tele

from conf import g1_pick_osc_conf
from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
_GRIP_CLOSE_THRESHOLD = 0.5

orca_logger = get_orca_logger(
    name="RecordWaypoints",
    log_file="record_waypoints.log",
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def _fmt(lst, n=4):
    return "[" + ", ".join(f"{v:.{n}f}" for v in lst) + "]"


def query_r_ee_pose(env):
    ee_site = env.site(g1_pick_osc_conf.r_arm["ee_site_name"])
    base_body = env.body(g1_pick_osc_conf.base_body)
    result = env.query_site_pos_and_quat_B([ee_site], [base_body])
    pos = result[ee_site]["xpos"].tolist()
    quat_xyzw = result[ee_site]["xquat"][[1, 2, 3, 0]].tolist()
    grip_norm = 0.0
    try:
        name = env.actuator(g1_pick_osc_conf.gripper_r["actuator_names"][0])
        aid = env.model.actuator_name2id(name)
        lo, hi = g1_pick_osc_conf.gripper_r["actuator_ranges"][0]
        val = float(env.ctrl[aid])
        grip_norm = float(np.clip((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    except Exception:
        pass
    return pos, quat_xyzw, grip_norm


def write_yaml(waypoints, output_path):
    lo, hi = g1_pick_osc_conf.gripper_r["actuator_ranges"][0]
    lines = [
        f"gripper_open: {lo}",
        f"gripper_close: {hi}",
        "",
        "segments:",
        "",
    ]
    for wp in waypoints:
        grip = "close" if wp["grip_norm"] > _GRIP_CLOSE_THRESHOLD else "open"
        lines += [
            "  - steps: 300",
            "    l_hold: true",
            f"    r_target_b: {_fmt(wp['r_pos_b'])}",
            f"    r_quat_b: {_fmt(wp['r_quat_b'])}",
            f"    gripper_r: {grip}",
            "",
        ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[完成] 已写出 {len(waypoints)} 个路点 → {output_path}", flush=True)


def _obs_callback(env):
    n = len(g1_pick_osc_conf.gripper_l["actuator_names"]) + len(
        g1_pick_osc_conf.gripper_r["actuator_names"]
    )
    return {
        "/action/end/position": np.zeros((2, 3), dtype=np.float32),
        "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
        "/action/effector/motor": np.zeros(n, dtype=np.float32),
        "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
    }


def _print_help(output_path):
    print("", flush=True)
    print("=" * 60, flush=True)
    print("  ✓ 场景加载完成，进入路点采集", flush=True)
    print(f"  输出文件: {output_path}", flush=True)
    print("-" * 60, flush=True)
    print("  【操作说明】", flush=True)
    print("  右臂移动    右手柄位姿", flush=True)
    print("  右夹爪      A/B 键 或 右扳机", flush=True)
    print("  左臂        已锁定，不响应手柄", flush=True)
    print("  ★ 记录路点  左右 Grip 同时按下", flush=True)
    print("  ↺ 丢弃重录  单按右 Squeeze：丢掉上一个路点（没有就提示），并重置场景", flush=True)
    print("  退出        Ctrl+C（自动保存 YAML）", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config", default="example.yaml")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--agent_name", default="g1_pick_southgrid_usda_1")
    parser.add_argument("--output", default="waypoints_output.yaml")
    parser.add_argument("--debounce", type=float, default=0.5)
    parser.add_argument("--lerobot_out", default=None)
    parser.add_argument("--repo_id", default="local/g1_pick_osc_strip")
    parser.add_argument("--task", default="按红色按钮")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--clock", choices=("sim", "wall"), default="wall")
    parser.add_argument("--cameras", default="head,wrist_r")
    parser.add_argument("--camera_source", choices=("websocket", "mp4"), default="websocket")
    parser.add_argument("--dls_lambda", type=float, default=0.2)
    parser.add_argument("--dls_sigma_th", type=float, default=0.12)
    parser.add_argument("--null_kp", type=float, default=10.0)
    parser.add_argument("--joint_strip", choices=["off", "on"], default="on")
    parser.add_argument("--strip_col", choices=["off", "keep"], default="off")
    parser.add_argument("--time_step", type=float, default=0.001)
    parser.add_argument("--frame_skip", type=int, default=5)
    args = parser.parse_args()

    output_path = (
        args.output if os.path.isabs(args.output) else os.path.join(base_dir, args.output)
    )

    from controllers.controllers import install_osc_patches
    install_osc_patches(
        dls_lambda=args.dls_lambda,
        dls_sigma_th=args.dls_sigma_th,
        null_kp=args.null_kp,
    )

    default_joint_values = {}
    for jn, v in zip(g1_pick_osc_conf.l_arm["joint_names"], tele._L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(g1_pick_osc_conf.r_arm["joint_names"], tele._R_INIT_JOINT_VALUES):
        default_joint_values[jn] = v

    _record_event = threading.Event()
    _undo_event = threading.Event()
    _shutdown = threading.Event()
    _waypoints: list[dict] = []

    print("=" * 60, flush=True)
    print("  G1 路点采集启动中...", flush=True)
    print(f"  输出文件: {output_path}", flush=True)
    print("  等待 Pico 连接...", flush=True)
    print("  请确认: adb reverse tcp:8001 tcp:8001", flush=True)
    print("=" * 60, flush=True)
    pico_device = PicoJoystickDevice(PicoJoystick())

    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    if "data_collection" in scene_config:
        scene_config["data_collection"]["agent_joint_prefix"] = f"{args.agent_name}_"
    else:
        scene_config["data_collection"] = {"agent_joint_prefix": f"{args.agent_name}_"}
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    scene_manager.get_scene_data(os.path.basename(__file__), "beginscene")

    strip = None
    if args.joint_strip == "on":
        bake_qpos = {
            f"{args.agent_name}_{jn}": float(v)
            for jn, v in zip(g1_pick_osc_conf.l_arm["joint_names"], tele._L_INIT_JOINT_VALUES)
        }
        strip = mj_joint_strip.install(
            None,
            args.agent_name,
            kill_collision=(args.strip_col == "off"),
            required_cameras=("cam_head",),
            bake_qpos=bake_qpos,
            log=print,
        )

    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback,
        env_index=0,
        device=pico_device,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=args.frame_skip,
        time_step=args.time_step,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    stripped = bool(strip is not None and strip.applied)
    if stripped:
        alive = set(env.model.get_joint_dict() or {})
        for j in [k for k in default_joint_values if env.joint(k) not in alive]:
            default_joint_values.pop(j)

    env.reset()
    time.sleep(0.1)
    if strip is not None:
        mj_joint_strip.finish_install(env, strip, args.agent_name, log=print)
    if not manager.update_scene():
        print("update_scene failed", flush=True)
        env.close()
        return
    env.set_default_joint_values(default_joint_values)

    if not stripped:
        controllers.add_gripper_2f85_reverse_pico_controller(
            manager,
            env,
            g1_pick_osc_conf.gripper_l,
            g1_pick_osc_conf.base_body,
            pico_device,
            [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
        )
    controllers.add_gripper_2f85_reverse_pico_controller(
        manager,
        env,
        g1_pick_osc_conf.gripper_r,
        g1_pick_osc_conf.base_body,
        pico_device,
        [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
    )
    controllers.add_arm_osc_pico_controller(
        manager,
        env,
        g1_pick_osc_conf.r_arm,
        g1_pick_osc_conf.base_body,
        pico_device,
        PicoJoystickKey.R_TRANSFORM,
    )
    if not stripped:
        tele.pin_all_joints(env, args.agent_name)

    manager.set_task(EmptyTask(env))
    env.set_default_joint_values(default_joint_values)
    manager.set_init_ctrl()
    env.mj_forward()
    for _ in range(50):
        env.step(manager.run_controllers())
    for c in manager.controllers:
        c.reset()

    locked = {PicoJoystickKey.L_TRANSFORM}
    keys = [k for k in pico_device.keys if k not in locked]
    pico_device.update = lambda: pico_device.pico_joystick.update(keys)

    _monitor_stop = threading.Event()
    debounce = args.debounce

    def _monitor():
        both_prev = False
        last_rec = 0.0
        last_undo = 0.0
        last_status = time.perf_counter()
        r_only_n = 0
        r_used = False
        first_connect = False
        while not _monitor_stop.wait(0.02):
            try:
                pj = pico_device.pico_joystick
                n_clients = len(pj.clients)
                raw = pj.current_key_state
                now = time.perf_counter()
                if n_clients > 0 and not first_connect:
                    first_connect = True
                    orca_logger.info("[Pico] 手柄已连接，可以开始录点")
                    print(
                        "\n  ✓ Pico 手柄已连接！移动右手柄遥操右臂，"
                        "双 Grip 记录路点，右 Squeeze 丢弃上一点。",
                        flush=True,
                    )
                    try:
                        scene_manager.show_ui_message(
                            1, "已连接手柄，可以开始录点", "0x00ff00", showtime=5
                        )
                    except Exception:
                        pass
                if now - last_status >= 2.0:
                    last_status = now
                    if n_clients == 0:
                        orca_logger.info("[Pico] 无客户端连接（请确认 Pico 端 App 已启动）")
                    else:
                        orca_logger.info(f"[Pico] {n_clients} 个客户端已连接")
                if n_clients == 0 or raw is None:
                    both_prev = False
                    r_only_n = 0
                    r_used = False
                    continue
                l = bool((raw.get("leftHand") or {}).get("gripButtonPressed", False))
                r = bool((raw.get("rightHand") or {}).get("gripButtonPressed", False))
                both = l and r
                if both and not both_prev and (now - last_rec) >= debounce:
                    _record_event.set()
                    last_rec = now
                    r_only_n = 0
                    r_used = True
                if r and not l:
                    if not r_used:
                        r_only_n += 1
                        if r_only_n >= 8 and (now - last_undo) >= debounce:
                            _undo_event.set()
                            last_undo = now
                            r_used = True
                else:
                    r_only_n = 0
                    if not r:
                        r_used = False
                both_prev = both
            except Exception:
                pass

    threading.Thread(target=_monitor, daemon=True).start()
    signal.signal(signal.SIGINT, lambda *_: _shutdown.set())
    _print_help(output_path)

    try:
        while not _shutdown.is_set():
            env.step(manager.run_controllers())
            env.render()
            if _undo_event.is_set():
                _undo_event.clear()
                if _waypoints:
                    dropped = _waypoints.pop()
                    grip = (
                        "close"
                        if dropped["grip_norm"] > _GRIP_CLOSE_THRESHOLD
                        else "open"
                    )
                    print(
                        f"\n  ↺ 已丢弃路点 {len(_waypoints) + 1} "
                        f"({grip})。当前 {len(_waypoints)} 点。",
                        flush=True,
                    )
                    ui_msg = f"已丢弃，当前 {len(_waypoints)} 点"
                else:
                    print("\n  ↺ 没有可丢弃的路点。", flush=True)
                    ui_msg = "没有可丢弃的路点，重置场景"
                print("  ↺ 重置场景...", flush=True)
                try:
                    env.reset()
                    time.sleep(0.1)
                    manager.update_scene()
                    env.set_default_joint_values(default_joint_values)
                    manager.set_init_ctrl()
                    for _ in range(50):
                        env.step(manager.run_controllers())
                    print("  ↺ 场景重置完成。", flush=True)
                    try:
                        scene_manager.show_ui_message(1, ui_msg, "0xffaa00", showtime=2)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"  ⚠ 重置场景失败: {e}", flush=True)
            if _record_event.is_set():
                _record_event.clear()
                pos, quat, gnorm = query_r_ee_pose(env)
                _waypoints.append(
                    {
                        "r_pos_b": pos,
                        "r_quat_b": quat,
                        "grip_norm": gnorm,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                )
                grip = "close" if gnorm > _GRIP_CLOSE_THRESHOLD else "open"
                idx = len(_waypoints)
                print(f"\n  ★ 路点 {idx} 已记录", flush=True)
                print(f"    r_target_b: {_fmt(pos)}", flush=True)
                print(f"    r_quat_b:   {_fmt(quat)}", flush=True)
                print(f"    gripper_r:  {grip}", flush=True)
                try:
                    scene_manager.show_ui_message(
                        1, f"路点 {idx} → {grip}", "0x00ffff", showtime=2
                    )
                except Exception:
                    pass
    except KeyboardInterrupt:
        _shutdown.set()
    finally:
        _monitor_stop.set()
        try:
            env.close()
        except Exception:
            pass
        if _waypoints:
            write_yaml(_waypoints, output_path)
        else:
            print("\n[退出] 未记录任何路点，不写出文件。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"{e}\n{traceback.format_exc()}", flush=True)
    finally:
        os._exit(0)
