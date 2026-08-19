#!/usr/bin/env bash
# 路点回放自动采集：不接 Pico，回放 record_waypoints.py 录的 YAML。
# 请先在 OrcaLab 打开本目录 uni_osc.json 并运行仿真（localhost:50051）。
# 用法: bash run_scripted.sh [输出目录] [集数]
# 只想先确认动作对不对（不开相机、不写盘）: bash run_scripted.sh --dry
set -euo pipefail
cd "$(dirname "$0")"

WAYPOINTS="${WAYPOINTS:-waypoint_tool/my_waypoint_tool1.yaml}"

COMMON=(
    --task_config example.yaml
    --agent_name g1_pick_southgrid_usda_1
    --waypoint_files "$WAYPOINTS"
    --task "按红色按钮"
    --dls_lambda 0.23
    --joint_strip on --strip_col off
    --time_step 0.001 --frame_skip 5
)

if [[ "${1:-}" == "--dry" ]]; then
    OMP_NUM_THREADS=1 python g1_pick_osc_collection_scripted_lerobot_strip.py \
        "${COMMON[@]}" --dry_run
else
    OMP_NUM_THREADS=1 python g1_pick_osc_collection_scripted_lerobot_strip.py \
        "${COMMON[@]}" \
        --lerobot_out "${1:-$HOME/southgrid_datasets/g1_osc_scripted}" \
        --repo_id local/g1_pick_osc_scripted \
        --num_episodes "${2:-1}" \
        --fps 20 --clock sim \
        --cameras head,wrist_r --camera_source websocket
fi
