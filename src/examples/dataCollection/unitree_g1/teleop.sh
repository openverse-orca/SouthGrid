#! /bin/bash
# 关节剥离采集（当前推荐）：见 joint_strip/README.md
#   cd src/examples/dataCollection/unitree_g1/joint_strip && bash run_strip.sh ~/southgrid_datasets/g1_osc

cd "$(dirname "$0")"

# 原版遥操（不剥离）
python g1_pick_osc_collection_tele_lerobot.py \
    --agent_name g1_pick_southgrid_usda_1 \
    --teleop_only
