#!/bin/bash
# H-HASAC environment setup
set -e

# 1. Init HARL submodule
git submodule update --init --recursive

# 2. Install HARL
pip install -e HARL/

# 3. Apply patches (replace modified files in-place)
HARL=HARL/harl
cp harl_patches/off_policy_ha_runner.py       $HARL/runners/off_policy_ha_runner.py
cp harl_patches/soft_twin_continuous_q_critic.py $HARL/algorithms/critics/soft_twin_continuous_q_critic.py
cp harl_patches/envs_tools.py                 $HARL/utils/envs_tools.py
cp harl_patches/configs_tools.py              $HARL/utils/configs_tools.py
cp harl_patches/off_policy_buffer_ep.py       $HARL/common/buffers/off_policy_buffer_ep.py
cp configs/fiveg.yaml                         $HARL/configs/envs_cfgs/fiveg.yaml

echo "✅ Setup complete. Run: python scripts/train_h_hasac.py"
