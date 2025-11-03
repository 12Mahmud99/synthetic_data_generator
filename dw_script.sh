#!/bin/bash

# Usage:
# ./dw_script.sh total_time time_step x0_mean zeta left_well right_well barrier_height T boltzmann num_simulations device
# Example:
# ./dw_script.sh 10 0.01 0 0.5 -1 1 5 300 1.38e-23 5 cpu

if [ "$#" -ne 11 ]; then
    echo "Usage: $0 total_time time_step x0_mean zeta left_well right_well barrier_height T boltzmann num_simulations device"
    exit 1
fi

TOTAL_TIME=$1
TIME_STEP=$2
X0_MEAN=$3
ZETA=$4
LEFT_WELL=$5
RIGHT_WELL=$6
BARRIER_HEIGHT=$7
T=$8
BOLTZMANN=$9
NUM_SIM=${10}
DEVICE=${11}

TMP_PY=$(mktemp /tmp/dw_gen.XXXX.py)

cat <<EOF > $TMP_PY
import sys
import os
sys.path.append(os.getcwd())

import torch
import matplotlib.pyplot as plt
import pandas as pd
from generators import double_wells_generator

total_time = float("$TOTAL_TIME")
time_step = float("$TIME_STEP")
x0_mean = float("$X0_MEAN")
zeta = float("$ZETA")
left_well = float("$LEFT_WELL")
right_well = float("$RIGHT_WELL")
barrier_height = float("$BARRIER_HEIGHT")
T = float("$T")
boltzmann = float("$BOLTZMANN")
num_simulations = int("$NUM_SIM")
device = "$DEVICE"

positions, times = double_wells_generator(
    total_time=total_time,
    time_step=time_step,
    x0_mean=x0_mean,
    zeta=zeta,
    left_well=left_well,
    right_well=right_well,
    barrier_height=barrier_height,
    T=T,
    boltzmann=boltzmann,
    num_of_simulations=num_simulations,
    device=device
)

# Plot
for i in range(num_simulations):
    plt.plot(times.cpu().numpy(), positions[i].cpu().numpy(), label=f"sim {i+1}")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Double Well Langevin (Overdamped)")
plt.legend()
plt.show()

# Save CSV with time columns
positions_np = positions.cpu().numpy()
times_np = times.cpu().numpy()
csv_filename = "double_well_trajectories.csv"
df = pd.DataFrame(positions_np, columns=[f"{t:.4f}" for t in times_np])
df.to_csv(csv_filename, index=False)
print(f"Saved trajectories to {csv_filename}")
EOF

python3 $TMP_PY
rm $TMP_PY
