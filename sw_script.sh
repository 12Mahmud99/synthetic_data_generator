#!/bin/bash

# Usage:
# ./sw_script.sh total_time time_step x0_mean zeta k x_mu T boltzmann num_simulations device
# Example:
# ./sw_script.sh 10 0.01 0 0.5 1 0 300 1.38e-23 5 cpu

if [ "$#" -ne 10 ]; then
    echo "Usage: $0 total_time time_step x0_mean zeta k x_mu T boltzmann num_simulations device"
    exit 1
fi

TOTAL_TIME=$1
TIME_STEP=$2
X0_MEAN=$3
ZETA=$4
K=$5
X_MU=$6
T=$7
BOLTZMANN=$8
NUM_SIM=$9
DEVICE=${10}

TMP_PY=$(mktemp /tmp/sw_gen.XXXX.py)

cat <<EOF > $TMP_PY
import torch
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.append(os.getcwd())  # add current directory
from generators import single_well_generator  

total_time = float("$TOTAL_TIME")
time_step = float("$TIME_STEP")
x0_mean = float("$X0_MEAN")
zeta = float("$ZETA")
k = float("$K")
x_mu = float("$X_MU")
T = float("$T")
boltzmann = float("$BOLTZMANN")
num_simulations = int("$NUM_SIM")
device = "$DEVICE"

positions, times = single_well_generator(
    total_time=total_time,
    time_step=time_step,
    x0_mean=x0_mean,
    zeta=zeta,
    k=k,
    x_mu=x_mu,
    T=T,
    boltzmann=boltzmann,
    num_of_simulations=num_simulations,
    device=device
)

for i in range(num_simulations):
    plt.plot(times.cpu().numpy(), positions[i].cpu().numpy(), label=f"sim {i+1}")

plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Single Well Langevin (Overdamped)")
plt.legend()
plt.show()

positions_np = positions.cpu().numpy()
times_np = times.cpu().numpy() 
csv_filename = "single_well_syn_data.csv"
df = pd.DataFrame(positions_np, columns=[f"{t:.4f}" for t in times_np])
df.to_csv(csv_filename, index=False)
EOF

python3 $TMP_PY 
rm $TMP_PY