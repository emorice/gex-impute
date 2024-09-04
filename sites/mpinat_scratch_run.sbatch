#!/bin/bash
#SBATCH -J r2s_blood_0
#SBATCH -p medium
#SBATCH -A scc_users
#SBATCH -c 32
#SBATCH -t 48:00:00

set -e

eval `ssh-agent`
ssh-add

module load python/3.10

VENV='./venv'

if ! [ -d $VENV ]; then
	python3 -m venv $VENV
fi
. $VENV/bin/activate
pip install -U pip
pip install wheel
pip install -e .

python -m galp.client -k -j 32 -o 'logfile+http' -s store gex_impute.pipeline run_r2s_blood_0
