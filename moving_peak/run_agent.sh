#!/bin/bash
#
#SBATCH -p shared
#SBATCH -n 1
#SBATCH --mem-per-cpu=5G
#SBATCH -t 0-00:40 # time (D-HH:MM)
#SBATCH -o /n/home04/aboesky/ramanthan/Predictive_Coding/moving_peak/logs/myoutput_\%j.out
#SBATCH -e /n/home04/aboesky/ramanthan/Predictive_Coding/moving_peak/logs/myerrors_\%j.err

cd /n/home04/aboesky/ramanthan/Predictive_Coding/moving_peak

python3 run_agent.py $RESULTS_DIR $AGENT_I
