#!/bin/bash
#
#SBATCH -p test
#SBATCH -n 1
#SBATCH --mem=184G
#SBATCH -t 0-00:15 # time (D-HH:MM)
#
##SBATCH --account=ramanathan_lab
#SBATCH -o myoutput_\%j.out
#SBATCH -e myerrors_\%j.err

module load python/3.10.9-fasrc01
source activate predictive_coding

cd /n/home04/aboesky/ramanthan/Predictive_Coding/moving_peak

python3 generate_contours.py
