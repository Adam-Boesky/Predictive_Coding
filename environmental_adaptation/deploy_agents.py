"""Deploy agents for the experiment."""
import os
import shutil
import numpy as np

from subprocess import Popen


def deploy_agents():
    """Deploy the agents"""

    # Set up results dir
    N_agents = 100
    results_path = '/n/ramanathan_lab/aboesky/jonah_adaptation_fixed2'
    os.environ['RESULTS_DIR'] = results_path
    if os.path.exists(results_path) and os.path.isdir(results_path):
        shutil.rmtree(results_path)
    os.mkdir(results_path)

    # # Set start seeds to try
    # start_seeds = np.linspace(1, 200000, N_agents).astype("int")

    # Submit agents
    ps = []
    # for i, seed in enumerate(start_seeds):
    for i in range(N_agents):
        print(f'Submitting agent {i}')
        os.environ['SEED'] = str(i)

        # Open a pipe to the sbatch command.
        sbatch_command = f'sbatch --wait /n/home04/aboesky/ramanthan/Predictive_Coding/environmental_adaptation/run_agent.sh {results_path} {i}'
        proc = Popen(sbatch_command, shell=True)
        ps.append(proc)

    exit_codes = [p.wait() for p in ps]  # wait for processes to finish
    print(f'exit_codes = {exit_codes}')
    return exit_codes 


if __name__ == '__main__':
    deploy_agents()
