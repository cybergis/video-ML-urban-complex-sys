#!/bin/bash

#SBATCH --job-name=pointprocess
#SBATCH -n 2
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH -p gpu
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres=gpu:1

module purge
module add GPU
module load gnu/gnu-9.3.0

cmd="python run.py"

# Load the pyton environment
echo "Loading conda environment"
source /data/keeling/a/dkiv2/dkiv2/miniconda3/bin/activate 
conda activate old

cmds=(
    # "python run.py --dataset-name simulated --model-name e3d -lr 5e-3 --experiment-name testsim"
    "python run.py --dataset-name real --model-name convlstm -lr 5e-3 --experiment-name vis"
    "python run.py --dataset-name real --model-name e3d -lr 5e-3 --experiment-name vis"
    # "python run.py --dataset-name simulated --model-name convlstm -lr 5e-3 --experiment-name tests"
    # "python run.py --dataset-name simulated --model-name convlstm -lr 5e-3 --experiment-name testsim"
    "python run.py --dataset-name real --model-name predrnn -lr 5e-3 --experiment-name vis"
    # "python run.py --dataset-name simulated --model-name predrnn -lr 5e-3 --experiment-name testsim"
    "python run.py --dataset-name real --model-name predrnn2 -lr 5e-3 --experiment-name vis"
    # "python run.py --dataset-name simulated --model-name predrnn2 -lr 5e-3 --experiment-name testsim"
)

for ((i = 0; i < ${#cmds[@]}; i++))
do
    # Print the command and run it
    echo "Running command: ${cmds[$i]}"
    ${cmds[$i]}
done

echo "Experimental run complete. Script closing."

# End the script
exit