# Experiment Management (optional for Slurm)

This folder contains scripts that might be helpful if you are running the Experiments of Section 4.1 on Slurm, which is a workload manager used by many large computing clusters

## deploy_few_shot_universial_script.py

Creates and runs a slurm job with the specified parameters for experiments of Section 4.1.

## restart_from_cp.py

Continues the experiment from an optuna checkpoint if the Slurm job ran out of time or failed otherwise.