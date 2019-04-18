#!/bin/bash
#SBATCH --job-name=GPUjob     ### Job Name
#SBATCH --mail-type=ALL      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=2014csb1029@iitrpr.ac.in  # Where to send mail
#SBATCH --partition=gpu       ### Quality of Service (like a queue in PBS)
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --output=/NAS/home01/<asc1 OR asc2>/LogFile_if_code_fails.out # Standard output and error log

source activate pytorch

nvidia-smi > amazing_code1_console_output.txt
python /NAS/home01/<asc1 OR asc2>/your/code/directory/amazing_code1.py >> amazing_code1_console_output.txt &

nvidia-smi > amazing_code2_console_output.txt
python /NAS/home01/<asc1 OR asc2>/your/code/directory/amazing_code2.py >> amazing_code2_console_output.txt &

nvidia-smi > amazing_code2_console_output.txt
python /NAS/home01/<asc1 OR asc2>/your/code/directory/amazing_code2.py >> amazing_code2_console_output.txt