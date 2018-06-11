#!/bin/bash
#SBATCH --job-name=HEAT
#SBATCH --output=HEAT_%j.out
#SBATCH --error=HEAT_%j.err
#SBATCH --ntasks=8
#SBATCH --nodes=4
#SBATCH --time 0-02:00
#SBATCH --qos=debug
FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=20 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee 10_20.out
FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=20 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee -a 10_20.out 

FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=100 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee 10_100.out 
FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=100 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee -a 10_100.out 

FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=500 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee 10_500.out 
FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=500 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee -a 10_500.out 

FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=1000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee 10_1000.out 
FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=1000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee -a 10_1000.out 

FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=2000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee 10_2000.out 
FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=2000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee -a 10_2000.out 

#FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=4000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee 10_4000.out 
#FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=4000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee -a 10_4000.out 
#
#FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=8000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee 10_8000.out 
#FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=8000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee -a 10_8000.out 
#
#FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=16000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee 10_16000.out 
#FTI_FIRST_CKPT=10 FTI_SECOND_CKPT=16000 mpirun -n 8 ./hd.exe 2048 config.fti 2>&1 | tee -a 10_16000.out 
