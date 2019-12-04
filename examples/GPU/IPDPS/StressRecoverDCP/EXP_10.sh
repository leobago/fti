#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --job-name=ckptD
#SBATCH --cpus-per-task=40
#SBATCH --time=00-10:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=./EXP_10.out

baseConfig=EXP_10.fti
fileName=EXP_10/stdout
config=EXP_10_config.fti
tmp=${config}_tmp
curDir=$(pwd)

module purge
module load gcc/6.4.0  openmpi/3.0.0  cuda/10.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/scratch/bsc93/bsc93780/IPDPS/install/dcp/lib
mkdir -p ./EXP_10/

ratios=(1 2 3 4 5 6 7 7 9 10)
diffs=(1 2 3)  
for i in ${ratios[@]}; do
  ratioDir="${curDir}/profile/EXP_10/recover_${i}/"
  for j in ${diffs[@]}; do
    metaDir="${curDir}/Meta/"
    globalDir="${curDir}/Global/"
    diffDir="${ratioDir}diff_${j}/"
    mkdir -p ${diffDir}
    profileDir="${diffDir}run_"
    mkdir -p ${globalDir}
    mkdir -p ${metaDir}
    locarDir="${NVME1DIR}/Local/:${NVME1DIR}/Local/"
    awk -v var=$locarDir '$1 == "ckpt_dir" {$3 = var}1' $baseConfig > $tmp; cp $tmp $config
    awk -v var=$globalDir '$1 == "glbl_dir" {$3 = var}1' $config > $tmp; cp $tmp $config
    awk -v var=$metaDir '$1 == "meta_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
    awk -v var=$profileDir '$1 == "profile_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
    srun ./stressRecoverDCP.exe 48 0.5 $config 0.5 $i > "${fileName}_${i}_${j}.log" 2>"${fileName}_${i}_${j}.err"
    srun ./stressRecoverDCP.exe 48 0.5 $config 0.5 $i > "${fileName}_${i}_${j}_recover.log" 2>"${fileName}_${i}_${j}_recover.err"
    ./removeDir.sh
  done
done



