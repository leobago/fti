#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --job-name=ckptD
#SBATCH --cpus-per-task=40
#SBATCH --time=00-03:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=./EXP_15.out

baseConfig=EXP_15.fti
fileName=EXP_15/stdout
config=EXP_15_config.fti
tmp=${config}_tmp
curDir=$(pwd)

module purge
module load gcc/6.4.0  openmpi/3.0.0  cuda/10.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/scratch/bsc93/bsc93780/IPDPS/install/dcp/lib
mkdir -p ./EXP_15/

ratios=(0.1 0.5 0.9)
diffs=(0.2 0.5 0.8)  
for i in ${ratios[@]}; do
  ratioDir="${curDir}/profile/EXP_15/ratio_${i}/"
  for j in ${diffs[@]}; do
    metaDir="${curDir}/Meta/"
    globalDir="${curDir}/Global/"
    diffDir="${ratioDir}diff_${j}/"
    mkdir -p ${diffDir}
    profileDir="${diffDir}run__"
    mkdir -p ${globalDir}
    mkdir -p ${metaDir}
    locarDir="${NVME1DIR}/Local/:${NVME2DIR}/Local/"
    awk -v var=$locarDir '$1 == "ckpt_dir" {$3 = var}1' $baseConfig > $tmp; cp $tmp $config
    awk -v var=$globalDir '$1 == "glbl_dir" {$3 = var}1' $config > $tmp; cp $tmp $config
    awk -v var=$metaDir '$1 == "meta_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
    awk -v var=$profileDir '$1 == "profile_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
    srun ./stressDCP.exe 48 $i $config $j 20 > "${fileName}_${i}_${j}.log" 2>"${fileName}_${i}_${j}.err"
    ./removeDir.sh
  done
done



