#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --job-name=ckptD
#SBATCH --cpus-per-task=160
#SBATCH --time=00-00:45:00
#SBATCH --gres=gpu:4
#SBATCH --output=./test.out

baseConfig=config.fti
fileName=stdout
config=finalConfig.fti
tmp=${baseConfig}_tmp
curDir=$(pwd)

module purge
module load gcc/6.4.0  openmpi/3.0.0  cuda/10.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/scratch/bsc93/bsc93780/IPDPS/install/unoptimized/lib

ratios=(0.1 0.5 0.9)
for i in ${ratios[@]}; do
profileDir="${curDir}/profile/ratio_${i}/"
metaDir="${curDir}/Meta/"
globalDir="${curDir}/Global/"
mkdir -p ${profileDir}
profileDir="${profileDir}run_${j}_"
awk -v var=$NVME1DIR '$1 == "ckpt_dir" {$3 = var}1' $baseConfig > $tmp; cp $tmp $config
awk -v var=$globalDir '$1 == "glbl_dir" {$3 = var}1' $config > $tmp; cp $tmp $config
awk -v var=$metaDir '$1 == "meta_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
awk -v var=$profileDir '$1 == "profile_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
srun ./stress.exe 48 $i $config > "${fileName}_${i}.log"
done



