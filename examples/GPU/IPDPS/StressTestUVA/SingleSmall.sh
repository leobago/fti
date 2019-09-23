#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --job-name=ckptU
#SBATCH --cpus-per-task=40
#SBATCH --time=00-00:45:00
#SBATCH --gres=gpu:4
#SBATCH --output=./SingleSmall.out

baseConfig=config.fti
fileName=SingleSmall/stdout
config=SingleSmall_config.fti
tmp=${config}_tmp
curDir=$(pwd)

module purge
module load gcc/6.4.0  openmpi/3.0.0  cuda/10.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/scratch/bsc93/bsc93780/IPDPS/install/dcp/lib
mkdir -p ./SingleSmall/

ratios=(0.1 0.5 0.9)
for i in ${ratios[@]}; do
profileDir="${curDir}/profile/SingleSmall/ratio_${i}/"
metaDir="${curDir}/Meta/"
globalDir="${curDir}/Global/"
mkdir -p ${profileDir}
profileDir="${profileDir}run_${j}_"
mkdir -p ${globalDir}
mkdir -p ${metaDir}
locarDir="${NVME1DIR}/Local/:${NVME2DIR}/Local/"
awk -v var=$locarDir '$1 == "ckpt_dir" {$3 = var}1' $baseConfig > $tmp; cp $tmp $config
awk -v var=$globalDir '$1 == "glbl_dir" {$3 = var}1' $config > $tmp; cp $tmp $config
awk -v var=$metaDir '$1 == "meta_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
awk -v var=$profileDir '$1 == "profile_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
srun ./stressUVA.exe 48 $i $config > "${fileName}_${i}.log" 2>"${fileName}_${i}.err"
done



