#!/bin/bash
#SBATCH --ntasks=32
#SBATCH --job-name=ckptD
#SBATCH --cpus-per-task=5
#SBATCH --time=00-00:45:00
#SBATCH --gres=gpu:4
#SBATCH --output=./SingleSmall.out

baseConfig=configH0.fti
fileName=SingleSmall/stdout
config=SingleSmall_config.fti
tmp=${config}_tmp
curDir=$(pwd)

module purge
module load gcc/6.4.0  openmpi/3.0.0  cuda/10.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/scratch/bsc93/bsc93780/IPDPS/install/dcp/lib
mkdir -p ./SingleSmall/

ratios=(0.2 0.5 0.8)
for i in ${ratios[@]}; do
profileDir="${curDir}/profile/SingleSmall/ratio_${i}/"
metaDir="${curDir}/Meta/"
globalDir="${curDir}/Global/"
mkdir -p ${profileDir}
profileDir="${profileDir}run_${j}_"
mkdir -p ${globalDir}
mkdir -p ${metaDir}
locarDir="${NVME1DIR}/Local/"
awk -v var=$locarDir '$1 == "ckpt_dir" {$3 = var}1' $baseConfig > $tmp; cp $tmp $config
awk -v var=$globalDir '$1 == "glbl_dir" {$3 = var}1' $config > $tmp; cp $tmp $config
awk -v var=$metaDir '$1 == "meta_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
awk -v var=$profileDir '$1 == "profile_dir" {$3 = var}1' $config> $tmp; cp $tmp $config
srun ./stressDCP.exe 40 0.5 $config $i > "${fileName}_${i}.log" 2>"${fileName}_${i}.err"
done



