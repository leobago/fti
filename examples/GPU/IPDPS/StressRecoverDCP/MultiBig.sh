#!/bin/bash
#SBATCH --ntasks=512
#SBATCH --job-name=ckptD
#SBATCH --cpus-per-task=5
#SBATCH --time=00-00:45:00
#SBATCH --gres=gpu:4
#SBATCH --output=./MultiBig.out

baseConfig=MultiBig.fti
fileName=MultiBig/stdout
config=MultiBig_config.fti
tmp=${config}_tmp
curDir=$(pwd)

module purge
module load gcc/6.4.0  openmpi/3.0.0  cuda/10.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/scratch/bsc93/bsc93780/IPDPS/install/dcp/lib
mkdir -p ./MultiBig/

ratios=(0.1 0.5 0.9)
for i in ${ratios[@]}; do
profileDir="${curDir}/profile/MultiBig/ratio_${i}/"
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
srun ./stressDCP.exe 768 0.5 $config 0.2 20 > "${fileName}_${i}.log" 2>"${fileName}_${i}.err"
done



