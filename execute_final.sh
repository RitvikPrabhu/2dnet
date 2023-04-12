#!/bin/bash

module reset
module list

#export profile_prefix="dlprof --output_path=${SLURM_JOBID} --profile_name=dlpro_${SLURM_NODEID}_rank${SLURM_PROCID} --mode=pytorch -f true --reports=all -y 60 -d 120 --nsys_base_name=nsys_${SLURM_NODEID}_rank${SLURM_PROCID}  --nsys_opts=\"-t osrt,cuda,nvtx,cudnn,cublas\" "


#export profile_prefix="nsys profile -t cuda,nvtx,cudnn,cublas --show-output=true --force-overwrite=true --delay=60 --duration=220 --export=sqlite -o ${SLURM_JOBID}/profile_rank${SLURM_PROCID}_node_${SLURM_NODEID}"
export file="trainers.py"
export CMD="python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir $SLURM_JOBID --amp ${mp} --num_w $num_data_w  --new_load ${new_load} --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback} --enable_profile ${enable_profile} --gr_mode ${gr_mode} --gr_backend ${gr_back} --enable_gr=${enable_gr}"
echo "CMD: ${CMD}"

module load EasyBuild/4.6.1
module use $EASYBUILD_INSTALLPATH_MODULES
module load CUDA/11.6.0

module load Anaconda3
conda init
source ~/.bashrc
conda activate tttt
export infer_command="python ddnet_inference.py filepath ${SLURM_PROCID} --batch ${batch_size} --epochs ${epochs} --out_dir $SLURM_JOBID"
# change base conda env to nightly pytorch
if [ "$enable_gr" = "true" ]; then
  conda activate pytorch_night
else
  conda activate tttt
fi


export filepath=${SLURM_PROCID}
if [  "$inferonly"  == "true" ]; then
  export filepath=$1
  python ddnet_inference.py --filepath ${SLURM_PROCID} --batch ${batch_size} --epochs ${epochs} --out_dir ${SLURM_JOBID}
else
  if [ "$enable_profile" = "true" ];then
    dlprof --output_path=${SLURM_JOBID} --nsys_base_name=nsys_{SLURM_PROCID} --profile_name=dlpro_{SLURM_PROCID} --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas --cuda-memory-usage=true --kill=none" -f true --reports=all --delay 60 --duration 60 ${CMD}
  else
    $CMD
    $infer_command
  fi
fi


