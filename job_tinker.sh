#!/bin/bash
#SBATCH --job-name=ddnet
#SBATCH --nodes 1
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8384                # total memory per node (4 GB per cpu-core is default)
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1             #GPU per node
#SBATCH --partition=dgx_normal_q # slurm partition
#SBATCH --time=24:30:00          # time limit
#SBATCH -A HPCBIGDATA2           # account name

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=8888
echo "master port: $MASTER_PORT"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=$WORLD_SIZE"
echo "slurm job: $SLURM_JOBID"
#expor job_id=$SLURM_JOBID
mkdir -p $SLURM_JOBID;cd $SLURM_JOBID
set | grep SLURM | while read line; do echo "# $line"; done > slurm.txt
env | grep -i -v slurm | sort > env.txt
#cp ../sparse_ddnet.py .
#cp ../sparse_ddnet_pp.py .

mkdir -p ./loss/
mkdir -p ./reconstructed_images/
mkdir -p ./reconstructed_images/test
mkdir -p ./visualize
mkdir -p ./visualize/test/
mkdir -p ./visualize/test/mapped/
mkdir -p ./visualize/test/diff_target_out/
mkdir -p ./visualize/test/diff_target_in/
mkdir -p ./visualize/test/input/
mkdir -p ./visualize/test/target/
cd ..
echo "tmpfs for this job at $TMPDIR"
echo "Staging full data per node"

#cd $TMPDIR/tmpfs
export dest_dir=$TMPDIR/tmpfs
#cp -r /projects/synergy_lab/garvit217/enhancement_data $dest_dir
echo "Staged full data per node on $dest_dir"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURMTMPDIR=$SLURMTMPDIR"

echo "working directory = "$SLURM_SUBMIT_DIR
echo "custom module path: $EASYBUILD_INSTALLPATH_MODULES"


echo "current dir: $PWD"
chmod u+x trainers_new_dl.py
chmod u+x trainers.py
chmod u+x ddnet_inference.py

: "${NEXP:=1}"



if [ "$enable_gr" = "true" ]; then
  export conda_env="pytorch_night"
else
  export conda_env="ddnet_test"
fi
#module reset
#module load site/tinkercliffs/easybuild/arc.arcadm
module load site/tinkercliffs/easybuild/setup
module load apps
module load Anaconda3/2020.11
# module restore cu117
export infer_command="conda run -n ${conda_env} python ddnet_inference.py --filepath ${SLURM_JOBID} --batch ${batch_size} --epochs ${epochs} --out_dir ${SLURM_JOBID} --model ${model} --retrain ${retrain}"

if [ "$new_load" = "true" ]; then
  export file="trainers_new_dl.py"
else
  export file="trainers.py"
fi


export CMD="python ${file} --batch ${batch_size} --epochs ${epochs} --retrain ${retrain} --out_dir ${SLURM_JOBID} --amp ${mp} --num_w $num_data_w  --new_load ${new_load} --prune_amt $prune_amt --prune_t $prune_t  --wan $wandb --lr ${lr} --dr ${dr} --distback ${distback} --enable_profile ${enable_profile} --gr_mode ${gr_mode} --gr_backend ${gr_back} --enable_gr=${enable_gr} --schedtype ${schedtype} --model ${model} --port ${MASTER_PORT}"
echo "CMD: ${CMD}"


# if [ "$enable_gr" = "true" ]; then
#   export imagefile="/projects/synergy_lab/ayush/containers/pytorch_2.0.sif"
# else
#   export imagefile="/projects/synergy_lab/ayush/containers/pytorch_22.04.sif"
# fi

# module list
# module load containers/apptainer
# export BASE="apptainer  exec --nv --writable-tmpfs --bind=/projects/synergy_lab,/cm/shared,${TMPFS} ${imagefile} "

conda init
source /home/ritvikp/.bashrc
conda activate ddnet_test
if [  "$inferonly"  == "true" ]; then
  export filepath=$1
  python ddnet_inference.py --filepath ${filepath} --batch ${batch_size} --epochs ${epochs} --out_dir ${filepath}
elif [ "$enable_profile" = "true" ];then
#  module load CUDA/11.7.0
  echo "cuda home: ${CUDA_HOME}"
  srun --wait=120 --kill-on-bad-exit=0 --cpu-bind=none dlprof --output_path=${SLURM_JOBID} --nsys_base_name=nsys_${SLURM_PROCID} --profile_name=dlpro_${SLURM_PROCID} --mode=pytorch --nsys_opts="-t osrt,cuda,nvtx,cudnn,cublas,cusparse,mpi, --cuda-memory-usage=true" -f true --reports=all --delay 60 --duration 120 ${CMD}
else
  for _experiment_index in $(seq 1 "${NEXP}"); do
    (
  	echo "Beginning trial ${_experiment_index} of ${NEXP}"
  	srun  --unbuffered --wait=120 --kill-on-bad-exit=0 --cpu-bind=none $CMD
    )
  done
  wait
  $infer_command
fi
