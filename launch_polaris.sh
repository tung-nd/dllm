. /etc/profile

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"

# Figure out training environment
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $PBS_NODEFILE)
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
fi

# Fallback for single-node runs where MASTER_RANK may be unset
MASTER_RANK=${MASTER_RANK:-$HOSTNAME}

# GPUs per node (override via env GPUS_PER_NODE if needed)
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TOTAL_PROCS=$((NNODES * GPUS_PER_NODE))

#NNODES=1
#RANKS=$HOSTNAME
echo $NNODES
# Commands to run prior to the Python script for setting up the environment
PRELOAD="source /etc/profile ; "
PRELOAD+="module use /soft/modulefiles/;"
PRELOAD+="ml conda;"
PRELOAD+="conda activate mae_polaris;"
PRELOAD+="export OMP_NUM_THREADS=1 ; "
PRELOAD+="export NODES=$NNODES; "
PRELOAD+="export MASTER_ADDR=$MASTER_RANK; "
PRELOAD+="export MASTER_PORT=${MASTER_PORT:-29500}; "
# PRELOAD+="export NCCL_IB_DISABLE=1; "
PRELOAD+="export MPICH_GPU_SUPPORT_ENABLED=0; "
PRELOAD+="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "

# time python process to ensure timely job exit
TIMER="timeout 718m "

# torchrun launch configuration
LAUNCHER="accelerate launch "
LAUNCHER+="--num_machines=$NNODES --num_processes=$TOTAL_PROCS "
LAUNCHER+="--rdzv_backend=c10d "
LAUNCHER+="--main_process_ip=$MASTER_RANK --main_process_port=${MASTER_PORT:-29500} "

# Training script and parameters
CMD="--config_file configs/accelerate/fsdp.yaml train.py --config configs/s1k_finetune_ar_cot.yaml"

FULL_CMD=" $PRELOAD $TIMER $LAUNCHER $CMD $@ "
echo "Training Command: $FULL_CMD"


# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do #${RANKS[*]:0:21}; do #$RANKS; do
    NODE_LAUNCHER="$LAUNCHER --machine_rank=$RANK"
    NODE_CMD=" $PRELOAD $TIMER $NODE_LAUNCHER $CMD $@ "
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval "$NODE_CMD" &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $NODE_CMD" &
    fi
    RANK=$((RANK+1))
done

wait
