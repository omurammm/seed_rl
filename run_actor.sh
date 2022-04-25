NUM_ACTORS=$1
ENV_BATCH_SIZE=$2
TASK=$3

NUM_ENVS=$(($NUM_ACTORS*$ENV_BATCH_SIZE))

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
CUDA_VISIBLE_DEVICES='0'
python atari/r2d2_main.py --run_mode=actor --logtostderr --num_envs=$NUM_ENVS --task=$TASK --env_batch_size=$ENV_BATCH_SIZE
