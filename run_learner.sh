NUM_ACTORS=$1
ENV_BATCH_SIZE=$2

NUM_ENVS=$(($NUM_ACTORS*$ENV_BATCH_SIZE))

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
python atari/r2d2_main.py --run_mode=learner --logtostderr --num_envs=$NUM_ENVS --env_batch_size=$ENV_BATCH_SIZE
