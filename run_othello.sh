#!/bin/bash
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# die () {
#     echo >&2 "$@"
#     exit 1
# }

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# cd $DIR

# ENVIRONMENT=$1
# AGENT=$2
# NUM_ACTORS=$3
# ENV_BATCH_SIZE=$4
# shift 4

# export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/

# ACTOR_BINARY="CUDA_VISIBLE_DEVICES='' python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=actor";
# LEARNER_BINARY="python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=learner";
# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# NUM_ENVS=$(($NUM_ACTORS*$ENV_BATCH_SIZE))


tmux new-session -d -t seed_rl
# mkdir -p /tmp/seed_rl
# cat >/tmp/seed_rl/instructions <<EOF
# Welcome to the SEED local training of ${ENVIRONMENT} with ${AGENT}.
# SEED uses tmux for easy navigation between different tasks involved
# in the training process. To switch to a specific task, press CTRL+b, [tab id].
# You can stop training at any time by executing 'stop_seed'
# EOF
# tmux send-keys "alias stop_seed='/seed_rl/stop_local.sh seed_rl'" ENTER
# tmux send-keys clear
# tmux send-keys KPEnter
# tmux send-keys "cat /tmp/seed_rl/instructions"
# tmux send-keys KPEnter
# tmux send-keys "python3 check_gpu.py 2> /dev/null"
# tmux send-keys KPEnter
# tmux send-keys "stop_seed"


NUM_ACTORS=$1
ENV_BATCH_SIZE=$2
NAME=$3
NUM_ENVS=$(($NUM_ACTORS*$ENV_BATCH_SIZE))

PRE_COMMAND='source ~/.bashrc ; conda activate seedrl ; export PYTHONPATH="$(dirname '"$PWD"')":$PYTHONPATH'

tmux new-window -d -n learner
# COMMAND='rm /tmp/agent -Rf; '"${LEARNER_BINARY}"' --logtostderr --pdb_post_mortem '"$@"' --num_envs='"${NUM_ENVS}"' --env_batch_size='"${ENV_BATCH_SIZE}"''
COMMAND='python othello/r2d2_main.py --run_mode=learner --logtostderr --num_envs='"${NUM_ENVS}"' --env_batch_size='"${ENV_BATCH_SIZE}"' --name='"${NAME}"''


echo $COMMAND
tmux send-keys -t "learner" "$PRE_COMMAND" ENTER
tmux send-keys -t "learner" "$COMMAND" ENTER

for ((id=0; id<$NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    export CUDA_VISIBLE_DEVICES='0,1'
    # COMMAND=''"${ACTOR_BINARY}"' --logtostderr --pdb_post_mortem '"$@"' --num_envs='"${NUM_ENVS}"' --task='"${id}"' --env_batch_size='"${ENV_BATCH_SIZE}"''
    COMMAND='python othello/r2d2_main.py --run_mode=actor --logtostderr --num_envs='${NUM_ENVS}' --task='${id}' --env_batch_size='${ENV_BATCH_SIZE}' --name='${NAME}''
    tmux send-keys -t "actor_${id}" "$PRE_COMMAND" ENTER
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER
done

# tmux new-window -d -n tensorboard
# tmux send-keys -t "tensorboard" "tensorboard --logdir /tmp/agent/ --bind_all" ENTER

tmux attach -t seed_rl
