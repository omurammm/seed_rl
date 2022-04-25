# coding=utf-8
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

"""Atari env factory."""

import tempfile

from absl import flags
from absl import logging
import atari_py  
import gym
from seed_rl.atari import atari_preprocessing
from seed_rl.common import common_flags  
import sys
sys.path.append("..")
from teacher_student_env import PPOStudentEnv

FLAGS = flags.FLAGS

# Environment settings.
flags.DEFINE_string('game', 'Breakout', 'Game name.')
flags.DEFINE_integer('max_random_noops', 30,
                     'Maximal number of random no-ops at the beginning of each '
                     'episode.')
flags.DEFINE_boolean('sticky_actions', False,
                     'When sticky actions are enabled, the environment repeats '
                     'the previous action with probability 0.25, instead of '
                     'playing the action given by the agent. Used to introduce '
                     'stochasticity in ATARI-57 environments, see '
                     'Machado et al. (2017).')


def create_environment(task, config):  
  # logging.info('Creating environment: %s', config.game)
  
  return atari_preprocessing.AtariPreprocessing(
      env,
      frame_skip=config.num_action_repeats,
      max_random_noops=config.max_random_noops)
