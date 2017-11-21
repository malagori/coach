#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from agents.agent import *


class HumanAgent(Agent):
    def __init__(self, env, tuning_parameters, replicated_device=None, thread_id=0):
        Agent.__init__(self, env, tuning_parameters, replicated_device, thread_id)

        screen.log_title("Human Control Mode - Use keyboard keys to move. Press escape to quit.")

    def train(self):
        return 0

    def choose_action(self, curr_state, phase=RunPhase.TRAIN):
        action = self.env.get_action_from_user()

        time.sleep(0.1)

        if not self.env.renderer.is_open:
            self.save_replay_buffer_and_exit()

        return action, {"action_value": 0}

    def save_replay_buffer_and_exit(self):
        replay_buffer_path = os.path.join(logger.experiments_path, 'replay_buffer.p')
        self.memory.tp = None
        to_pickle(self.memory, replay_buffer_path)
        screen.log_title("Replay buffer was stored in {}".format(replay_buffer_path))
        exit()
