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

import numpy as np
from utils import *
from configurations import Preset
from renderer import Renderer


class EnvironmentWrapper(object):
    def __init__(self, tuning_parameters):
        """
        :param tuning_parameters:
        :type tuning_parameters: Preset
        """
        # env initialization
        self.game = []
        self.actions = {}
        self.observation = []
        self.reward = 0
        self.done = False
        self.default_action = 0
        self.last_action_idx = 0
        self.measurements = []
        self.action_space_low = 0
        self.action_space_high = 0
        self.action_space_abs_range = 0
        self.discrete_controls = True
        self.action_space_size = 0
        self.key_to_action = {}
        self.width = 1
        self.height = 1
        self.is_state_type_image = True
        self.measurements_size = 0
        self.phase = RunPhase.TRAIN
        self.tp = tuning_parameters
        self.record_video_every = self.tp.visualization.record_video_every
        self.env_id = self.tp.env.level
        self.video_path = self.tp.visualization.video_path
        self.is_rendered = self.tp.visualization.render
        self.seed = self.tp.seed
        self.frame_skip = self.tp.env.frame_skip
        self.human_control = self.tp.env.human_control
        self.wait_for_explicit_human_action = False
        self.is_rendered = self.is_rendered or self.human_control
        self.game_is_open = True
        self.renderer = Renderer()

    def _update_observation_and_measurements(self):
        # extract all the available measurments (ovservation, depthmap, lives, ammo etc.)
        pass

    def _restart_environment_episode(self, force_environment_reset=False):
        """
        :param force_environment_reset: Force the environment to reset even if the episode is not done yet. 
        :return: 
        """
        pass

    def _idx_to_action(self, action_idx):
        """
        Convert an action index to one of the environment available actions.
        For example, if the available actions are 4,5,6 then this function will map 0->4, 1->5, 2->6
        :param action_idx: an action index between 0 and self.action_space_size - 1
        :return: the action corresponding to the requested index
        """
        return self.actions[action_idx]

    def _action_to_idx(self, action):
        """
        Convert an environment action to one of the available actions of the wrapper.
        For example, if the available actions are 4,5,6 then this function will map 4->0, 5->1, 6->2
        :param action: the environment action
        :return: an action index between 0 and self.action_space_size - 1, or -1 if the action does not exist
        """
        for key, val in self.actions.items():
            if val == action:
                return key
        return -1

    def _preprocess_observation(self, observation):
        """
        Do initial observation preprocessing such as cropping, rgb2gray, rescale etc.
        :param observation: a raw observation from the environment
        :return: the preprocessed observation
        """
        pass

    def get_action_from_user(self):
        """
        Get an action from the user keyboard
        :return: action index
        """
        if self.wait_for_explicit_human_action:
            while len(self.renderer.pressed_keys) == 0:
                self.renderer.get_events()

        if self.key_to_action == {}:
            # the keys are the numbers on the keyboard corresponding to the action index
            if len(self.renderer.pressed_keys) > 0:
                action_idx = self.renderer.pressed_keys[0] - ord("1")
                if 0 <= action_idx < self.action_space_size:
                    return action_idx
        else:
            # the keys are mapped through the environment to more intuitive keyboard keys
            for key in self.renderer.pressed_keys:
                for env_key in self.key_to_action.keys():
                    if key in env_key:
                        return self.key_to_action[env_key]
                    
        # return the default action 0 so that the environment will continue running
        return self.default_action

    def step(self, action_idx):
        """
        Perform a single step on the environment using the given action
        :param action_idx: the action to perform on the environment
        :return: A dictionary containing the observation, reward, done flag, action and measurements
        """
        pass

    def render(self):
        """
        Call the environment function for rendering to the screen
        """
        self.renderer.render_image(self.get_rendered_image())

    def reset(self, force_environment_reset=False):
        """
        Reset the environment and all the variable of the wrapper
        :param force_environment_reset: forces environment reset even when the game did not end
        :return: A dictionary containing the observation, reward, done flag, action and measurements
        """
        self._restart_environment_episode(force_environment_reset)
        self.done = False
        self.reward = 0.0
        self.last_action_idx = 0
        self._update_observation_and_measurements()
        return {'observation': self.observation,
                'reward': self.reward,
                'done': self.done,
                'action': self.last_action_idx,
                'measurements': self.measurements}

    def get_random_action(self):
        """
        Returns an action picked uniformly from the available actions
        :return: a numpy array with a random action
        """
        if self.discrete_controls:
            return np.random.choice(self.action_space_size)
        else:
            return np.random.uniform(self.action_space_low, self.action_space_high)

    def change_phase(self, phase):
        """
        Change the current phase of the run. 
        This is useful when different behavior is expected when testing and training
        :param phase: The running phase of the algorithm
        :type phase: RunPhase
        """
        self.phase = phase

    def get_rendered_image(self):
        """
        Return a numpy array containing the image that will be rendered to the screen.
        This can be different from the observation. For example, mujoco's observation is a measurements vector.
        :return: numpy array containing the image that will be rendered to the screen
        """
        return self.observation
