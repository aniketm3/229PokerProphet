import numpy as np
import os
import torch
import sys
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import tournament, plot_curve
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import csv
from rlcard.agents import CFRAgent

class UnsupervisedAgent(object):
    def __init__(self, model):
        self.model = model
        self.use_raw = False
        # Dynamically set number of actions based on the model classes
        self.num_actions = len(model.classes_)

    def preprocess_state(self, state):
        """Preprocess the game state to fit the model's expected input format."""
        # Number of possible actions is fixed based on the game's rules
        num_possible_actions = 3
        # Create the binary representation of legal actions
        legal_actions_binary = [1 if i in state['legal_actions'] else 0 for i in range(num_possible_actions)]
        # Extract the observation array
        obs = state['obs']
        # Concatenate the observation array with the binary representation of legal actions
        processed_obs = np.concatenate((obs, np.array(legal_actions_binary)))
        # Reshape for the model
        processed_obs_reshaped = processed_obs.reshape(1, -1)
        return processed_obs_reshaped

    def step(self, state):
        """Predicts the next action to take based on the current state."""
        processed_obs_reshaped = self.preprocess_state(state)
        # Predict action
        action = self.model.predict(processed_obs_reshaped)
        return action[0]  # Assuming 'predict' returns a list, get the first element

    def eval_step(self, state):
        """Evaluates the next action to take (and additional info) based on the current state."""
        processed_obs_reshaped = self.preprocess_state(state)
        # Predict action
        action = self.model.predict(processed_obs_reshaped)
        # Optionally, compute and return action probabilities
        # This is a placeholder; adapt it if your model can provide probabilities
        probs = np.zeros(self.num_actions)
        probs[action[0]] = 1.0  # Assuming the model is very confident in its prediction
        info = {'probs': probs}
        return action[0], info
