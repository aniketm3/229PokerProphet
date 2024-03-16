
import csv
import os
import torch
import sys
import numpy as np
sys.path.append('/Users/aniket/github/229PokerProphet-1/rlcard')
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import tournament, plot_curve
import matplotlib.pyplot as plt


model_paths = [
    '/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_0.pth',
    '/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_1.pth',
    '/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_2.pth'
]
models = [torch.load(path) for path in model_paths]

# Define the EnsembleAgent class
class EnsembleAgent:
    def __init__(self, models):
        self.models = models
        self.use_raw = False  

    def eval_step(self, state):
        action_probs = np.mean([model.predict(state) for model in self.models], axis=0)
        action = np.argmax(action_probs)
        return action, action_probs

env = rlcard.make('leduc-holdem')
ensemble_agent = EnsembleAgent(models)
opponents = [RandomAgent(num_actions=env.num_actions)]
env.set_agents([ensemble_agent, opponents[0]])


ensemble_performances = []

# Number of trials
num_trials = 500

# Run the tournament 'num_trials' times
print(f"Running {num_trials} tournaments...")
for trial in range(num_trials):
    results = tournament(env, num_trials)
    ensemble_performances.append(results[0])

    if (trial + 1) % 10 == 0:
        print(f"Completed {trial + 1}/{num_trials} trials. Current average performance: {np.mean(ensemble_performances):.4f}")

# Plotting the sorted performances
plt.figure()
plt.hist(ensemble_performances, bins=20, density=True) 
plt.title('Histogram of Ensemble Agent Rewards Over 100 Trials')
plt.xlabel('Reward')
plt.ylabel('Density')
plt.grid(True)
plt.show()
mean_performance = np.mean(ensemble_performances)
variance_performance = np.var(ensemble_performances)

# Print the statistics
print(f"Mean of Ensemble Agent's Performance: {mean_performance}")
print(f"Variance of Ensemble Agent's Performance: {variance_performance}")