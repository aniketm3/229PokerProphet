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
from unsupervised_agent import UnsupervisedAgent
from sklearn.neighbors import KNeighborsClassifier


# TRAIN MY UNSUPERVISED MODEL
df = pd.read_csv('/Users/charlieabowd/229PokerProphet/CFR_training_data.csv')
# Features (dropping player for now for simplicity in testing agent)
X = df.drop(columns=['player','game_id','round','action']) 
# Target variable
y = df['action']  

# IF WE WANT A LOGISTIC REGRESSION MODEL
logreg_model = LogisticRegression(max_iter = 1000).fit(X, y)

# ANOTHER OPTION, K-NEAREST-NEIGHBORS
#knn_model = KNeighborsClassifier(n_neighbors=5).fit(X, y)

# NOW, CREATE GAME OF UNSUPERVISED AGENT AGAINST CFR AGENT
# Create a Leduc Hold'em environment
env = rlcard.make('leduc-holdem')

# load CFR agent to play against
cfr_model_path = '/Users/charlieabowd/229PokerProphet'
CFR_agent = CFRAgent(env, model_path = cfr_model_path)
CFR_agent.load()

# switch model depending on which one we trained above
agent1 = UnsupervisedAgent(model = logreg_model)
agent2 = CFR_agent
env.set_agents([agent1, agent2])
# Start a new game
state, _ = env.reset()

unsup_performances = []

# Number of trials
num_trials = 500

# Run the tournament 'num_trials' times
print(f"Running {num_trials} tournaments...")
for trial in range(num_trials):
    results = tournament(env, num_trials)
    # Assuming your ensemble agent is the first in the list
    unsup_performances.append(results[0])

    if (trial + 1) % 10 == 0:
        print(f"Completed {trial + 1}/{num_trials} trials. Current average performance: {np.mean(unsup_performances):.4f}")

# Plotting the sorted performances
plt.figure()
plt.hist(unsup_performances, bins=20, density=True)  # 'density=True' will normalize the histogram
plt.title('Histogram of LogReg Rewards Over 100 Trials')
plt.xlabel('Reward')
plt.ylabel('Density')
plt.grid(True)
plt.show()
mean_performance = np.mean(unsup_performances)
variance_performance = np.var(unsup_performances)

# Print the statistics
print(f"Mean of Unsupervised Agent's Performance: {mean_performance}")
print(f"Variance of Unsupervised Agent's Performance: {variance_performance}")



