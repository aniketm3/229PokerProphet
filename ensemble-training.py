
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


# Load your trained models
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
        self.use_raw = False  # Set based on your model's requirements

    def eval_step(self, state):
        # Get average predictions from the models
        action_probs = np.mean([model.predict(state) for model in self.models], axis=0)
        # Choose action with the highest average probability
        action = np.argmax(action_probs)
        return action, action_probs

# Set up the environment and evaluate
env = rlcard.make('leduc-holdem')
ensemble_agent = EnsembleAgent(models)
opponents = [RandomAgent(num_actions=env.num_actions)]
env.set_agents([ensemble_agent, opponents[0]])

# Run evaluation
num_games = 1000
print("Evaluating ensemble agent...")
results = tournament(env, num_games)

# Performance analysis
ensemble_performance = np.mean(results)
print(f"Ensemble agent performance over {num_games} games: {ensemble_performance}")

# Plotting
plt.figure()
plt.bar(['Ensemble Agent'], [ensemble_performance])
plt.title('Performance of Ensemble Agent')
plt.ylabel('Average Reward')
plt.show()

# # Ensure the directory for saving the CSV exists
# results_dir = '/Users/aniket/github/229PokerProphet-1/results'
# os.makedirs(results_dir, exist_ok=True)
# csv_path = os.path.join(results_dir, 'ensemble_payoffs.csv')

# # Load your trained models (ensure these are the correct paths)
# model_paths = [
#     '/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_0.pth',
#     '/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_1.pth',
#     '/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_2.pth'
# ]
# models = [torch.load(path) for path in model_paths]

# # Define the EnsembleAgent class
# class EnsembleAgent:
#     def __init__(self, models):
#         self.models = models
#         self.use_raw = False

#     def eval_step(self, state):
#         action_probs = np.array([model.predict(state) for model in self.models])  # Convert to numpy array first
#         avg_action_probs = torch.mean(torch.from_numpy(action_probs), dim=0)
#         return torch.argmax(avg_action_probs).item(), avg_action_probs.numpy()


# # Set up the environment and agents for evaluation
# env = rlcard.make('leduc-holdem')
# ensemble_agent = EnsembleAgent(models)
# random_agent = RandomAgent(num_actions=env.num_actions)
# env.set_agents([ensemble_agent, random_agent])

# # Evaluate the ensemble agent
# num_games = 1000
# results = tournament(env, num_games)

# # Save results to CSV file
# with open(csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['episode', 'reward'])
#     for i, payoff in enumerate(results):
#         writer.writerow([i, payoff])  # Note: Adjust this line if 'results' structure differs

# # Plot the results
# plot_dir = '/Users/aniket/github/229PokerProphet-1/plots'
# os.makedirs(plot_dir, exist_ok=True)  # Ensure the directory exists
# plot_curve(csv_path, 'Ensemble Averaged Prediction Model', 'ensemble')

#plot_curve(csv_path, 'Ensemble Averaged Prediction Model', 'ensemble')


#ATTEMPT 1
# import torch
# import numpy as np
# import sys
# sys.path.append('/Users/aniket/github/229PokerProphet-1/rlcard')
# import csv

# import rlcard
# from rlcard.agents import RandomAgent
# from rlcard.utils import tournament, plot_curve

# # Load the trained models
# model_paths = ['/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_0.pth',
#                 '/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_1.pth',
#                 '/Users/aniket/github/229PokerProphet-1/experiments/leduc_holdem_dqn_result/model_2.pth']
# models = [torch.load(path) for path in model_paths]

# # Define a new agent class for ensemble averaging
# class EnsembleAgent:
#     def __init__(self, models):
#         self.models = models
#         self.use_raw = False

#     def eval_step(self, state):
#         action_probs = []
#         for model in self.models:
#             action_probs.append(model.predict(state))
#         # Convert list to NumPy array
#         action_probs_array = np.array(action_probs)

#         # Convert NumPy array to PyTorch tensor
#         action_probs_tensor = torch.from_numpy(action_probs_array)

#         # Compute average action probabilities
#         avg_action_probs = torch.mean(action_probs_tensor, dim=0)
#         return torch.argmax(avg_action_probs).item(), avg_action_probs.numpy()

# # Set up the environment for evaluation
# env = rlcard.make('leduc-holdem')


# # Initialize agents for evaluation
# ensemble_agent = EnsembleAgent(models)
# random_agent = RandomAgent(num_actions=env.num_actions)  # Or any other opponent agent

# # Set the agents for the environment
# env.set_agents([ensemble_agent, random_agent])

# # Evaluate the ensemble averaged prediction model
# num_games = 1000  # Number of games to play for evaluation
# results = tournament(env, num_games)

# # Plot the results
# # Assuming results is a dictionary with the payoffs stored as a list under the key 'payoffs'
# payoffs = results  # If you want to keep all agents' payoffs
# # Or, if you just want the ensemble agent's payoffs:
# ensemble_payoffs = [results[0]]  # Assuming the ensemble agent is the first agent

# # Path for the CSV file where the payoffs will be saved
# csv_path = '/Users/aniket/github/229PokerProphet-1/experiments/ensemble_payoffs.csv'

# # Save ensemble_payoffs to CSV
# with open(csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['episode', 'payoff'])  # Assuming these are the correct headers
#     for i, payoff in enumerate(ensemble_payoffs):
#         writer.writerow([i, payoff])

# # Plot the results using the saved CSV file
# plot_curve(csv_path, 'Ensemble Averaged Prediction Model', 'ensemble')
