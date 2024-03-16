import rlcard
from rlcard.agents import CFRAgent
import pandas as pd
import numpy as np

# Plays 1000 games of Leduc holdem with 2 CFR agents, stores results in df

# Initialize the environment and agents
env = rlcard.make('leduc-holdem', config={'allow_step_back': True})

# path where I stored my pretrained model
cfr_model_path = '/Users/charlieabowd/229PokerProphet'
# load the agent 
CFR_agent = CFRAgent(env, model_path = cfr_model_path)
CFR_agent.load()

num_players = 2
random_agents = [CFR_agent for _ in range(num_players)]
env.set_agents(random_agents)

# Update columns to match the structure of each row in data
base_columns = ['game_id', 'round', 'player', 'action']
# the observation array is always of length 36
obs_columns = [f'obs_{i}' for i in range(36)]  
# columns storing binary rep. of legal actions
legal_actions_columns = [f'legal_action_{i}' for i in range(3)]  
# Combine column names
column_names = base_columns[:-1] + obs_columns + legal_actions_columns + [base_columns[-1]]

# list of all the training data
data = []

# Simulate 1000 games
for game_id in range(1000):
    state, player_id = env.reset()

    done = False
    while not done:
        current_player = env.get_player_id()
        current_state = env.get_state(current_player)
        
        # Determine action using the agent's step or eval_step method
        action, _ = random_agents[current_player].eval_step(current_state)
        
        # Proceed with the environment step using the chosen action
        next_state, player_id = env.step(action)
        
        # Check if the game is done
        done = env.is_over()
    
        # Get binary representation of legal actions (out of 3 possible)
        legal_actions_binary = [1 if i in current_state['legal_actions'] else 0 for i in range(3)]
        
        # Flatten 'obs' and combine with legal actions and other info
        row = [game_id, env.timestep, current_player] + current_state['obs'].tolist() + legal_actions_binary + [action]
        #add do the list
        data.append(row)
    

# Create a DataFrame from the collected data
df = pd.DataFrame(data, columns=column_names)
df.to_csv('/Users/charlieabowd/229PokerProphet/CFR_training_data.csv', index=False)  # Set index=False if you don't want to save the index

# Display the first few rows of the DataFrame
print(df.head())
