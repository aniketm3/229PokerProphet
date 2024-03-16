import rlcard
from rlcard.agents import CFRAgent
import pandas as pd
import numpy as np

# Trains the CFR Agent, saves it to specified filepath

# Initialize the environment
env = rlcard.make('leduc-holdem', config={'allow_step_back': True})

# Specify your custom model path here
custom_model_path = '/Users/charlieabowd/229PokerProphet'  # Change this to your desired path

# Initialize the CFR agent with your custom model path
agent = CFRAgent(env, model_path=custom_model_path)

# Define the number of training iterations
num_iterations = 10000

# Training loop
for i in range(num_iterations):
    print(f'Training iteration: {i+1}/{num_iterations}')
    agent.train()  # Perform one iteration of CFR
    
    # Optionally, you can save the model at certain intervals
    if (i+1) % 100 == 0:
        print('Saving model...')
        agent.save()

# Final model save
print('Final model save...')
agent.save()

print('Training complete.')

