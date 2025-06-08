"""
Brenda Silva Machado 

plotreward.py
"""

import os
import pickle
import matplotlib.pyplot as plt

def plot_reward(path):
    pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
    plt.figure(figsize=(10, 6))

    for i, file in enumerate(pkl_files):
        file_path = os.path.join(path, file)
        with open(file_path, 'rb') as f:
            fitness_data = pickle.load(f)
        
        plt.plot(fitness_data, label=f'Epoch {file.split(".")[0]}')

    plt.xlabel('Epoch')
    plt.ylabel('Average Rewards')
    plt.legend()  
    plt.show()


plot_reward('src/data/reward/') 
