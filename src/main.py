"""
Brenda Silva Machado

main.py
"""

import os
import torch
import numpy as np
import pickle
import cv2
from collections import deque
from torch.utils.data import DataLoader

from cnn import CNN 
from cnn_dataset import CNNDataset
from car_racing_v0 import CarRacing


def gray_scale(state: np.ndarray) -> np.ndarray:
    """
    Converte uma imagem RGB para escala de cinza normalizada (84x84).
    """
    gray_image = np.dot(state[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    gray_resized = cv2.resize(gray_image, (84, 84))
    normalized = gray_resized / 255.0
    return normalized


def preprocess_state(state: np.ndarray, frame_history: deque) -> tuple[torch.Tensor, deque]:
    """
    Preprocessa o estado atual da simulação para o formato esperado pela CNN:
    - Converte em escala de cinza
    - Empilha 4 frames
    - Retorna tensor PyTorch (1, 4, 84, 84)
    """
    gray_frame = gray_scale(state)
    frame_history.append(gray_frame)

    while len(frame_history) < 4:
        frame_history.append(gray_frame)

    stacked_frames = np.stack(list(frame_history), axis=0)
    state_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0)
    return state_tensor, frame_history


def train_model():
    """
    Carrega o dataset, inicializa o modelo e executa o processo de treinamento.
    """
    dataset_path = 'src/data/trajectories'
    dataset = CNNDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CNN(input_shape=(4, 84, 84))
    epochs = 1
    learning_rate = 0.001

    print(f"Iniciando treinamento: {epochs} época(s), alpha = {learning_rate}")
    model.train_model(dataloader, epochs, learning_rate)
    model.save_model('src/data/model/car_racing_model.pth')
    print("Treinamento concluído e modelo salvo.")


def eval_model():
    """
    Avalia o modelo treinado no ambiente CarRacing.
    Salva a recompensa média por episódio em 'src/data/rewards/avg_reward.pkl'.
    """
    env = CarRacing(render_mode="human")
    model = CNN(input_shape=(4, 84, 84))
    model.load_state_dict(torch.load('src/data/model/car_racing_model.pth'))
    model.eval()

    rewards = []
    actions = []
    frame_history = deque(maxlen=4)
    max_episodes = 10

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while True:
            state_tensor, frame_history = preprocess_state(state, frame_history)

            with torch.no_grad():
                steering, throttle_brake = model(state_tensor)

            action = [
                steering.item(),
                throttle_brake[0, 0].item(),  # brake
                throttle_brake[0, 1].item()   # throttle
            ]
            actions.append(action)

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if steps % 200 == 0 or terminated or truncated:
                print(f"\nAção: {[f'{x:+0.2f}' for x in action]}")
                print(f"Steps {steps}, Recompensa acumulada: {total_reward:+0.2f}")

            steps += 1
            if terminated or truncated or steps == 2000:
                rewards.append(total_reward)
                print(f"Fim do episódio {episode}")
                break

    env.close()

    avg_reward = np.mean(rewards)
    print(f"Recompensa média: {avg_reward:.2f}")

    os.makedirs('src/data/rewards', exist_ok=True)

    with open('src/data/rewards/avg_reward.pkl', 'wb') as f:
        pickle.dump(avg_reward, f)


if __name__ == "__main__":
    train_model()
    eval_model()
