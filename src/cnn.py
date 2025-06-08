"""
Brenda Silva Machado

cnn.py
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self, input_shape=(4, 84, 84)):
        """
        Inicializa a arquitetura da CNN com 3 camadas convolucionais e 2 fully connected.
        Gera duas saídas: direção (1 valor) e controle de freio/aceleração (2 valores).
        """
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self._conv_output_size = self._get_conv_output_size(input_shape)

        self.fc1 = nn.Linear(self._conv_output_size, 512)
        self.fc2 = nn.Linear(512, 512)

        self.steering = nn.Linear(512, 1)         
        self.brake_throttle = nn.Linear(512, 2)   # [freio, aceleração]

        self.mse_loss = nn.MSELoss()

        self.loss_log = []

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Inicializa os pesos usando Kaiming Normal.
        """
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2, self.steering, self.brake_throttle]:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def _get_conv_output_size(self, shape):
        """
        Calcula o tamanho da saída após as camadas convolucionais.
        """
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv3(self.conv2(self.conv1(input)))
            return int(torch.flatten(output, 1).size(1))

    def forward(self, x):
        """
        Propagação direta pela rede.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        steering_output = self.steering(x)
        brake_throttle_output = self.brake_throttle(x)

        return steering_output, brake_throttle_output

    def compute_loss(self, steering_pred, throttle_brake_pred, steering_real, throttle_real, brake_real):
        """
        Calcula o erro quadrático médio para as três saídas.
        """
        steering_loss = self.mse_loss(steering_pred.squeeze(1), steering_real)
        throttle_loss = self.mse_loss(throttle_brake_pred[:, 0], throttle_real)
        brake_loss = self.mse_loss(throttle_brake_pred[:, 1], brake_real)
        return steering_loss + throttle_loss + brake_loss

    def train_model(self, dataloader, epochs, learning_rate):
        """
        Treina o modelo usando o otimizador Adam e salva a loss média por época.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_log = []

        for epoch in range(epochs):
            running_loss = 0.0

            print(f"\n Epoch {epoch+1}/{epochs}")
            for states, actions_real in dataloader:
                steering_real = actions_real[:, 0]
                throttle_real = actions_real[:, 1]
                brake_real = actions_real[:, 2]

                optimizer.zero_grad()
                steering_pred, throttle_brake_pred = self(states)

                loss = self.compute_loss(steering_pred, throttle_brake_pred, steering_real, throttle_real, brake_real)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            mean_loss = running_loss / len(dataloader)
            self.loss_log.append(mean_loss)
            print(f"Loss média da época: {mean_loss:.6f}")

        os.makedirs('src/data/loss', exist_ok=True)
        with open('src/data/loss/mean_loss.pkl', 'wb') as f:
            pickle.dump(self.loss_log, f)

    def save_model(self, path):
        """
        Salva o modelo treinado no caminho especificado.
        """
        torch.save(self.state_dict(), path)
