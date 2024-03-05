import numpy as np
from torch.distributions.categorical import Categorical
import torch
from model import CleanerModel, QTrainer
from game import VacuumCleaner
import tqdm
import random

BATCH_SIZE = 20


class CleanerAgent:
    def __init__(self, model: CleanerModel):
        self.game = VacuumCleaner()
        self.game.reset()
        self.model = model
        self.memory = []
        self.memory_capacity = 10000
        self.epsilon = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_timestep = 1000
        self.q_trainer = QTrainer(self.model)
        self.n_games = 0
        self.step_at_a_game = 0
        self.max_step_at_a_game = 30

    def get_state(self):
        env_state = self.game.state  # (8 * 8)
        env_state = env_state.flatten()
        result = np.zeros((self.game.num_cleaner, 66))
        for idx, cleaner in enumerate(self.game.cleaners):
            result[idx] = np.concatenate((env_state, cleaner))
        return result

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            preds = self.model(state)
            preds = torch.softmax(preds, dim=0)
        distribution = Categorical(preds)
        rand = np.random.uniform(0, 1)
        if rand < self.epsilon:
            actions = distribution.sample()
        else:
            actions = torch.argmax(preds, dim=-1)
        one_hot = np.zeros((state.shape[0], 5), dtype=int)
        for idx, action in enumerate(actions):
            one_hot[idx][action.item()] = 1

        return one_hot

    def add_to_memory(self, states, actions, rewards, next_states, dones):
        if len(self.memory) >= self.memory_capacity:
            return

        for idx in range(len(dones)):
            self.memory.append([states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]])

    def train_on_short_memory(self, states, actions, rewards, next_states, dones):
        self.q_trainer.train_step(states, actions, rewards, next_states, dones)

    def train_on_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
        mini_batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.q_trainer.train_step(states, actions, rewards, next_states, dones)

    def train(self):
            while True:
                states = self.get_state()
                actions = self.get_action(states)
                rewards, done = self.game.step(actions)
                dones = np.full((len(states), ), done)
                next_states = self.get_state()
                self.add_to_memory(states, actions, rewards, next_states, dones)

                self.train_on_short_memory(states, actions, rewards, next_states, dones)

                if done or self.step_at_a_game >= self.max_step_at_a_game:
                    self.step_at_a_game = 0
                    self.game.reset()
                    self.n_games += 1
                    self.train_on_long_memory()
                    self.epsilon -= 0.1
                    # We need to store the model when getting the higher score
                    print('Save...', done, self.epsilon)
                    self.model.save()


if __name__ == '__main__':
    test = CleanerAgent()
    game_model = CleanerModel(66, 128, 5)
    test.train(game_model)
