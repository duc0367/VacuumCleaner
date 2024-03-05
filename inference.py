from model import CleanerModel
from agent import CleanerAgent
import time

import torch

PATH = 'model/model.pth'
model = CleanerModel(66, 128, 5)
model.load_state_dict(torch.load(PATH))

agent = CleanerAgent(model)


if __name__ == '__main__':
    while True:
        states = agent.get_state()
        actions = agent.get_action(states)
        rewards, done = agent.game.step(actions)
        time.sleep(0.5)