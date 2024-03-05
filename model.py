import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class CleanerModel(nn.Module):
    def __init__(self, in_state, hidden_state, out_state):
        super(CleanerModel, self).__init__()
        self.linear1 = nn.Linear(in_state, hidden_state)
        self.linear2 = nn.Linear(hidden_state, out_state)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        filename = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), filename)


class QTrainer:
    def __init__(self, model, lr=0.001, gamma=0.99):
        self.loss_fn = nn.MSELoss()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        if len(states.shape) == 1:
            states = states.unsqueeze(0)  # (B, N)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)  # (B)
            next_states = next_states.unsqueeze(0)  # (B, N)
            dones = dones.unsqueeze(0)  # (B)

        preds = self.model(states)  # (B, O)
        targets = preds.clone()

        # new Q: reward
        for idx, done in enumerate(dones):
            Q_new = rewards[idx]
            # predict next step
            if not done:
                next_pred = self.model(next_states[idx])
                Q_new = rewards[idx] + self.gamma * torch.max(next_pred)

            targets[idx, torch.argmax(actions[idx])] = Q_new
        loss = self.loss_fn(targets, preds)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    test_model = CleanerModel(3, 10, 5)
    test_q = QTrainer(test_model)
    print(test_q.train_step([0, 1, 2], [0,0, 0,0, 1], 1, [2,3,4], 0))