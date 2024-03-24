class Bandit:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.theta = np.random.randn(n_actions, n_features)

    def get_reward(self, action, x):
        return x @ self.theta[action] + np.random.normal()

    def get_optimal_reward(self, x):
        return np.max(x @ self.theta.T)


# Usage example

# Define the bandit environment
n_actions = 10
n_features = 5
bandit = Bandit(n_actions, n_features)

# Define the model agent
model_agent = model(n_features, *args)

# Define the context (features)
x = np.random.randn(n_features)

# The agent makes a prediction
pred_rewards = np.array([model_agent.predict(x) for _ in range(n_actions)])

# The agent chooses an action
action = np.argmax(pred_rewards)

# The agent gets a reward
reward = bandit.get_reward(action, x)

# The agent updates its parameters
model_agent.update(x, reward)


import matplotlib.pyplot as plt
from optimxrl.Agents.neuralbandit import NeuralBandit
from linucb import LinUCB
import gym

import numpy as np
from optimx import make


model_name = f"personalized_unispinux_strategy"
VERSION = "0.0"
MODEL_ENV = "dev"
model_db2 = make(
    f"cache/{model_name}-v{VERSION}",
    db_name="dqnmodel_test2.db",
    env=MODEL_ENV,
    db_type="diskcache",
)


# Define the bandit environment
n_actions = 10
n_features = 5

# Define the agents

neural_agent = NeuralBandit(n_actions, n_features, 64, lr=0.01, model_db=model_db2)
linucb_agent = LinUCB(
    actions=[a for a in range(n_actions)],
    obs_dim=n_features,
    alpha=0.5,
    model_db=model_db2,
)
# Run the simulation for each agent
n_steps = 1000
agents = [neural_agent, linucb_agent]
cumulative_rewards = {agent.__class__.__name__: np.zeros(n_steps) for agent in agents}
cumulative_regrets = {agent.__class__.__name__: np.zeros(n_steps) for agent in agents}
for agent in agents:
    print(agent.__class__.__name__)
    for t in range(n_steps):
        x = np.random.randn(n_features)
        if agent.__class__.__name__ == "LinUCB":
            context = x.tolist()
            xx = {a: context for a in range(n_actions)}
            action = agent.act(xx, "test2")

        else:
            action = agent.act(x[True, :], "test")
            # action = np.argmax(pred_rewards)
        reward = bandit.get_reward(action, x)

        optimal_reward = bandit.get_optimal_reward(x)
        if agent.__class__.__name__ == "LinUCB":
            agent.learn(xx, {action: reward}, "test2")
        else:
            agent.learn(x[True, :], action, reward, "test", True)
        cumulative_rewards[agent.__class__.__name__][t] = (
            reward
            if t == 0
            else cumulative_rewards[agent.__class__.__name__][t - 1] + reward
        )
        cumulative_regrets[agent.__class__.__name__][t] = (
            optimal_reward - reward
            if t == 0
            else cumulative_regrets[agent.__class__.__name__][t - 1]
            + optimal_reward
            - reward
        )

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(121)
for agent_name, rewards in cumulative_rewards.items():
    plt.plot(rewards, label=agent_name)
plt.xlabel("Steps")
plt.ylabel("Cumulative Rewards")
plt.legend()

plt.subplot(122)
for agent_name, regrets in cumulative_regrets.items():
    plt.plot(regrets, label=agent_name)
plt.xlabel("Steps")
plt.ylabel("Cumulative Regrets")
plt.legend()

plt.show()
