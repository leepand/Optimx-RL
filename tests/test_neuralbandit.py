import matplotlib.pyplot as plt

# Define the bandit environment
n_actions = 10
n_features = 5
bandit = Bandit(n_actions, n_features)

# Define the agents
linucb_agent = LinUCB(n_actions, n_features, alpha=0.1)
neural_agent = NeuralBandit(n_actions, n_features, learning_rate=0.01)
tree_agent = DecisionTreeBandit(n_actions, n_features, max_depth=5)

# Run the simulation for each agent
n_steps = 1000
agents = [linucb_agent, tree_agent, neural_agent]
cumulative_rewards = {agent.__class__.__name__: np.zeros(n_steps) for agent in agents}
cumulative_regrets = {agent.__class__.__name__: np.zeros(n_steps) for agent in agents}
for agent in agents:
    print(agent)
    for t in range(n_steps):
        x = np.random.randn(n_features)
        pred_rewards = agent.predict([x])
        action = np.argmax(pred_rewards)
        reward = bandit.get_reward(action, x)
        optimal_reward = bandit.get_optimal_reward(x)
        agent.update(action, x, reward)
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
