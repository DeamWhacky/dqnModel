import torch
from flappy_bird_environment import FlappyBirdDNN
from DNNAgent import DQNAgent


def train(num_episodes=2000):

    # Create environment (no rendering during training)
    env = FlappyBirdDNN(render=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    print("Starting training...\n")

    for episode in range(num_episodes):

        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:

            # Select action
            action = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train agent
            agent.train_step()

            state = next_state
            total_reward += reward

        print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # Save trained model
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("\nTraining complete. Model saved as dqn_model.pth")

    env.close()

def watch(num_episodes=20):

    env = FlappyBirdDNN(render=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    # Load trained model
    agent.policy_net.load_state_dict(torch.load("dqn_model.pth"))
    agent.policy_net.eval()

    agent.epsilon = 0  # No exploration

    print("\nWatching trained agent...\n")

    for episode in range(num_episodes):

        state, _ = env.reset()
        done = False
        total_reward = 0
        pipes_passed = 0

        while not done:

            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            
            pipes_passed = info.get("score", pipes_passed)

            done = terminated or truncated

            state = next_state
            total_reward += reward

        print(f"Watch Episode {episode+1} | Pipes: {pipes_passed} | Reward: {total_reward:.2f}")
    env.close()