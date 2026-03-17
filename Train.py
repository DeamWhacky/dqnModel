import torch
import os
from flappy_bird_environment import FlappyBirdDNN
from DNNAgent import DQNAgent


def train(reward_type, num_episodes=2000):
    
    model_name = f"dqn_{reward_type}.pth"

    if os.path.exists(model_name):
        choice = input(f"{model_name} exists. Retrain and overwrite? (y/n): ").strip().lower()
        
        if choice != 'y':
            print("Training cancelled.")
            return

    # Create environment (no rendering during training)
    env = FlappyBirdDNN(reward_type, render=False)

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
    torch.save(agent.policy_net.state_dict(), model_name)
    print(f"\nTraining complete. Model saved as {model_name}")

    env.close()

def watch(reward_type, num_episodes=20):

    env = FlappyBirdDNN(reward_type, render=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    # Load trained model
    model_name = f"dqn_{reward_type}.pth"

    if not os.path.exists(model_name):
        print(f"Model {model_name} not found. Train first.")
        return

    agent.policy_net.load_state_dict(torch.load(model_name))
    agent.policy_net.eval()

    agent.epsilon = 0  # No exploration

    print("\nWatching trained agent...\n")
    
    total_rewards_sum = 0
    total_pipes_sum = 0
    
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
            
        total_rewards_sum += total_reward
        total_pipes_sum += pipes_passed
        
        print(f"Watch Episode {episode+1} | Pipes: {pipes_passed} | Reward: {total_reward:.2f}")
        
    avg_reward = total_rewards_sum / num_episodes
    avg_pipes = total_pipes_sum / num_episodes          

    print(f"\nEvaluation Results ({reward_type})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Pipes Passed: {avg_pipes:.2f}")
    
    env.close()