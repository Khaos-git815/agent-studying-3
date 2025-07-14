print("Very first line executed")
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from DQN_model import DQN
from utils import ReplayBuffer
from env import ContinuousMazeEnv

print("Script started")
try:
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    # Device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Environment
    env = ContinuousMazeEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # DQN models
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    buffer = ReplayBuffer()
    gamma = 0.99
    epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay = 0.999
    episodes = 500
    batch_size = 128
    update_freq = 10
    rewards = []
    # --- Training Loop ---
    DEBUG_EPISODES = 3
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_num = 0
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                q_vals = None
            else:
                with torch.no_grad():
                    q_vals = model(torch.FloatTensor(state).to(device))
                    action = q_vals.argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            # Debug printouts for first DEBUG_EPISODES
            if episode < DEBUG_EPISODES:
                print(f"[Ep {episode+1} Step {step_num}] State: {state}, Action: {action}, Reward: {reward}")
                if q_vals is not None:
                    print(f"[Ep {episode+1} Step {step_num}] Q-values: {q_vals.cpu().numpy()}")
            step_num += 1
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(buffer) > batch_size:
                states, actions, rewards_batch, next_states, dones = buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_batch = torch.FloatTensor(rewards_batch).to(device)
                dones = torch.FloatTensor(dones).to(device)
                q_values = model(states).gather(1, actions).squeeze(-1)
                max_next_q = target_model(next_states).max(1)[0]
                target_q = rewards_batch + gamma * max_next_q * (1 - dones)
                loss = criterion(q_values, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % update_freq == 0:
            target_model.load_state_dict(model.state_dict())
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    # --- Save Model and Rewards ---
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/dqn_model.pth")
    np.save("results/rewards.npy", np.array(rewards))
    # --- Plot Training Reward ---
    plt.plot(rewards)
    plt.title("Training Reward over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("results/training_plot.png")
    plt.show()
    # --- Evaluation: 100 Consecutive Successes at epsilon=0.1 ---
    print("\nEvaluating agent for 100 consecutive successes at epsilon=0.1...")
    eval_episodes = 100
    successes = 0
    epsilon_eval = 0.1
    for ep in range(eval_episodes):
        state, _ = env.reset()
        # Force agent to start at (0.1, 0.5)
        env.agent_pos = np.array([0.1, 0.5], dtype=np.float32)
        state = env.agent_pos.copy()
        done = False
        steps = 0
        while not done and steps < 500:
            if np.random.rand() < epsilon_eval:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(torch.FloatTensor(state).to(device)).argmax().item()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            steps += 1
        # Check if reached goal
        if np.linalg.norm(env.agent_pos - env.goal_pos) <= env.goal_radius:
            successes += 1
        else:
            print(f"Failed at episode {ep+1}. Consecutive successes: {successes}")
            break
    # --- Save evaluation results to file ---
    with open("results/eval_results.txt", "w") as f:
        if successes == eval_episodes:
            f.write("Agent successfully reached the goal 100 times in a row!\n")
            f.write(f"Consecutive successes: {successes}\n")
        else:
            f.write(f"Agent did not reach the goal 100 times in a row. Max consecutive: {successes}\n")
    # --- Visualize Agent's Path After Training ---
    print("\nVisualizing agent's path for 3 episodes after training...")
    render_env = ContinuousMazeEnv(render_mode="human")
    for ep in range(3):
        state, _ = render_env.reset()
        render_env.agent_pos = np.array([0.1, 0.5], dtype=np.float32)
        state = render_env.agent_pos.copy()
        done = False
        steps = 0
        while not done and steps < 500:
            render_env.render()
            with torch.no_grad():
                action = model(torch.FloatTensor(state).to(device)).argmax().item()
            next_state, reward, done, _, _ = render_env.step(action)
            state = next_state
            steps += 1
        print(f"Visualization episode {ep+1} finished in {steps} steps.")
    render_env.close()
except Exception as e:
    print(f"Exception occurred: {e}")
    import traceback
    traceback.print_exc()
