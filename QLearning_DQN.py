
from helper import plot_and_save_comparison, plot_and_save_comparison_error_bars
from helper import should_start_recording, should_stop_recording
from helper import plot_and_save_comparison_average
from helper import get_parameters, build_bins
from cartpole_class import CustomCartPoleEnv
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import time
import gym



########################### GLOBAL VARIABLES ####################################



episodes = 1000

learning_rate, discount_factor, epsilon = get_parameters()

video_path_QL = "videos/QL"

video_path_DQN = "videos/DQN"

number_of_evaluation_episodes = 20

force_mags = [5, 10, 15]
x_thresholds = [0.6, 1.2, 2.4]



########################### Q-Learning BEGIN ####################################



def discretize_state(state, bins, num_bins):
    discretized = []
    for i in range(len(state)):
        # Extract each component of the state as a scalar
        state_value = state[i]
        min_bin = bins[i][0]
        max_bin = bins[i][-1]

        # Normalize the state component within its bin range
        scale = (state_value - min_bin) / (max_bin - min_bin)

        # Clamp the scale to [0,1] to avoid out-of-bounds issues
        scale = max(0, min(scale, 1)) 

        # Calculate the discretized index
        scaled_index = scale * (num_bins - 1)
        index = int(round(scaled_index))
        index = max(0, min(index, num_bins - 1))
        
        discretized.append(index)
    
    return tuple(discretized)


def policy(state, Q_table):
    
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q_table[state])  # Exploit


def update_Q_table(state, action, reward, new_state, Q_table):
    best_next_action = np.argmax(Q_table[new_state]) 
    td_target = reward + discount_factor * Q_table[new_state][best_next_action]
    td_delta = td_target - Q_table[state][action]
    Q_table[state][action] += learning_rate * td_delta



########################### Q-Learning END ######################################



########################### Deep Q-Networks BEGIN ###############################



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            target = reward
            if not done:
                # Bellman equation
                target = (reward + self.gamma *
                          torch.max(self.model(next_state)).item())
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



########################### Deep Q-Networks END ###############################



########################### Q-Learning TRAINING BEGIN #########################        



# Training the Agent
def train_QL():
    # Increase the number of episodes to increase training time.
    for episode in range(episodes):
        total_reward = 0
        steps = 0
        # Reset environment before getting current_state
        current_state, _ = env.reset()
        # Discretize the initial state
        current_state = discretize_state(current_state, bins, num_bins)
        done = False
        

        while not done:
            # Select action based on policy
            action = policy(current_state, Q_table)
            # Execute action in environment
            next_state, reward, done, _, _ = env.step(action)
            # Discretize the next state
            next_state = discretize_state(next_state, bins, num_bins)
            # Update Q-table
            update_Q_table(current_state, action, reward, next_state, Q_table)
            # Move to the next state
            current_state = next_state

            total_reward += reward
            steps += 1

            if done:
                cumulative_rewards_QL.append(total_reward)
                episode_lengths_QL.append(steps)
                #print(f"Episode QL: {episode+1}, Total Reward: {total_reward}, Steps: {steps}")
                break



########################### Q-Learning TRAINING END ###########################



########################### Deep Q-Networks TRAINING BEGINNING ################



def train_DQN():

    # Training loop
    for episode in range(episodes):
        total_reward = 0
        steps = 0
        state, _ = env.reset()
        state = np.reshape(state, [1, 4])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                cumulative_rewards_DQN.append(total_reward)
                episode_lengths_DQN.append(steps)
                #print(f"Episode DQN: {episode+1}, Total Reward: {total_reward}, Steps: {steps}")
                #print(f"Episode DQN: {episode}/{1000}, score: {time}, e: {agent.epsilon:.2f}")
                break

        if len(agent.memory) > 32:
            agent.replay(32)

    env.close()



########################### Deep Q-Networks TRAINING END ######################



########################### EVALUATION BEGIN ##################################



def evaluate_and_record(agent, env, num_episodes, video_path, name_prefix):
    cumulative_rewards = []
    env = RecordVideo(env, video_folder=video_path, name_prefix=name_prefix, video_length=1000)
    env.reset()
    try:
        for episode in range(num_episodes):
            total_reward = 0
            state, _ = env.reset()
            done = False
            while not done:
                if isinstance(agent, DQNAgent):
                    state = np.reshape(state, [1, 4])
                    action = agent.act(state)
                else:
                    discretized_state = discretize_state(state, bins, num_bins)
                    action = policy(discretized_state, Q_table)
                
                state, reward, done, _, _ = env.step(action)
                total_reward += reward
                if done:
                    cumulative_rewards.append(total_reward)
                    break
    finally:
        env.close()

    return cumulative_rewards

counter = 0

for force_mag in force_mags:

    velocity = force_mag

    for x_threshold in x_thresholds:

        master_cumulative_rewards_QL = []

        master_episode_lengths_QL = []

        master_cumulative_rewards_DQN = []

        master_episode_lengths_DQN = []

        master_cumulative_rewards_eval_QL = []

        master_cumulative_rewards_eval_DQN = []

        track_width = x_threshold

        counter +=1

        print('counter: ' + str(counter) + ', velocity: ' + str(velocity) + ', track_width: ' + str(track_width))

        for z in range(0, 5):

            cumulative_rewards_QL = []

            episode_lengths_QL = []

            cumulative_rewards_DQN = []

            episode_lengths_DQN = []

            cumulative_rewards_eval_QL = []

            cumulative_rewards_eval_DQN = []

            num_bins, bins = build_bins()

            env = CustomCartPoleEnv(render_mode="rgb_array")
            #env = gym.make('CartPole-v1', render_mode="rgb_array")

            Q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, env.action_space.n))

            agent = DQNAgent(state_dim=4, action_dim=2)

            train_QL()

            train_DQN()

            master_cumulative_rewards_QL.append(cumulative_rewards_QL)

            master_episode_lengths_QL.append(episode_lengths_QL)

            master_cumulative_rewards_DQN.append(cumulative_rewards_DQN)

            master_episode_lengths_DQN.append(episode_lengths_DQN)

            # Q_table is trained Q-learning model
            env = CustomCartPoleEnv(render_mode="rgb_array")
            #env = gym.make('CartPole-v1', render_mode="rgb_array")
            cumulative_rewards_eval_QL = evaluate_and_record(Q_table, env, num_episodes=number_of_evaluation_episodes, video_path=video_path_QL, name_prefix="QL_Eval")

            master_cumulative_rewards_eval_QL.append(cumulative_rewards_eval_QL)

            # agent is trained DQN model
            env = CustomCartPoleEnv(render_mode="rgb_array")
            #env = gym.make('CartPole-v1', render_mode="rgb_array")
            cumulative_rewards_eval_DQN = evaluate_and_record(agent, env, num_episodes=number_of_evaluation_episodes, video_path=video_path_DQN, name_prefix="DQN_Eval")
            
            master_cumulative_rewards_eval_DQN.append(cumulative_rewards_eval_DQN)

        file_name_1 = 'CR_Training' + '_vel_' + str(velocity) + '_track_' + str(track_width)
        file_name_2 = 'EL_Training' + '_vel_' + str(velocity) + '_track_' + str(track_width)
        file_name_3 = 'CR_Evaluation' + '_vel_' + str(velocity) + '_track_' + str(track_width)
        
        plot_and_save_comparison_average(cumulative_rewards_QL, cumulative_rewards_DQN, 'Cumulative Rewards - Training Averages (5 trainings)', 'charts', 'Average Reward Per Episode', file_name_1)
        plot_and_save_comparison_average(episode_lengths_QL, episode_lengths_DQN, 'Episode Lengths - Training Averages (5 trainings)', 'charts', 'Average Episode Length', file_name_2)
        plot_and_save_comparison_average(cumulative_rewards_eval_QL, cumulative_rewards_eval_DQN, 'Cumulative Rewards - Evaluation Averages (5 evaluations)', 'charts', 'Average Reward Per Episode', file_name_3)


"""

plot_and_save_comparison_error_bars(master_cumulative_rewards_QL, master_cumulative_rewards_DQN, 'Cumulative Rewards - Training', 'charts', 'Total Reward Per Episode', 'CR_Training')
plot_and_save_comparison_error_bars(master_episode_lengths_QL, master_episode_lengths_DQN, 'Episode Lengths - Training', 'charts', 'Episode Length', 'EL_Training')
plot_and_save_comparison_error_bars(master_cumulative_rewards_eval_QL, master_cumulative_rewards_eval_DQN, 'Cumulative Rewards - Evaluation', 'charts', 'Total Reward Per Episode', 'CR_Evaluation')


print('QL cumulative_rewards: ', cumulative_rewards_QL)
print(len(cumulative_rewards_QL))
print('QL episode_lengths: ', episode_lengths_QL)
print(len(episode_lengths_QL))
print('QL EVAL: ', cumulative_rewards_eval_QL)

print('\n')

print('DQN cumulative_rewards: ', cumulative_rewards_DQN)
print(len(cumulative_rewards_DQN))
print('DQN episode_lengths: ', episode_lengths_DQN)
print(len(episode_lengths_DQN))
print('DQN EVAL: ', cumulative_rewards_eval_DQN)


How can I control these things, or test different variations using python gym library? What are the effects of parameters of the system? How fast you should move the
cart (the voltage you apply to move the cart)? Is it better to accelerate faster? Is
there any optimal point for the speed of the cart? What are the effects of friction
coefficients? What is the effect of the limitation along the horizontal axis?  



# Plotting and saving the graphs
plot_and_save_comparison(cumulative_rewards_QL, cumulative_rewards_DQN, 'Cumulative Rewards - Training', 'charts', 'Total Reward Per Episode', 'CR_Training')
plot_and_save_comparison(episode_lengths_QL, episode_lengths_DQN, 'Episode Lengths - Training', 'charts', 'Episode Length', 'EL_Training')
plot_and_save_comparison(cumulative_rewards_eval_QL, cumulative_rewards_eval_DQN, 'Cumulative Rewards - Evaluation', 'charts', 'Total Reward Per Episode', 'CR_Evaluation')
"""

########################### EVALUATION END ####################################

