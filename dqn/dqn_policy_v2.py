import random

import torch
import numpy as np
import collections
from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog


class DQNPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_outputs = self.action_space.n
        # print(self.num_outputs)
        self.config = config

        self.lr = self.config["lr"]  # Extra options need to be added in dqn.py

        # Discount value
        self.gamma = self.config["gamma"]

        # Epsilon value
        self.epsilon = self.config["epsilon"]
        self.eps_decay = self.config["eps_decay"]

        # GPU settings
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Experience buffer
        self.buffer_size = self.config["buffer_size"]
        self.experience_buffer = collections.deque(maxlen=self.buffer_size)

        self.dqn_model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=self.num_outputs,
            name="DQNModel",
            model_config=self.config["dqn_model"],
            framework="torch",
        ).to(self.device, non_blocking=True)

        # Define network optimizer
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.lr)

        # Define loss calculator
        self.loss_calculator = torch.nn.MSELoss()

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        # Worker function
        obs_batch_t = torch.tensor(obs_batch).type(torch.FloatTensor)
        # for obs in obs_batch:
        # print(obs)
        if random.random() < self.epsilon:
            action = self.action_space.sample()
            print("random")
        else:
            q_values = self.dqn_model(obs_batch_t)
            action = torch.argmax(q_values).item()
            print("niet random")
        print(action)
        # print(q_values)
        # print(action)
        # Gaat epsilon laten decayen totdat deze kleiner dan 0.01 wordt, omdat we anders nooit meer random een actie gaan kiezen
        self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
        print(self.epsilon)
        print(self.bla)
        # self.action_space.sample() for _ in obs_batch
        return [action], [], {}

    def learn_on_batch(self, samples):
        # Trainer function
        # print(samples)
        # print(samples["dones"])
        # print(samples["rewards"])
        obs_batch_t = torch.tensor(np.array(samples["obs"])).type(torch.FloatTensor)
        actions_batch_t = torch.tensor(np.array(samples["actions"])).type(torch.FloatTensor)
        rewards_batch_t = torch.tensor(np.array(samples["rewards"])).type(torch.FloatTensor)
        next_obs_batch_t = torch.tensor(np.array(samples["new_obs"])).type(torch.FloatTensor)
        dones_batch_t = torch.tensor(np.array(samples["dones"]))
        for obs, action, reward, next_obs in zip(obs_batch_t, actions_batch_t, rewards_batch_t, next_obs_batch_t):
            experience = [obs, action, reward, next_obs]
            self.experience_buffer.append(experience)

        # print(len(self.experience_buffer))
        # dequeue van collection
        # print(self.experience_buffer)

        # Calculate the amount of q_values calculated
        number_of_q_values = len(dones_batch_t)
        curr_q_values = torch.empty(number_of_q_values, requires_grad=False)
        better_q_values = torch.empty(number_of_q_values, requires_grad=False)

        # Tensor counter
        counter = 0
        # Run through every episode again to get Q Value and update it
        # In this version I will calculate the loss with only the used q values
        for obs, action, reward, next_obs, done in zip(obs_batch_t, actions_batch_t, rewards_batch_t, next_obs_batch_t,
                                                       dones_batch_t):
            # While dones is false episode is running
            if not done:
                # Calculates both q_values
                both_q_values = self.dqn_model(obs)
                curr_q_value = torch.max(both_q_values)
                next_best_q_value = torch.max(self.dqn_model(next_obs))
                # print(curr_q_value)
                better_q_value = reward + self.gamma * next_best_q_value
                # print(better_q_value)
                curr_q_values[counter] = curr_q_value
                better_q_values[counter] = better_q_value
                counter += 1

            else:
                # if done, there will be no more further q value. So formula becomes Q(s,a) + aplha*R
                # Calculates both q_values
                both_q_values = self.dqn_model(obs)
                curr_q_value = torch.max(both_q_values)
                better_q_value = reward
                # print(better_q_value)
                curr_q_values[counter] = curr_q_value
                better_q_values[counter] = better_q_value
                counter += 1

        #Have to always include a couple of entries from experience buffer in here?


        # Have to check q_values and better q_values here
        # print(curr_q_values)
        # print(better_q_values)
        loss = self.loss_calculator(curr_q_values, better_q_values)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(self.bla)
        return {"learner_stats": {"default_policy/loss": loss.detach()}}

    def get_weights(self):
        # Trainer function
        weights = {}
        weights["dqn_model"] = self.dqn_model.cpu().state_dict()
        self.dqn_model.to(self.device, non_blocking=False)
        return weights

    def set_weights(self, weights):
        # Worker function
        if "dqn_model" in weights:
            self.dqn_model.load_state_dict(weights["dqn_model"], strict=True)
            self.dqn_model.to(self.device, non_blocking=False)
