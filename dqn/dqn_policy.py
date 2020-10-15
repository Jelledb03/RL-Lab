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
        #print(self.num_outputs)
        self.config = config

        self.lr = self.config["lr"] # Extra options need to be added in dqn.py

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
        #for obs in obs_batch:
            #print(obs)
        q_values = self.dqn_model(obs_batch_t)
        #print(q_values)
        #Nog E-greedy exploration aan toevoegen
        action = torch.argmax(q_values).item()
        #print(action)


        #self.action_space.sample() for _ in obs_batch
        return [action], [], {}

    def learn_on_batch(self, samples):
        # Trainer function
        #print(samples)
        print(samples["dones"])
        print(samples["rewards"])
        obs_batch_t = torch.tensor(np.array(samples["obs"])).type(torch.FloatTensor)
        actions_batch_t = torch.tensor(np.array(samples["actions"])).type(torch.FloatTensor)
        rewards_batch_t = torch.tensor(np.array(samples["rewards"])).type(torch.FloatTensor)
        next_obs_batch_t = torch.tensor(np.array(samples["new_obs"])).type(torch.FloatTensor)
        dones_batch_t = torch.tensor(np.array(samples["dones"]))
        for obs, action, reward, next_obs in zip(obs_batch_t, actions_batch_t, rewards_batch_t, next_obs_batch_t):
            experience = [obs, action, reward, next_obs]
            self.experience_buffer.append(experience)

        print(len(self.experience_buffer))
        #dequeue van collection
        print(self.experience_buffer)

        #Run through every episode again to get Q Value and update it
        #While dones is false episode is running
        for obs, action, reward, next_obs, done in zip(obs_batch_t, actions_batch_t, rewards_batch_t, next_obs_batch_t, dones_batch_t):
            curr_episode = []
            print(done)
            if not done:
                #Hier moet dan de volgende max Q value wel gevonden worden en geupdate worden
                #Dit zijn current q_values
                curr_q_values = self.dqn_model(obs)
                
                print("hello")

            else:
                #Als hem done is, is er geen future q value meer. Dus formule van Q wordt dan gewoon Q(s,a) + aplha*R
                print(self.bla)


        print(self.blabla)
        return {"learner_stats": {"loss": 0}}

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
