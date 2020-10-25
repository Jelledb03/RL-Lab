import ray
import os
import pandas
from ray import tune
from ray.rllib.models import ModelCatalog

from dqn import DQNTrainer, DQNModel

if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    tune.run(
        DQNTrainer,
        # checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"episode_reward_mean": 250},
        config={
            "num_gpus": 0,
            "num_workers": 2,
            "framework": "torch",
            "rollout_fragment_length": 50,
            "env": "CartPole-v1",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 0.001,
            #"lr": tune.grid_search([0.001, 0.005, 0.0005]),
            # epsilon greedy
            "epsilon": 0.8,
            #"epsilon": tune.grid_search([0.3, 0.5, 0.8]),
            "eps_decay": 0.7,
            # gamma is the discount value
            "gamma": 0.7,
            #"gamma": tune.grid_search([0.7, 0.8, 0.95]),
            "buffer_size": 40000,
            "buffer_slice_size": 10,
            #"buffer_slice_size": tune.grid_search([10, 100, 1000]),

            "dqn_model": {
                "custom_model": "DQNModel",
                "custom_model_config": {
                },  # extra options to pass to your model
            }
        }
    )
