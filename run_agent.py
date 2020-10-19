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
            "num_workers": 1,
            "framework": "torch",
            "rollout_fragment_length": 50,
            "env": "CartPole-v1",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 0.001,
            # "lr": tune.grid_search([0.001, 0.0001, 0.00001]),
            # epsilon greedy
            "epsilon": 0.3,
            "eps_decay": 0.99,
            # gamma is the discount value
            "gamma": 0.7,
            # "gamma": tune.grid_search([0.7, 1, 0.9]),
            "buffer_size": 4000,

            "dqn_model": {
                "custom_model": "DQNModel",
                "custom_model_config": {
                },  # extra options to pass to your model
            }
        }
    )
