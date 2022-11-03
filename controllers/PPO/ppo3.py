import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes


def train(hyperparams, env, total_episodes, default_policy=False, model_name='ppo_trained_model'):

    if default_policy:
        model = PPO('MlpPolicy', env, verbose=0)
    else:
        # Custom MLP policy of two layers of size 32 each with Relu activation function
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[hyperparams['hidden_neurons']] * hyperparams['num_layers'])

        # Create the agent
        model = PPO("MlpPolicy", env,
                    batch_size=int(hyperparams['batch_size']),
                    n_epochs=hyperparams['n_epochs'],
                    learning_rate=hyperparams['learning_rate'],
                    gamma=hyperparams['gamma'],
                    clip_range=hyperparams['clip_range'],
                    n_steps=hyperparams['n_steps'],
                    policy_kwargs=policy_kwargs, verbose=0)

    print('training...')

    # Stops training when the model reaches the maximum number of episodes
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=total_episodes, verbose=1)

    # Train the agent
    model.learn(total_timesteps=int(100*total_episodes), callback=callback_max_episodes)

    model.save(model_name)



