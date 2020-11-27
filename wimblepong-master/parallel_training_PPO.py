import torch
import gym
import numpy as np
import argparse
from wimblepong.PPO_agent import Agent, ActorCritic
from parallel_env import ParallelEnvs


def transform_observations(previous_observations, observations, x_arena_res=200, y_arena_res=200):
    nb_proc = observations.shape[0]
    res = np.zeros((nb_proc, x_arena_res * y_arena_res))
    for i in range(nb_proc):
        mat = abs(2*observations[i].dot([0.07, 0.72, 0.21]) - previous_observations[i].dot([0.07, 0.72, 0.21])) \
            if previous_observations is not None else np.zeros((y_arena_res, x_arena_res))
        res[i] = mat.ravel()
    return res


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_timesteps=500000, update_steps=50, load=False):
    # Create a Gym environment
    # This creates 64 parallel envs running in 8 processes (8 envs each)
    env = ParallelEnvs(env_name, processes=8, envs_per_process=8)

    # Get dimensionalities of actions and observations
    y_arena_res, x_arena_res, _ = env.observation_space.shape
    observation_space_dim = y_arena_res * x_arena_res
    action_space_dim = env.action_space.n

    # Instantiate agent and its policy
    AC_old = ActorCritic(observation_space_dim, action_space_dim)
    AC = ActorCritic(observation_space_dim, action_space_dim)

    if load :
        AC.load_state_dict(torch.load('Model/modelPG_last.mdl'))

    agent = Agent(AC_old, AC)


    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    # Reset the environment and observe the initial state
    observations = env.reset()
    previous_observations = None
    next_states = transform_observations(previous_observations, observations, x_arena_res, y_arena_res)

    avg_rewards = None

    # Loop forever
    for timestep in range(train_timesteps):
        states = next_states
        # Get action from the agent
        actions, log_probs = agent.get_action(states)
        previous_observations = observations

        # Perform the action on the environment, get new state and reward
        observations, rewards, dones, info = env.step(actions.detach().numpy())
        next_states = transform_observations(previous_observations, observations, x_arena_res, y_arena_res)

        if (rewards != 0).sum() != 0:
            if avg_rewards is None:
                avg_rewards = rewards.sum() / (rewards != 0).sum()
            else:
                avg_rewards = (avg_rewards + rewards.sum() / (rewards != 0).sum()) / 2

        for j in range(len(rewards)):
            agent.store_outcome(states[j], log_probs[j], actions[j], rewards[j], next_states[j], dones[j])

        if timestep % update_steps == update_steps-1:
            print(f"Update @ step {timestep}, avg_rewards = {avg_rewards}")
            agent.update(0)
            avg_rewards = None

        if timestep % 10000 == 0:
          model_name = "modelPG_" + str(timestep)
          path = f'Model/{model_name}.mdl'
          print(f'Saving {model_name} model...')
          torch.save(agent.policy.state_dict(), path)
          print(f'{model_name} saved successfully.')


# Function to test a trained policy
def test(env_name, episodes, params, render):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    y_arena_res, x_arena_res, _ = env.observation_space.shape
    observation_space_dim = y_arena_res * x_arena_res
    action_space_dim = env.action_space.n

    # Instantiate agent and its policy
    AC_old = ActorCritic(observation_space_dim, action_space_dim)
    AC = ActorCritic(observation_space_dim, action_space_dim)
    agent = Agent(AC_old, AC)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0", help="Environment to use")
    parser.add_argument("--train_timesteps", type=int, default=1000, help="Number of timesteps to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--load", action='store_true', help="load a saved model")
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(args.env, train_timesteps=args.train_timesteps)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, args.render_test)

