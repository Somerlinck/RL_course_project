import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from wimblepong.PG_agent_old import Agent, Policy
import pandas as pd
import matplotlib.pyplot as plt


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_episodes=5000):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    y_arena_res, x_arena_res, _ = env.observation_space.shape
    observation_space_dim = y_arena_res * x_arena_res
    action_space_dim = env.action_space.n

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()
        previous_observation = None
        # Loop until the episode is over
        while not done:
            # Get action from the agent
            state = abs(2*observation.dot([0.07, 0.72, 0.21]) - previous_observation.dot([0.07, 0.72, 0.21]))\
                if previous_observation is not None else np.zeros((y_arena_res, x_arena_res))
            #state = state.ravel()
            #plt.imshow(state)
            #plt.show()
            action, log_prob = agent.get_action(state)
            previous_observation = observation
            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action)
            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(state, log_prob, reward)

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if print_things:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # Let the agent do its magic (update the policy)
        agent.episode_finished()

    # Training is finished - plot rewards
    if print_things:
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history")
        plt.show()
        print("Training finished.")
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["PG"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % (env_name, train_run_id))
    return data


# Function to test a trained policy
def test(env_name, episodes, params, render):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    y_arena_res, x_arena_res, _ = env.observation_space.shape
    observation_space_dim = y_arena_res * x_arena_res
    action_space_dim = env.action_space.n

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(params)
    agent = Agent(policy)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        previous_observation = None
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            state = abs(2*observation.dot([0.07, 0.72, 0.21]) - previous_observation.dot([0.07, 0.72, 0.21])) \
                if previous_observation is not None else np.zeros((y_arena_res, x_arena_res))
            state = state.ravel()
            previous_observation = observation

            action, _ = agent.get_action(state, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default="model_PG105000.mdl", help="Model to be tested")
    parser.add_argument("--env", type=str, default="WimblepongVisualSimpleAI-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--housekeeping", action="store_true",
                        help="Plot, player and ball positions and velocities at the end of each episode")
    parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
    parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(args.env, train_episodes=args.train_episodes)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, True) #args.render_test)
