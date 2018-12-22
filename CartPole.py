"""
This file contains my resolution for the Open.Ai CartPole-V1 challenge.
Enjoy and Learn

Th3 0bservator
December, 22, 2018
"""


import gym
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define Game Commands
RIGHT_CMD = [0, 1]
LEFT_CMD = [1, 0]

# Define Reward Config
START_REWARD = 0
MIN_REWARD = 100

# Initialize Game Environment
env = gym.make('CartPole-v1')


def play_random_games(games=100):
    """
    Play Random Games to Get Some Observations
    :param games:
    :return:
    """

    # Storage for All Games Movements
    all_movements = []

    for episode in range(games):

        # Reset Game Reward
        episode_reward = 0

        # Define Storage for Current Game Data
        current_game_data = []

        # Reset Game Environment
        env.reset()

        # Get First Random Movement
        action = env.action_space.sample()

        while True:

            # Play
            observation, reward, done, info = env.step(action)

            # Get Random Action (On Real, its get a "Next" movement to compensate Previous Movement)
            action = env.action_space.sample()

            # Store Observation Data and Action Taken
            current_game_data.append(
                np.hstack((observation, LEFT_CMD if action == 0 else RIGHT_CMD))
            )

            if done:
                break

            # Compute Reward
            episode_reward += reward

        # Save All Data (Only for the Best Games)
        if episode_reward >= MIN_REWARD:
            print('.', end='')
            all_movements.extend(current_game_data)

    # Create DataFrame
    dataframe = pd.DataFrame(
        all_movements,
        columns=['cart_position', 'cart_velocity', 'pole_angle', 'pole_velocity_at_tip', 'action_to_left', 'action_to_right']
    )

    # Convert Action Columns to Integer
    dataframe['action_to_left'] = dataframe['action_to_left'].astype(int)
    dataframe['action_to_right'] = dataframe['action_to_right'].astype(int)

    return dataframe


def generate_ml(dataframe):
    """
    Train and Generate NN Model
    :param dataframe:
    :return:
    """

    # Define Neural Network Topology
    model = Sequential()
    model.add(Dense(64, input_dim=4, activation='relu'))
    # model.add(Dense(128,  activation='relu'))
    # model.add(Dense(128,  activation='relu'))
    model.add(Dense(64,  activation='relu'))
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(2,  activation='sigmoid'))

    # Compile Neural Network
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Fit Model with Data
    model.fit(
        dataframe[['cart_position', 'cart_velocity', 'pole_angle', 'pole_velocity_at_tip']],
        dataframe[['action_to_left', 'action_to_right']],
        epochs=20
    )

    return model


def play_game(ml_model, games=100):
    """
    Play te Game
    :param ml_model:
    :param games:
    :return:
    """

    for i_episode in range(games):

        # Define Reward Var
        episode_reward = 0

        # Reset Env for the Game
        observation = env.reset()

        while True:
            render = env.render()

            # Predict Next Movement
            current_action_pred = ml_model.predict(observation.reshape(1, 4))

            # Define Movement
            current_action = np.argmax(current_action_pred)

            # Make Movement
            observation, reward, done, info = env.step(current_action)

            if done:
                episode_reward += 1
                print(f"Episode finished after {i_episode+1} steps", end='')
                break

            episode_reward += 1

        print(f" Score = {episode_reward}")


print("[+] Playing Random Games")
df = play_random_games(games=10000)

print("[+] Training NN Model")
ml_model = generate_ml(df)

# from ann_visualizer.visualize import ann_viz;
# ann_viz(ml_model , title="My first neural network")

print("[+] Playing Games with NN")
play_game(ml_model=ml_model, games=100)

