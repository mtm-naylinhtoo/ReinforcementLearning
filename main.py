# pip install gym==0.25.2
# pip install tensorflow keras-rl2
# pip install gym[classic_control]
# changed in rl.callbacks "from keras import __version__ as KERAS_VERSION"
# path is "AppData\Local\Programs\Python\Python39\Lib\site-packages\rl\callbacks.py"

import gym
import random
import numpy as np
import tensorflow as tf
# from keras import __version__
# tf.keras.__version__ = __version__

from keras.layers import Dense, Flatten
from keras.models import Sequential
# from keras.optimizers import Adam
from keras.optimizers.legacy import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=0.01)
    return dqn
agent = build_agent(model, actions)

agent.compile(Adam(lr=0.01), metrics=['mae'])
agent.fit(env, nb_steps=5000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()

