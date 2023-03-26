#Author: Nikhil Malhotra
#Date: 03/25/2023
#Description: This file contains the code to train a reinforcement learning agent to play super mario bros using stable baselines.

import os
import gym
import gym_super_mario_bros
from stable_baselines3.common.vec_env import DummyVecEnv , VecFrameStack
from gym.wrappers import FrameStack , GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT 
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO

#Make the environment first. We use the nes_py wrapper to make the environment compatible with stable baselines
env = gym_super_mario_bros.make('SuperMarioBros-v0')
#print(env.action_space)
#print(env.observation_space.shape)

#Since the action space is huge we embed this space with JoypadSpace wrapper to make it compatible with stable baselines
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#print("After wrapping environment with Joypad space")
#print(env.action_space)
#print(env.observation_space.shape)

# This method is just used to check the environment
def check_environment():
    done = False
    obs=env.reset()
    for step in range(1000):
        env.render()
        #Take next randome action 
        obs,reward,done,info = env.step(env.action_space.sample())
        if done:
            break
    env.close()

# Use this method to framestack the environment. This is done to make the environment more stable
def make_stacked_environment():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4 , channels_order='last' )
    print("State Shape: " ,env.reset().shape)
    print("########################")
    return env

#Train the model
def train_environment(env):
    #Make log path
    model_save_path = os.path.join('saved_models', 'PPO_Mario')
    log_path = os.path.join('training_logs', 'PPO_Mario')
    #Create the model
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
    #Train the model
    model.learn(total_timesteps=1000000)
    #Save the model
    model.save(model_save_path)

#Run the trained model
def run_trained_environment(env):
    #Load the model
    model = PPO.load('saved_models/PPO_Mario.zip')
    done = False
    obs=env.reset()
    for step in range(10000):
        env.render()
        #Take next random action 
        next_action , _  = model.predict(obs)
        obs,reward,done,info= env.step(next_action)
        if done:
            break
    env.close()


#Main method
if __name__ == "__main__":
    #check_environment()
    stacked_env = make_stacked_environment()
    train_environment(stacked_env)
    #run_trained_environment(stacked_env)