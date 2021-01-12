from policy import Policy
import numpy as np
import pickle
import gym
import os


class Visualizer(object):
    def __init__(self, game, network, train_directory):
        self.game = game
        env_name = '%sNoFrameskip-v4' % game
        env = gym.make(env_name)
        env = gym.wrappers.Monitor(env, 'monitor_%s' % game, mode='evaluation', force=True)

        vb_file = "230_vb.npy"
        vb = np.load(vb_file)
        parameters_file = "params230"

        self.policy = Policy(env, network, "elu")


        with open(parameters_file, 'rb') as f:
            parameters = pickle.load(f)['parameters']

        self.policy.set_parameters(parameters)
        self.policy.set_vb(vb)

    def play_game(self):
        return self.policy.rollout(render=True)


if __name__ == '__main__':
    vis = Visualizer('Qbert', 'Nature', train_directory='networks')
    rew = vis.play_game()
    
         
