
import os
import random
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
from gym import spaces
from mujoco_py import load_model_from_path, MjSim, MjViewer
import MujocoCollision as mjcol



'''
model = load_model_from_path("xmls/Tesrt.xml")
sim = MjSim(model)

viewer = MjViewer(sim)

sim_state = sim.get_state()


while True:
    sim.set_state(sim_state)

    for i in range(800):
        for j in range(len(sim.data.ctrl)):
            sim.data.ctrl[j] = random.uniform(-1, 1)
        sim.step()
        viewer.render()

        center = sim.data.get_geom_xpos('basket1')
        dist = GetDistance_Geometry2Geometry(sim, 'basket1','head')
        b1,b2 = GetCapsuleCenterLine( GetCapsuleData(sim, 'basket1') )
        viewer.add_marker(pos=b2,label=str(dist),size=[.05, .05, .05])

    if os.getenv('TESTING') is not None:
        break
'''


class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.model = load_model_from_path("xmls/Tesrt.xml")
        self.sim = MjSim(self.model)
        self.viewer = None
        self.sim_state = self.sim.get_state()
        self.bodynames = [
            'torso1', 'head', 'uwaist', 'lwaist', 'butt',
            'right_thigh1', 'right_shin1', 'right_foot_cap1', 'right_foot_cap2', 'left_thigh1',
            'left_shin1', 'left_foot_cap1', 'left_foot_cap2', 'right_uarm1', 'right_larm',
            'right_hand', 'left_uarm1', 'left_larm', 'left_hand'
        ]

        ones_act = np.ones(len(self.sim.data.ctrl))
        ones_obs = np.ones(self._get_state().shape[0])

        self.action_space = spaces.Box(-ones_act, ones_act)
        self.observation_space = spaces.Box(-ones_obs, ones_obs)

    def _get_state(self):
        torso = []
        ret = []
        for i in range(len(self.bodynames)):
            vec = self.sim.data.get_geom_xpos(self.bodynames[i])
            if i==0:
                ret = vec
                torso = vec
            if i!=0:
                ret = np.append(ret, vec-torso)
        return ret
    
    def _get_reward(self):
        return self.sim.data.get_geom_xpos('head')[2]

    def _step(self, action):
        for i in range(len(action)):
            self.sim.data.ctrl[i] = action[i] * 0.5
        
        self.sim.step()
        self.sim.step()
        return self._get_state(),self._get_reward(),False,{}

    def _reset(self):
        self.sim.set_state(self.sim_state)
        return self._get_state()

    def _render(self, mode = 'human', close = False):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        self.viewer.render()

#env = gym.make('CartPole-v0')
#n_actions = env.action_space.n

env = MyEnv()
n_actions = len(env.action_space.high)

obs_size = env.observation_space.shape[0]

#q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
#    obs_size, n_actions,
#    n_hidden_layers=3, n_hidden_channels=50)


q_func = chainerrl.q_functions.FCQuadraticStateQFunction(
    obs_size, n_actions,
    n_hidden_layers=3, n_hidden_channels=50,
    action_space = env.action_space, scale_mu=True)

optimizer = chainer.optimizers.Adam(eps=1e-3)
optimizer.setup(q_func)

gamma = 0.9

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=0.5, end_epsilon=0.15, decay_steps=1000000, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=1)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    update_interval=1,replay_start_size=500,
    target_update_interval=100, phi=phi)

max_episode_len = 600
i = 0
while True:
    obs = env.reset()
    i=i+1
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        env.render()
    if i % 1 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    #env.render()
    agent.stop_episode_and_train(obs, reward, done)
print('Finished.')



