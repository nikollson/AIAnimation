"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.
The Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow r1.3
gym 0.8.0
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

import os
import random
from gym import spaces
from mujoco_py import load_model_from_path, MjSim, MjViewer
import MujocoCollision as mjcol

class MyEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	stepcount = 0
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
		#if self.stepcount%20==0:
		#	print(action)
		for i in range(len(action)):
			self.sim.data.ctrl[i] = action[i] * 0.25
		self.stepcount += 2
		self.sim.step()
		self.sim.step()

		isEnd = False
		if(self.stepcount >= 900):
			isEnd = True
		return self._get_state(),self._get_reward(),isEnd,{}

	def _reset(self):
		self.sim.set_state(self.sim_state)

		for i in range(2):
		   self._step((np.random.rand(len(self.sim.data.ctrl))-0.5)*2);

		self.stepcount = 0
		return self._get_state()

	def _render(self, mode = 'human', close = False):
		if self.viewer is None:
			self.viewer = MjViewer(self.sim)
		self.viewer.render()


GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 5000
MAX_GLOBAL_EP = 40000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

#env = gym.make(GAME)
env = MyEnv()

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):
	def __init__(self, scope, globalAC=None):

		if scope == GLOBAL_NET_SCOPE:   # get global network
			with tf.variable_scope(scope):
				self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
				self.a_params, self.c_params = self._build_net(scope)[-2:]
		else:   # local net, calculate losses
			with tf.variable_scope(scope):
				self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
				self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
				self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

				mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

				td = tf.subtract(self.v_target, self.v, name='TD_error')
				with tf.name_scope('c_loss'):
					self.c_loss = tf.reduce_mean(tf.square(td))

				with tf.name_scope('wrap_a_out'):
					mu, sigma = mu * A_BOUND[1], sigma + 1e-4

				normal_dist = tf.distributions.Normal(mu, sigma)

				with tf.name_scope('a_loss'):
					log_prob = normal_dist.log_prob(self.a_his)
					exp_v = log_prob * td
					entropy = normal_dist.entropy()  # encourage exploration
					self.exp_v = ENTROPY_BETA * entropy + exp_v
					self.a_loss = tf.reduce_mean(-self.exp_v)

				with tf.name_scope('choose_a'):  # use local params to choose action
					self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])
				with tf.name_scope('local_grad'):
					self.a_grads = tf.gradients(self.a_loss, self.a_params)
					self.c_grads = tf.gradients(self.c_loss, self.c_params)

			with tf.name_scope('sync'):
				with tf.name_scope('pull'):
					self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
					self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
				with tf.name_scope('push'):
					self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
					self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

	def _build_net(self, scope):
		w_init = tf.random_normal_initializer(0., .1)
		with tf.variable_scope('actor'):
			l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
			mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
			sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
		with tf.variable_scope('critic'):
			l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
			v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
		a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
		c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
		return mu, sigma, v, a_params, c_params

	def update_global(self, feed_dict):  # run by a local
		SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

	def pull_global(self):  # run by a local
		SESS.run([self.pull_a_params_op, self.pull_c_params_op])

	def choose_action(self, s):  # run by a local
		s = s[np.newaxis, :]
		return SESS.run(self.A, {self.s: s})[0]


class Worker(object):
	def __init__(self, name, globalAC):
		self.env = MyEnv()
		self.name = name
		self.AC = ACNet(name, globalAC)

	def work(self):
		global GLOBAL_RUNNING_R, GLOBAL_EP
		total_step = 1
		buffer_s, buffer_a, buffer_r = [], [], []
		while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
			s = self.env.reset()
			ep_r = 0
			for ep_t in range(MAX_EP_STEP):
				if self.name == 'W_0':
					self.env.render()
				a = self.AC.choose_action(s)
				s_, r, done, info = self.env.step(a)
				done = True if ep_t == MAX_EP_STEP - 1 or done else False

				ep_r += r
				buffer_s.append(s)
				buffer_a.append(a)
				buffer_r.append((r+8)/8)    # normalize

				if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
					if done:
						v_s_ = 0   # terminal
					else:
						v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
					buffer_v_target = []
					for r in buffer_r[::-1]:    # reverse buffer r
						v_s_ = r + GAMMA * v_s_
						buffer_v_target.append(v_s_)
					buffer_v_target.reverse()

					buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
					feed_dict = {
						self.AC.s: buffer_s,
						self.AC.a_his: buffer_a,
						self.AC.v_target: buffer_v_target,
					}
					self.AC.update_global(feed_dict)
					buffer_s, buffer_a, buffer_r = [], [], []
					self.AC.pull_global()

				s = s_
				total_step += 1
				if done:
					if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
						GLOBAL_RUNNING_R.append(ep_r)
					else:
						GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
					print(
						self.name,
						"Ep:", GLOBAL_EP,
						"| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
						  )
					GLOBAL_EP += 1
					break

if __name__ == "__main__":
	SESS = tf.Session()

	with tf.device("/cpu:0"):
		OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
		OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
		GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
		workers = []
		# Create worker
		for i in range(N_WORKERS):
			i_name = 'W_%i' % i   # worker name
			workers.append(Worker(i_name, GLOBAL_AC))

	COORD = tf.train.Coordinator()
	SESS.run(tf.global_variables_initializer())

	if OUTPUT_GRAPH:
		if os.path.exists(LOG_DIR):
			shutil.rmtree(LOG_DIR)
		tf.summary.FileWriter(LOG_DIR, SESS.graph)

	worker_threads = []
	for worker in workers:
		job = lambda: worker.work()
		t = threading.Thread(target=job)
		t.start()
		worker_threads.append(t)
	COORD.join(worker_threads)

	plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
	plt.xlabel('step')
	plt.ylabel('Total moving reward')
	plt.show()