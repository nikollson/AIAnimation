

# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
# 
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

import os
import random
from gym import spaces
from mujoco_py import load_model_from_path, MjSim, MjViewer
import MujocoCollision as mjcol


#-- constants
ENV = 'CartPole-v0'

RUN_TIME = 600
#THREADS = 8
#OPTIMIZERS = 2
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 4
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = .4
EPS_STOP  = .2
EPS_STEPS = 750000

MIN_BATCH = 512
#LEARNING_RATE = 5e-3
LEARNING_RATE = 0.1

#LOSS_V = .5			# v loss coefficient
#LOSS_ENTROPY = .01 	# entropy coefficient
LOSS_V = 10000			# v loss coefficient
LOSS_ENTROPY = 2000 	# entropy coefficient

#-- gym env

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
        if self.stepcount%20==0:
            print(action)
        for i in range(len(action)):
            self.sim.data.ctrl[i] = action[i] * 1.2
        self.stepcount += 2
        self.sim.step()
        self.sim.step()

        isEnd = False
        if(self.stepcount >= 1200):
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

#---------
class Brain:
	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self):
		self.session = tf.Session()
		K.set_session(self.session)
		K.manual_variable_initialization(True)

		self.model = self._build_model()
		self.graph = self._build_graph(self.model)

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()

		self.default_graph.finalize()	# avoid modifications

	def _build_model(self):

		l_input = Input( batch_shape=(None, NUM_STATE) )
		l_dense = Dense(200, activation='linear')(l_input)
		l_dense2 = Dense(100, activation='softmax')(l_dense)

		out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense2)
		out_value   = Dense(1, activation='softmax')(l_dense2)

		model = Model(inputs=[l_input], outputs=[out_actions, out_value])
		model._make_predict_function()	# have to initialize before threading

		return model

	def _build_graph(self, model):
		s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
		a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
		r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
		
		p, v = model(s_t)

		log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
		advantage = r_t - v

		loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
		loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
		entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

		loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

		optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
		minimize = optimizer.minimize(loss_total)

		return s_t, a_t, r_t, minimize

	def optimize(self):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return

		with self.lock_queue:
			if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
				return 									# we can't yield inside lock

			s, a, r, s_, s_mask = self.train_queue
			self.train_queue = [ [], [], [], [], [] ]

		s = np.vstack(s)
		a = np.vstack(a)
		r = np.vstack(r)
		s_ = np.vstack(s_)
		s_mask = np.vstack(s_mask)

		if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

		v = self.predict_v(s_)
		r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
		
		s_t, a_t, r_t, minimize = self.graph
		self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

	def train_push(self, s, a, r, s_):
		with self.lock_queue:
			self.train_queue[0].append(s)
			self.train_queue[1].append(a)
			self.train_queue[2].append(r)

			if s_ is None:
				self.train_queue[3].append(NONE_STATE)
				self.train_queue[4].append(0.)
			else:	
				self.train_queue[3].append(s_)
				self.train_queue[4].append(1.)

	def predict(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return p, v

	def predict_p(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return p

	def predict_v(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return v

#---------
frames = 0
class Agent:
	def __init__(self, eps_start, eps_end, eps_steps):
		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps

		self.memory = []	# used for n_step return
		self.R = 0.

	def getEpsilon(self):
		if(frames >= self.eps_steps):
			return self.eps_end
		else:
			return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

	def act(self, s):
		eps = self.getEpsilon()			
		global frames; frames = frames + 1

		if random.random() < eps:
			#return random.randint(0, NUM_ACTIONS-1)
			a= (np.random.rand(NUM_ACTIONS)-0.5)*0.5;
			#print(a)
			return a
		else:
			s = np.array([s])
			p = brain.predict_p(s)[0]

			## a = np.argmax(p)
			#a = np.random.choice(NUM_ACTIONS, p=p)
			a = p

			return a
	
	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, self.R, s_

		#a_cats = np.zeros(NUM_ACTIONS)	# turn action into one-hot representation
		#a_cats[a] = 1 
		a_cats = a;

		self.memory.append( (s, a_cats, r, s_) )

		self.R = ( self.R + r * GAMMA_N ) / GAMMA

		if s_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				brain.train_push(s, a, r, s_)

				self.R = ( self.R - self.memory[0][2] ) / GAMMA
				self.memory.pop(0)		

			self.R = 0

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			brain.train_push(s, a, r, s_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)	
	
	# possible edge case - if an episode ends in <N steps, the computation is incorrect
		
#---------
class Environment(threading.Thread):
	stop_signal = False

	def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		threading.Thread.__init__(self)

		self.render = render
		#self.env = gym.make(ENV)
		self.env = MyEnv();
		self.agent = Agent(eps_start, eps_end, eps_steps)

	def runEpisode(self):
		s = self.env.reset()

		R = 0
		while True:         
			time.sleep(THREAD_DELAY) # yield 

			if self.render: self.env.render()

			a = self.agent.act(s)
			s_, r, done, info = self.env.step(a)

			if done: # terminal state
				s_ = None

			self.agent.train(s, a, r, s_)

			s = s_
			R += r

			if done or self.stop_signal:
				break

		print("Total R:", R)

	def run(self):
		while not self.stop_signal:
			self.runEpisode()

	def stop(self):
		self.stop_signal = True

#---------
class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			brain.optimize()

	def stop(self):
		self.stop_signal = True

#-- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
#NUM_ACTIONS = env_test.env.action_space.n
NUM_ACTIONS = len(env_test.env.action_space.high);
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()	# brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
	o.start()

for e in envs:
	e.start()

time.sleep(RUN_TIME)

for e in envs:
	e.stop()
for e in envs:
	e.join()

for o in opts:
	o.stop()
for o in opts:
	o.join()

print("Training finished")
env_test.run()