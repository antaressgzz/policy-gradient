"""
This is an Actor-Critic episodic algorithms using eligibility trace(TD lambda) 
It use two neural networks to paramatrize the actor and critic. 
For more detail about this algorithm, please refer to page 274, section 13.5 of 
'Reinforcement Learning: An Introduction' by Sutton and Barto(v2018). 
This program is aimed at solve some simple classic control problems on open AI gym.

@author: Ziyang Zhang
"""

import tensorflow as tf
import numpy as np

class PGagent:
    def __init__(self,
                 feature_num,
                 action_num,
                 h_size=20,  # size of the network
                 gamma=0.99,
                 alpha_c=2 ** -14,
                 alpha_a=2 ** -12,
                 lambda_c=0.8, 
                 lambda_a=0.8,
                 tensor_board=True):
        self.nF = feature_num
        self.nA = action_num
        self.I = 1
        self.gamma = gamma
        self.sess = tf.Session()
        self.bulid_graph(alpha_c, alpha_a, lambda_c, lambda_a, h_size)
        self.sess.run(tf.global_variables_initializer())
        self.learning_counter = 0
        self.tensor_board = tensor_board
        
        if tensor_board == True:
            # tensorboard --logdir=logs
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

    def bulid_graph(self, alpha_c, alpha_a, lambda_c, lambda_a, h_size):
        self.state = tf.placeholder(tf.float32, [self.nF, 1], name='state')
        self.action = tf.placeholder(tf.int32, name='action')
        self.delta = tf.placeholder(tf.float32, name='delta')

        with tf.variable_scope('critic'):
            self.state_value, critic_params, critic_trace, g_critic = self.build_net('critic', h_size, 1)
            # Weights and biases colletion of the critic network
            critic_params = tf.get_collection('critic_params')
            # Eligibility trace colletion
            critic_trace  = tf.get_collection('critic_trace')
            # op for initialize the eligiblity trace at start of every episode
            with tf.name_scope('init_critic_trace'):
                self.init_critic_trace = [ t.assign(np.zeros(t.get_shape())) for t in critic_trace ]
            # op for update the eligiblity trace at each learning step
            with tf.name_scope('updateCtrace'):
                self.updateCtrace =[tf.assign(z, self.gamma*lambda_c*z+self.I*g) for (z, g) in zip(critic_trace, g_critic)]
            # update the parameters of the parameterization of the critic
            with tf.name_scope('updateCparams'):
                self.updateCparams = [tf.assign(w, w+alpha_c*self.delta*z) for (w, z) in zip(critic_params, critic_trace)]
        
        with tf.variable_scope('actor'):
            self.policy, actor_params, actor_trace, g_policy = self.build_net('actor', h_size, self.nA)
            actor_params = tf.get_collection('actor_params')
            actor_trace  = tf.get_collection('actor_trace')
            with tf.name_scope('init_actor_trace'):
                self.init_actor_trace = [ t.assign(np.zeros(t.get_shape())) for t in actor_trace ]
            with tf.name_scope('updateAtrace'):
                self.updateAtrace =[tf.assign(z, self.gamma*lambda_a*z+self.I*g) for (z, g) in zip(actor_trace, g_policy)]
            with tf.name_scope('updateAparams'):
                self.updateAparams = [tf.assign(theta, theta+alpha_a*self.delta*z) for (theta, z) in zip(actor_params, actor_trace)]
                
    def build_net(self, name, h_size, o_size):
        w_initializer = tf.random_normal_initializer(0, 0.01)
        b_initializer = tf.constant_initializer(0.01)
        n1 = name + '_params'
        n2 = name + '_trace'
        params = [n1, tf.GraphKeys.GLOBAL_VARIABLES] # Collection for parameters of network
        trace  = [n2, tf.GraphKeys.GLOBAL_VARIABLES] # Collection for eligiblity trace
        with tf.variable_scope('l1'):
            w1   = tf.get_variable('w1', [h_size, self.nF], tf.float32, initializer=w_initializer ,collections=params)
            w1_t = tf.get_variable('w1_t', [h_size, self.nF], tf.float32, initializer=tf.zeros_initializer, collections=trace)   # Eligibility trace of w1                    
            b1   = tf.get_variable('b1', [h_size, 1], tf.float32, initializer=b_initializer, collections=params)            
            b1_t = tf.get_variable('b1_t', [h_size, 1], tf.float32, initializer=tf.zeros_initializer, collections=trace)
            tf.summary.histogram('w1', w1)
            tf.summary.histogram('w1_t', w1_t)
            tf.summary.histogram('b1', b1)
            tf.summary.histogram('b1_t', b1_t)
            l1 = tf.nn.leaky_relu( tf.matmul(w1, self.state) + b1)
            
        with tf.variable_scope('l2'):
            w2   = tf.get_variable('w2', [o_size, h_size], tf.float32, initializer=w_initializer, collections=params)          
            w2_t = tf.get_variable('w2_t', [o_size, h_size], tf.float32, initializer=tf.zeros_initializer, collections=trace)            
            b2   = tf.get_variable('b2', [o_size, 1], tf.float32, initializer=b_initializer, collections=params)           
            b2_t = tf.get_variable('b2_t', [o_size, 1], tf.float32, initializer=tf.zeros_initializer, collections=trace)
            tf.summary.histogram('w2', w2)
            tf.summary.histogram('w2_t', w2_t)
            tf.summary.histogram('b2', b2)
            tf.summary.histogram('b2_t', b2_t)            
            outputs = tf.matmul(w2, l1) + b2
            
        if name == 'actor':
            outputs = tf.reshape(outputs, [-1])
            softmax = tf.nn.softmax(outputs)
            log_p   = tf.log(softmax)
            gradients = tf.gradients(log_p[self.action] , [w1, b1, w2, b2]) # Gradients of actor paramatrization
            return softmax, params, trace, gradients
        else:
            gradients = tf.gradients(outputs[0], [w1, b1, w2, b2]) # Gradients of critic paramatrization
            return outputs, params, trace, gradients

    def episode_start(self):
        # The eligiblity trace needs to be set to 0 at every start of an episode
        self.sess.run(self.init_actor_trace)
        self.sess.run(self.init_critic_trace)
        self.I = 1
           
    def choose_action(self, observ):
        probs = self.sess.run(self.policy, feed_dict={self.state:observ})
        action = np.random.choice(self.nA, p=probs)
        return action
        
    def learn(self, reward, observ, observ_, action, done):
        state_value  = self.sess.run(self.state_value, feed_dict={self.state:observ})       
        if done:
            # For terminal state
            state_value_ = 0
        else:
            state_value_ = self.sess.run(self.state_value, feed_dict={self.state:observ_})      
        delta = reward + self.gamma * state_value_ - state_value
        self.sess.run(self.updateAtrace,  feed_dict={self.state:observ, self.action:action}) # Update trace
        self.sess.run(self.updateAparams, feed_dict={self.delta:delta}) # Update network parameters
        self.sess.run(self.updateCtrace,  feed_dict={self.state:observ})
        self.sess.run(self.updateCparams, feed_dict={self.delta:delta})
        self.I *= self.gamma
        self.learning_counter += 1
       
        if self.tensor_board and self.learning_counter % 5 == 0:
            s = self.sess.run(self.merged)
            self.writer.add_summary(s, self.learning_counter)