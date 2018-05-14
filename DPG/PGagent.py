
"""
@author: Ziyang Zhang
"""

import tensorflow as tf
import numpy as np

class PGagent:
    def __init__(self,
                 feature_num,
                 action_num,
                 h_size=20,
                 gamma=0.99,
                 alpha_c=0.1, 
                 alpha_a=0.1, 
                 lambda_c=0.8, 
                 lambda_a=0.8):
        self.nF = feature_num
        self.nA = action_num
        self.sess = tf.Session()
        self.I = 1
        self.gamma = gamma
        self.bulid_graph(alpha_c, alpha_a, lambda_c, lambda_a, h_size)
        self.sess.run(tf.global_variables_initializer())
        
        
    def bulid_graph(self, alpha_c, alpha_a, lambda_c, lambda_a, h_size):
        self.state = tf.placeholder(tf.float32, [self.nF, 1], name='state')
        self.action = tf.placeholder(tf.int32, name='action')
        self.delta = tf.placeholder(tf.float32, name='delta')

        with tf.variable_scope('critic'):
            self.state_value, critic_params, critic_trace, critic_trace0, g_critic = self.build_net('critic', h_size, 1)
            critic_params = tf.get_collection('critic_params')
            critic_trace = tf.get_collection('critic_trace')
            critic_trace0 = tf.get_collection('critic_trace0')
            self.init_critic_trace = [tf.assign(t, t0) for (t, t0) in zip(critic_trace, critic_trace0)]
            self.updateCtrace =[tf.assign(z, self.gamma*lambda_c*z+self.I*g) for (z, g) in zip(critic_trace, g_critic)]
            self.updateCparams = [tf.assign(w, w+alpha_c*self.delta*z) for (w, z) in zip(critic_params, critic_trace)]
        
        with tf.variable_scope('actor'):
            self.policy, actor_params, actor_trace, actor_trace0, g_policy = self.build_net('actor', h_size, self.nA)
            actor_params = tf.get_collection('actor_params')
            actor_trace = tf.get_collection('actor_trace')
            actor_trace0 = tf.get_collection('actor_trace0')
            self.init_actor_trace = [tf.assign(t, t0) for (t, t0) in zip(actor_trace, actor_trace0)]
            self.updateAtrace =[tf.assign(z, self.gamma*lambda_a*z+self.I*g) for (z, g) in zip(actor_trace, g_policy)]
            self.updateAparams = [tf.assign(theta, theta+alpha_a*self.delta*z) for (theta, z) in zip(actor_params, actor_trace)]
                
    def build_net(self, name, h_size, o_size):
        w_initializer = tf.random_normal_initializer(0, 0.1)
#        b_initializer = tf.constant_initializer(0.1)
        n1 = name + '_params'
        n2 = name + '_trace'
        n3 = name + '_trace0'
        params = [n1, tf.GraphKeys.GLOBAL_VARIABLES]
        trace  = [n2, tf.GraphKeys.GLOBAL_VARIABLES]
        trace0 = [n3, tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [h_size, self.nF], tf.float32, 
                                 initializer=w_initializer ,collections = params)
            w1_t = tf.get_variable('w1_t', [h_size, self.nF], tf.float32, 
                                   initializer=tf.zeros_initializer, collections = trace)
            w1_t0 = tf.get_variable('w1_t0', [h_size, self.nF], tf.float32, 
                                   initializer=tf.zeros_initializer, collections = trace0)           
#                b1 = tf.get_variable('b1', [h_size, 1], tf.float32, collections = params)
#                b1_t = tf.get_variable('b1_t', [h_size, 1], tf.float32, 
#                                       initializer=tf.zeros_initializer, collections = trace)               
#                b1_t0 = tf.get_variable('b1_t0', [h_size, 1], tf.float32, 
#                                        initializer=tf.zeros_initializer, collections = trace0)
            l1 = tf.nn.leaky_relu(tf.matmul(w1, self.state))
            
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [o_size, h_size], tf.float32, 
                                 initializer=w_initializer, collections = params)
            w2_t = tf.get_variable('w2_t', [o_size, h_size], tf.float32, 
                                   initializer=tf.zeros_initializer, collections = trace)
            w2_t0 = tf.get_variable('w2_t0', [o_size, h_size], tf.float32, 
                                   initializer=tf.zeros_initializer, collections = trace0)
           
#                b2 = tf.get_variable('b2', [1, 1], tf.float32, collections = params)
#                b2_t = tf.get_variable('b2_t', [1, 1], tf.float32, 
#                                       initializer=tf.zeros_initializer, collections = trace)               
#                b2_t0 = tf.get_variable('b2_t0', [1, 1], tf.float32, 
#                                        initializer=tf.zeros_initializer, collections = trace0)            
            outputs = tf.matmul(w2, l1)
            
        if name == 'actor':
            outputs = tf.reshape(outputs, [2])
            softmax = tf.nn.softmax(outputs)
            log_p = tf.log(softmax)
            gradients = tf.gradients(log_p[self.action] , [w1, w2])
            return softmax, params, trace, trace0, gradients
        else:
            gradients = tf.gradients(outputs[0], [w1, w2])
            return outputs, params, trace, trace0, gradients

    def episode_start(self):
        self.sess.run(self.init_actor_trace)
        self.sess.run(self.init_critic_trace)
        self.I = 1
           
    def choose_action(self, observ):
        probs = self.sess.run(self.policy, feed_dict={self.state:observ})
        action = np.random.choice(self.nA, p=probs)
        return action
        
    def learn(self, reward, observ, observ_, action):
        state_value  = self.sess.run(self.state_value, feed_dict={self.state:observ})
        state_value_ = self.sess.run(self.state_value, feed_dict={self.state:observ_})       
        delta = reward + self.gamma * state_value_ - state_value
        self.sess.run(self.updateAtrace, feed_dict={self.state:observ, self.action:action})
        self.sess.run(self.updateAparams, feed_dict={self.delta:delta})
        self.sess.run(self.updateCtrace, feed_dict={self.state:observ})
        self.sess.run(self.updateCparams, feed_dict={self.delta:delta})
        self.I *= self.gamma