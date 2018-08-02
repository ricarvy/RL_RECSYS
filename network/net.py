# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 15:15
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : net.py
# @Software: PyCharm

import os, sys
lib_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(lib_path)


import tensorflow as tf
import logging

from datetime import datetime

from network.actor import DDPG_Actor
from network.critic import DDPG_Critic
from utils.json_loader import config_load


class Model(object):
    def __init__(self,
                 config,
                 optimizer=None,
                 sess=None):
        o_t = datetime.now()
        logging.basicConfig(level=logging.DEBUG)
        self.name = "AC-DDPG"
        logging.info(f"Start create model of agent named {self.name} ..............................")


        self.state_dim = config["normal_args"]["state_dim"]
        self.action_dim = config["normal_args"]["action_dim"]
        self.actor_learning_rate = config["network"]["actor"]["learning_rate"]
        self.critic_learning_rate = config["network"]["critic"]["learning_rate"]
        self.tau = config["normal_args"]["tau"]
        logging.warning("The params are : state_dim, action_dim, actor_learing_rate, critic_learning_rate, tau")
        logging.info(f"{self.name}'s state_dim is {self.state_dim}")
        logging.info(f"{self.name}'s action_dim is {self.action_dim}")
        logging.info(f"{self.name}'s actor_learning_rate is {self.actor_learning_rate}")
        logging.info(f"{self.name}'s critic_learning_rate is {self.critic_learning_rate}")
        logging.info(f"{self.name}'s tau is {self.tau}")

        #tf.reset_default_graph()
        self.sess = sess or tf.Session()

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        global_step_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_step")
        self.sess.run(tf.variables_initializer(global_step_vars))

        self.actor_scope = "actor_net"
        with tf.name_scope(self.actor_scope):
            self.actor = DDPG_Actor(self.state_dim,
                        self.action_dim,
                        learning_rate=self.actor_learning_rate,
                        tau=self.tau,
                        scope=self.actor_scope,
                        sess=self.sess)

        self.critic_scope = "critic_net"
        with tf.name_scope(self.critic_scope):
            self.critic = DDPG_Critic(self.state_dim,
                        self.action_dim,
                        learning_rate=self.critic_learning_rate,
                        tau=self.tau,
                        scope=self.critic_scope,
                        sess=self.sess)
        n_t = datetime.now()
        logging.info(f"Create two types of scope : {self.actor_scope} , {self.critic_scope}")
        logging.info(f"Time consuming is {(o_t-n_t).microseconds} ms")


        logging.info("Agent Creation Ended ..............................")

    def update(self, state_batch, action_batch, y_batch, sess=None):

        sess = sess or self.sess

        ### update source critic net
        self.critic.update_source_critic_net(state_batch, action_batch, y_batch, sess)

        action_batch_for_grad = self.actor.predict_action_source_net(state_batch, sess)
        action_grad_batch = self.critic.get_action_grads(state_batch, action_batch_for_grad, sess)

        ### update source actor net
        self.actor.update_source_actor_net(state_batch, action_grad_batch, sess)

        ### update target critic net
        self.critic.update_target_critic_net(sess)

        ### update target actor net
        self.actor.update_target_actor_net(sess)

    def predict_action(self, observation, sess=None):
        sess = sess or self.sess
        return self.actor.predict_action_source_net(observation, sess)

if __name__ == '__main__':
    import numpy as np
    state_dim = 10
    action_dim = 3
    actor_learning_rate = np.random.rand(1)
    print("actor_learning_rate: ", actor_learning_rate)
    critic_learning_rate = np.random.rand(1)
    print("critic_learning_rate: ", critic_learning_rate)
    tau = np.random.rand(1)
    print("tau: ", tau)
    config = config_load("../config.json")
    sess = tf.Session()
    model = Model(config,
                  sess=sess)
    random_state = np.random.normal(size=state_dim)
    print("random_state", random_state)

    random_action = np.random.random(size=action_dim)
    print("random_action", random_action)

    # check prediction
    pred_action = model.predict_action([random_state])
    print("predict_action", pred_action)

    # check forward
    target_q = model.critic.predict_q_target_net([random_state], [random_action], sess)
    print("predict target q", target_q)
    y = target_q[0] + 1

    weight_target_critic = model.critic.run_layer_weight_target(sess)
    model.update([random_state], [random_action], [y])
    weight_target_critic_updated = model.critic.run_layer_weight_target(sess)
    gap_weight = np.array(weight_target_critic) - np.array(weight_target_critic_updated)
    print(gap_weight)