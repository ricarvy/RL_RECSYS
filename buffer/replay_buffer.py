# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 17:29
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : replay_buffer.py
# @Software: PyCharm

import logging
from datetime import datetime
from collections import deque
import random

class Replay_Buffer(object):
    def __init__(self, config):
        o_t = datetime.now()
        logging.info("create replay buffer of agent ..............................")


        self.buffer_size = config["replay_buffer"]["buffer_size"]
        self.batch_size = config["replay_buffer"]["batch_size"]
        self.memory = deque(maxlen=self.buffer_size)
        logging.info("The params are : buffer_size, batch_size")
        logging.info(f"Replay buffer's buffer_size is {self.buffer_size}")
        logging.info(f"Replay buffer's batch_size is {self.batch_size}")
        n_t = datetime.now()
        logging.info(f"Time consuming is {(n_t - o_t).microseconds}")

        logging.info("replay buffer creation ended ..............................")

    def __call__(self):
        return self.memory

    def store_transition(self, transition):
        self.memory.append(transition)

    def store_transitions(self, transitions):
        self.memory.extend(transitions)

    def get_batch(self, batch_size=None):
        b_s = batch_size or self.batch_size
        cur_men_size = len(self.memory)
        if cur_men_size < b_s:
            return random.sample(list(self.memory), cur_men_size)
        else:
            return random.sample(list(self.memory), b_s)

    def memory_state(self):
        return {"buffer_size": self.buffer_size,
                "current_size": len(self.memory),
                "full": len(self.memory)==self.buffer_size}

    def empty_transition(self):
        self.memory.clear()

# if __name__ == '__main__':
#     import numpy as np
#     replay_buffer = Replay_Buffer(buffer_size=4)
#     print(replay_buffer.memory_state())
#     replay_buffer.store_transition([1, 2, 3, 4, False])
#     print(replay_buffer.memory_state())
#     replay_buffer.store_transition([2, 2, 3, 4, False])
#     print(replay_buffer.memory_state())
#     replay_buffer.store_transition([3, 2, 3, 4, True])
#     print(replay_buffer.memory_state())
#     print(replay_buffer())
#
#     replay_buffer.store_transition([4, 2, 3, 4, True])
#     print(replay_buffer.memory_state())
#     print(replay_buffer())
#
#     replay_buffer.store_transitions([[5, 2, 3, 4, False]])
#     print(replay_buffer.memory_state())
#     print(replay_buffer())
#
#     # batch = replay_buffer.get_batch(3)
#     # print("batch", batch)
#     # transpose_batch = list(zip(*batch))
#     # print("transpose_batch", transpose_batch)
#     # s_batch = np.array(transpose_batch[0])
#     # a_batch = list(transpose_batch[1])
#     # r_batch = list(transpose_batch[2])
#     # next_s_batch = list(transpose_batch[3])
#     # done_batch = np.array(transpose_batch[4])
#     # print("s_batch", s_batch)
#     # print("a_batch", a_batch)
#     # print("r_batch", r_batch)
#     # print("next_s_batch", next_s_batch)
#     # print("done_batch", done_batch)
#     # print((1-done_batch)*s_batch)
#     #
#     # replay_buffer.empty_transition()
#     # print(replay_buffer.memory_state())
#     # print(replay_buffer())