import time

import numpy as np

from ma_gym.envs.combat.combat import Combat


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.agent_num = 4  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        team_size = self.agent_num
        grid_size = (15, 15)
        self.env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
        self.obs_dim = 150  # 设置智能体的观测维度 # set the observation dimension of agents
        self.action_dim = self.env.action_space[
            0].n  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional

    def reset(self):

        s = self.env.reset()

        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.array(s[i])  # np.random.random(size=(14,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        self.env.render("human")
        time.sleep(0.4)
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        action_index = [int(np.where(act == 1)[0][0]) for act in actions]
        next_s, r, done, info = self.env.step(action_index)
        for i in range(self.agent_num):
            # r[agent_i] + 100 if info['win'] else r[agent_i] - 0.1
            sub_agent_obs.append(np.array(next_s[i]))
            sub_agent_reward.append([r[i] + 100 if info['win'] else r[i] - 0.1])
            sub_agent_done.append(done[i])
            sub_agent_info.append(info)

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]