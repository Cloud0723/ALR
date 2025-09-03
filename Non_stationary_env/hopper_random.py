import mujoco_py
import gym
import numpy as np
import random
import math

from gym.envs.mujoco.hopper_v3 import HopperEnv

class HopperTransferEnv(HopperEnv):
    def __init__(self, torso_len: float = 0.2, foot_len: float = 0.195, **kwargs):
        super().__init__(**kwargs)
        self.model.body_pos[1][2] = 1.05 + torso_len
        self.model.body_pos[2][2] = -torso_len
        self.model.geom_size[1][1] = torso_len

        self.model.geom_size[4][1] = foot_len



class HopperRandomEnv(HopperEnv):
    def __init__(self, uncertainty=0, **kwargs):
        super().__init__(**kwargs)
        self.default_torso_len = 0.4   # 默认躯干长度
        self.default_foot_len = 0.395  # 默认脚长度
        self.w_torso = 0.05           # 躯干长度变化的频率
        self.w_foot = 0.05            # 脚长度变化的频率
        self.amp_torso = 0.3 * uncertainty  # 躯干长度变化的振幅
        self.amp_foot = 0.3* uncertainty    # 脚长度变化的振幅
        self.t = 0                     # 时间步计数

    def reset(self, torso_len=None, foot_len=None):
        '''动态调整躯干长度和脚长度，基于正弦波函数'''
        self.t += 1  # 增加时间步
        
        # 如果没有提供参数，使用基于正弦函数的动态值
        if torso_len is None:
            torso_len = self.default_torso_len + self.amp_torso * math.sin(self.w_torso * self.t)
        
        if foot_len is None:
            foot_len = self.default_foot_len + self.amp_foot * math.sin(self.w_foot * self.t)
        
        # 确保参数在合理范围内
        # torso_len = max(0.1, min(0.35, torso_len))
        # foot_len = max(0.1, min(0.3, foot_len))
        
        # 更新模型参数
        self.model.body_pos[1][2] = 1.05 + torso_len
        self.model.body_pos[2][2] = -torso_len
        self.model.geom_size[1][1] = torso_len

        self.model.geom_size[4][1] = foot_len

        return super().reset()

if __name__ == "__main__":
    env = gym.make("HopperRandom-v0", uncertainty=0.5)
    obs = env.reset()  # 使用基于正弦波的参数
    
    for _ in range(10):
        for i in range(50):
            env.step(np.random.rand(3))
            env.render()

        # 可以不传入参数，让环境自动使用基于正弦波的参数
        env.reset()
    
    env.close()