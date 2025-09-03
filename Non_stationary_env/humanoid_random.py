import mujoco_py
import gym
import numpy as np
import random
import math

from gym.envs.mujoco.humanoid_v3 import HumanoidEnv

class HumanoidTransferEnv(HumanoidEnv):
    def __init__(self, gravity=9.81, wind=0.0, **kwargs):
        super().__init__(**kwargs)
        self.model.opt.viscosity = 0.00002 
        self.model.opt.density = 1.2
        self.model.opt.gravity[:] = np.array([0., 0., -gravity])
        self.model.opt.wind[:] = np.array([-wind, 0., 0.])

class HumanoidRandomEnv(HumanoidEnv):
    
    def __init__(self, uncertainty=1.0, **kwargs):
        super().__init__(**kwargs)
        # 基本环境参数
        self.model.opt.viscosity = 0.00002 
        self.model.opt.density = 1.2

        # 默认参数 - 调整为与给定代码一致
        self.default_gravity = 14.715
        self.default_wind = 1.0
        
        # 变化参数 - 调整为与给定代码一致
        self.w_gravity = 0.5            # 重力变化的频率
        self.w_wind = 0.5               # 风力变化的频率
        self.amp_gravity = 4.905 * uncertainty  # 重力变化的振幅
        self.amp_wind = 0.5 * uncertainty      # 风力变化的振幅
        self.t = 0                      # 时间步计数

    def reset(self, gravity=None, wind=None):
        '''动态调整重力和风力，基于正弦波函数'''
        self.t += 1  # 增加时间步
        
        # 如果没有提供参数，使用基于正弦函数的动态值
        if gravity is None:
            gravity = self.default_gravity + self.amp_gravity * np.sin(self.w_gravity * self.t)
        
        if wind is None:
            wind = self.default_wind + self.amp_wind * np.sin(self.w_wind * self.t)
        
        # 确保参数在合理范围内
        gravity = max(5.0, min(25.0, gravity))  # 调整重力范围
        wind = max(0.0, min(2.0, wind))        # 调整风力范围
        
        # 更新模型参数
        self.model.opt.gravity[:] = np.array([0., 0., -gravity])
        self.model.opt.wind[:] = np.array([-wind, 0., 0.])

        return super().reset()

if __name__ == "__main__":
    env = gym.make("HumanoidRandom-v0")
    obs = env.reset()  # 使用基于正弦波的参数
    
    for i in range(100):
        print(f"Step {i}: Gravity={env.model.opt.gravity[2]}, Wind={env.model.opt.wind[0]}")
        env.step(np.random.rand(17))
        env.render()
        
        # 每20步重置一次环境，查看参数变化
        if i % 20 == 0:
            env.reset()
    
    env.close()