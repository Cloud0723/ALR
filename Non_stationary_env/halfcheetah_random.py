import mujoco_py
import gym
import numpy as np
import random
import math

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

class HalfCheetahTransferEnv(HalfCheetahEnv):
    def __init__(self, leg_length: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        # 修改腿部长度
        # 前腿部分
        self.model.geom_size[4][1] += leg_length  # 大腿 (thigh)
        self.model.geom_size[5][1] += leg_length  # 小腿 (shin)
        self.model.geom_size[6][1] += leg_length  # 脚 (foot)
        
        # 后腿部分
        self.model.geom_size[7][1] += leg_length  # 大腿 (thigh)
        self.model.geom_size[8][1] += leg_length  # 小腿 (shin)
        self.model.geom_size[9][1] += leg_length  # 脚 (foot)
        
        # 调整关节位置以适应新的腿长
        self.update_joint_positions(leg_length)
        
    def update_joint_positions(self, leg_length):
        # 根据腿长调整关节位置
        # 这些索引和偏移量需要根据实际模型结构调整
        # 前腿关节
        self.model.body_pos[2][0] += leg_length  # 前腿臀部
        self.model.body_pos[3][0] += leg_length * 2  # 前腿膝盖
        self.model.body_pos[4][0] += leg_length * 3  # 前腿脚踝
        
        # 后腿关节
        self.model.body_pos[5][0] -= leg_length  # 后腿臀部
        self.model.body_pos[6][0] -= leg_length * 2  # 后腿膝盖
        self.model.body_pos[7][0] -= leg_length * 3  # 后腿脚踝

class HalfCheetahRandomEnv(HalfCheetahEnv):
    def __init__(self, uncertainty=1.0, **kwargs):
        super().__init__(**kwargs)
        self.default_leg_length = 0.0  # 默认腿长基准值
        self.w_leg = 0.05  # 腿长变化的频率
        self.amp_leg = 0.25 * uncertainty  # 腿长变化的振幅
        self.t = 0  # 时间步计数
        
    def reset(self, leg_length=None):
        '''动态调整腿长，基于正弦波函数'''
        self.t += 1  # 增加时间步
        
        # 如果没有提供参数，使用基于正弦函数的动态值
        if leg_length is None:
            leg_length = self.default_leg_length + self.amp_leg * math.sin(self.w_leg * self.t)
        
        # 确保参数在合理范围内
        leg_length = max(-0.1, min(0.2, leg_length))
        
        # 更新前腿模型参数
        self.model.geom_size[4][1] = 0.046 + leg_length  # 大腿
        self.model.geom_size[5][1] = 0.046 + leg_length  # 小腿
        self.model.geom_size[6][1] = 0.046 + leg_length  # 脚
        
        # 更新后腿模型参数
        self.model.geom_size[7][1] = 0.046 + leg_length  # 大腿
        self.model.geom_size[8][1] = 0.046 + leg_length  # 小腿
        self.model.geom_size[9][1] = 0.046 + leg_length  # 脚
        
        # 调整关节位置以适应新的腿长
        self.update_joint_positions(leg_length)
        
        return super().reset()
    
    def update_joint_positions(self, leg_length):
        # 根据腿长调整关节位置
        # 前腿关节
        self.model.body_pos[2][0] = 0.5 + leg_length  # 前腿臀部
        self.model.body_pos[3][0] = 0.6 + leg_length * 2  # 前腿膝盖
        self.model.body_pos[4][0] = 0.7 + leg_length * 3  # 前腿脚踝
        
        # 后腿关节
        self.model.body_pos[5][0] = -0.5 - leg_length  # 后腿臀部
        self.model.body_pos[6][0] = -0.6 - leg_length * 2  # 后腿膝盖
        self.model.body_pos[7][0] = -0.7 - leg_length * 3  # 后腿脚踝

if __name__ == "__main__":
    env = gym.make("HalfCheetahRandom-v0")
    obs = env.reset()  # 使用基于正弦波的参数
    
    for _ in range(10):
        for i in range(50):
            env.step(np.random.rand(6))
            env.render()

        # 可以不传入参数，让环境自动使用基于正弦波的参数
        env.reset()
    
    env.close() 