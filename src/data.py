import math
import torch
import random
import numpy as np
from collections import deque, namedtuple
from .utils import angle_between

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'done', 'next_state'],
)
        

class AbsoluteState:
    def __init__(self, state_size: int):
        self.state_size = state_size

    def process(self, state):
        robot2control = 0

        robot = state['ally'][robot2control]['pos_xy']
        angle_robot = state['ally'][robot2control]['theta']
        v_robot = state['ally'][robot2control]['vel_xy']
        w_robot = state['ally'][robot2control]['w']
        ball = state['ball']['pos_xy']
        v_ball = state['ball']['vel_xy']
        
        goal = np.array([0.75,0])

        near_wall1 = 1 - (-robot[0] + 0.75)
        near_wall2 = 1 - (robot[0] + 0.75)
        near_wall3 = 1 - (-robot[1] + 0.65)
        near_wall4 = 1 - (robot[1] + 0.65)

        processed_state = [ robot[0], robot[1],
                            ball[0], ball[1],
                            v_robot[0], v_robot[1],
                            v_ball[0],v_ball[1],
                            goal[0], goal[1],
                            angle_robot, w_robot / 6.28,
                            near_wall1, near_wall2,
                            near_wall3, near_wall4,
                            ]

        return processed_state

class RelativeState:
    def __init__(self, state_size: int):
        self.state_size = state_size

        self.len_field = np.array([1.5, 1.3])
        self.diagonal_len_field = np.sqrt(np.sum(self.len_field**2))

    def process(self, state):
        robot2control = 0

        robot = state['ally'][robot2control]['pos_xy']
        angle_robot = state['ally'][robot2control]['theta']
        v_robot = state['ally'][robot2control]['vel_xy']
        w_robot = state['ally'][robot2control]['w']
        
        robot_vel_unit = np.array([math.cos(angle_robot), math.sin(angle_robot)])

        ball = state['ball']['pos_xy']
        v_ball = state['ball']['vel_xy']
        
        opponent_goal = np.array([0.75,0.0])
        ally_goal = -opponent_goal

        robot_ball = ball - robot
        norm_robot_ball = np.linalg.norm(robot_ball)
        velrobot_robotballangle = angle_between(robot_vel_unit, robot_ball)

        ball_opponentgoal = opponent_goal - ball
        ball_allygoal = ally_goal - ball

        robot_allygoal = ally_goal - robot
        norm_robot_allygoal = np.linalg.norm(robot_allygoal)

        vrobot_ballallygoal_angle = angle_between(robot_vel_unit, ball_allygoal)
        
        near_wall1 = -robot[0] + 0.75
        near_wall2 = robot[0] + 0.75
        near_wall3 = -robot[1] + 0.65
        near_wall4 = robot[1] + 0.65

        processed_state = [ velrobot_robotballangle, norm_robot_ball,
                            v_robot[0], v_robot[1],
                            w_robot / 6.28, angle_robot,
                            near_wall1, near_wall2,
                            near_wall3, near_wall4,
                            norm_robot_allygoal, 
                            vrobot_ballallygoal_angle,
                            v_ball[0], v_ball[1],
                            ] 
        
        return processed_state

class Memory(object):
    def __init__(self, memory_size: int, is_per: bool =False) -> None:
        super().__init__()
        self.memory = deque(maxlen=memory_size)
        self.is_per = is_per

    def append(self, experience: Experience) -> None:
        self.memory.append(experience)

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):

        state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        next_state_batch = []

        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, done, next_state in batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append([reward])
            done_batch.append([done])
            next_state_batch.append(next_state)
        
        return (
            None,
            torch.from_numpy(np.array(state_batch)).float(),
            torch.from_numpy(np.array(action_batch)).float(),
            torch.from_numpy(np.array(reward_batch, dtype=np.float32)).float(),
            torch.from_numpy(np.array(done_batch, dtype=np.float32)).float(),
            torch.from_numpy(np.array(next_state_batch)).float(),
            None,
        )
