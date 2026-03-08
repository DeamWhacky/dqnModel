import gymnasium as gym
import flappy_bird_gymnasium # registers FlappyBird-v0 env with gymnasium
import numpy as np


class FlappyBirdDNN(gym.Wrapper):

    def __init__(self, render=False):
        render_mode = "human" if render else None
        env = gym.make("FlappyBird-v0", render_mode=render_mode, disable_env_checker=True)
        super().__init__(env)

        self.observation_space = env.observation_space

        self.action_space = env.action_space


    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed)
        return state.astype(np.float32), info


    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        reward = self._shape_reward(reward, terminated)

        return state.astype(np.float32), reward, terminated, truncated, info


    def _shape_reward(self, reward, done):
        if done:
            return -100.0
        if reward > 0:
            return 20.0
        return 0.1
#------------------------------------------------------#    
# past this line starts the edging logic #edgeforbirds
#def _get_state(self):
    #game_state = self.env.game_state.getGameState()

    #player_vel = game_state["player_vel"]
    #pipe_dist = game_state["next_pipe_dist_to_player"]

    #pipe_top = game_state["next_pipe_top_y"]
    #pipe_bottom = game_state["next_pipe_bottom_y"]
    #player_y = game_state["player_y"]

    # Store for reward shaping
    #self.last_player_y = player_y
    #self.last_pipe_top = pipe_top
    #self.last_pipe_bottom = pipe_bottom

    #gap_center = (pipe_top + pipe_bottom) / 2
    #vertical_distance = player_y - gap_center

    #return np.array([
        #player_vel / 20,
        #pipe_dist / 400,
        #vertical_distance / 300
    #], dtype=np.float32)
#now we edge
#def _shape_reward(self, reward, done):

    #if done:
        #return -50  # death dominates everything

    #shaped_reward = 1  # survival reward

    #if reward > 0:
        #shaped_reward = 10  # pipe passed

    #EDGE BONUS
    # Distance from bird to closest pipe edge
    #d_top = abs(self.last_player_y - self.last_pipe_top)
    #d_bottom = abs(self.last_pipe_bottom - self.last_player_y)
    #min_dist = min(d_top, d_bottom)

    #threshold = 20  # pixels

    #if min_dist < threshold:
        #edge_bonus = (threshold - min_dist) / threshold
        #shaped_reward += edge_bonus * 0.3  # small multiplier

    #return shaped_reward