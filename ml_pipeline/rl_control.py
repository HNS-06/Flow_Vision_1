
import numpy as np
import pickle
import os
import random

class RLController:
    """
    Reinforcement Learning Agent (Q-Learning) for dynamic valve control.
    Objective: Maintain optimal water levels and meet demand.
    """

    def __init__(self, n_states=10, n_actions=3, ward_id=1):
        self.ward_id = ward_id
        self.q_table = {} # State -> [Q-values]
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1 # Exploration rate
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', f'rl_agent_ward_{ward_id}.pkl')
        
        # Actions: 0=Close, 1=Throttled(50%), 2=Open
        self.actions = [0, 1, 2]
        
    def _get_state(self, current_level, demand, time_of_day):
        """
        Discretize continuous state into a tuple key for Q-table.
        State: (Level_Bin, Demand_Bin, Time_Bin)
        """
        # 1. Level: 0-20 (Low), 20-80 (Normal), 80-100 (High)
        if current_level < 20: level_state = 0
        elif current_level < 80: level_state = 1
        else: level_state = 2
        
        # 2. Demand: <100 (Low), 100-200 (Med), >200 (High)
        if demand < 100: demand_state = 0
        elif demand < 200: demand_state = 1
        else: demand_state = 2
        
        # 3. Time: 0-6 (Night), 6-9 (Morning Peak), 9-18 (Day), 18-22 (Evening Peak), 22-24 (Night)
        if 6 <= time_of_day < 9 or 18 <= time_of_day < 22: time_state = 1 # Peak
        elif 9 <= time_of_day < 18: time_state = 2 # Day
        else: time_state = 0 # Off-peak
        
        return (level_state, demand_state, time_state)

    def get_action(self, current_level, demand, time_of_day, train=False):
        """Select action using Epsilon-Greedy policy"""
        state = self._get_state(current_level, demand, time_of_day)
        
        # Initialize state if new
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
            
        # Exploration vs Exploitation
        if train and random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state]) # Best action

    def update_q_table(self, state, action, reward, next_state):
        """Q-Learning update rule"""
        # Ensure states exist
        if state not in self.q_table: self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table: self.q_table[next_state] = np.zeros(len(self.actions))
        
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action] = new_value

    def calculate_reward(self, demand_met, overflow, cost):
        """
        Reward function
        +10 if demand met fully
        -10 if overflow
        -Cost * 0.1
        """
        reward = 0
        if demand_met: reward += 10
        else: reward -= 5 # Shortage penalty
        
        if overflow: reward -= 10
        
        reward -= cost * 0.05
        return reward

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.q_table = pickle.load(f)
            return True
        return False
