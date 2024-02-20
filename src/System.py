from dataclasses import dataclass

import numpy as np


@dataclass
class System:
    state_count: int = 1
    action_count: int = 1
    simulation_duration: int = 1
    system_wellness: float = 0.0
    history_interval: int = 1
    action_delay: int = 0
    effect_matrix: np.ndarray = None
    state_vector: np.ndarray = None
    state_history: np.ndarray = None
    action_effect_matrix: np.ndarray = None
    action_cost_vector: np.ndarray = None
    action_history: np.ndarray = None
    state_target_vector: np.ndarray = None
    state_penalty_vector: np.ndarray = None
    state_penalty_functions: list = None
    can_amend_future_Actions: bool = False
    clock: int = 0

    def __post_init__(self):
        self.effect_matrix = np.zeros((self.state_count, self.state_count)) if self.effect_matrix is None else self.effect_matrix
        self.state_vector = np.zeros((self.state_count, 1)) if self.state_vector is None else self.state_vector
        self.state_history = np.zeros((self.state_count, 1 + self.simulation_duration // self.history_interval))
        self.action_effect_matrix = np.zeros((self.state_count, self.action_count)) if self.action_effect_matrix is None else self.action_effect_matrix
        self.action_cost_vector = np.zeros((self.action_count, 1)) if self.action_cost_vector is None else self.action_cost_vector
        self.action_history = np.zeros((self.action_count, self.simulation_duration))
        self.state_target_vector = np.zeros((self.state_count, 1)) if self.state_target_vector is None else self.state_target_vector
        self.state_penalty_vector = np.zeros((self.state_count, 1)) if self.state_penalty_vector is None else self.state_penalty_vector
        self.state_penalty_functions = [] if self.state_penalty_functions is None else self.state_penalty_functions
        
    def random_init(self):
        self.effect_matrix = np.random.randn(self.state_count, self.state_count) if self.effect_matrix is None else self.effect_matrix
        self.state_vector = np.random.randn(self.state_count, 1) if self.state_vector is None else self.state_vector
        self.action_effect_matrix = np.random.randn(self.state_count, self.action_count) if self.action_effect_matrix is None else self.action_effect_matrix
        self.action_cost_vector = np.random.randn(self.action_count, 1) if self.action_cost_vector is None else self.action_cost_vector
        self.state_target_vector = np.random.randn(self.state_count, 1) if self.state_target_vector is None else self.state_target_vector
        self.state_penalty_vector = np.random.randn(self.state_count, 1) if self.state_penalty_vector is None else self.state_penalty_vector

    def tick(self):
        if self.clock < self.simulation_duration:
            self.state_history[:, self.clock] = self.state_vector[:, 0]
            inherent_change_vector = np.dot(self.effect_matrix, self.state_vector)
            action_change_vector = np.dot(self.action_effect_matrix, self.action_history[:, [self.clock]])
            change_vector = inherent_change_vector + action_change_vector
            self.state_vector += change_vector
            self.clock += 1
            return True
        else:
            self.state_history[:, self.clock] = self.state_vector[:, 0]
            return False

    def run_simulation(self):
        keep_running = True
        while keep_running:
            keep_running = self.tick()
