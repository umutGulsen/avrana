from dataclasses import dataclass, field

import numpy as np

from Dynamic import Dynamic


@dataclass
class System:
    state_count: int = 1
    action_count: int = 1
    system_wellness: float = 0.0
    action_delay: int = 0

    effect_matrix: np.ndarray = None
    state_vector: np.ndarray = None
    action_effect_matrix: np.ndarray = None
    action_cost_vector: np.ndarray = None

    state_target_vector: np.ndarray = None
    state_penalty_vector: np.ndarray = None
    state_penalty_functions: list = field(default_factory=list)
    state_name_list: list = field(default_factory=list)
    can_amend_future_Actions: bool = False
    lower_state_constraints: np.ndarray = None
    upper_state_constraints: np.ndarray = None

    extra_effects = []

    def __post_init__(self):
        self.effect_matrix = np.zeros((self.state_count, self.state_count)) if self.effect_matrix is None else self.effect_matrix
        self.state_vector = np.zeros((self.state_count, 1)) if self.state_vector is None else self.state_vector
        self.action_effect_matrix = 0*np.zeros((self.state_count, self.action_count)) if self.action_effect_matrix is None else self.action_effect_matrix
        self.action_cost_vector = np.zeros((self.action_count, 1)) if self.action_cost_vector is None else self.action_cost_vector
        self.state_target_vector = np.zeros((self.state_count, 1)) if self.state_target_vector is None else self.state_target_vector
        self.state_penalty_vector = np.zeros((self.state_count, 1)) if self.state_penalty_vector is None else self.state_penalty_vector
        self.state_name_list = [i for i in range(self.state_count)] if self.state_name_list is None else self.state_name_list
        self.lower_state_constraints = .1+np.zeros((self.state_count, 1)) if self.lower_state_constraints is None else self.lower_state_constraints
        self.upper_state_constraints = 20 * np.ones((self.state_count, 1)) if self.upper_state_constraints is None else self.upper_state_constraints

    def random_init(self):
        self.effect_matrix = np.random.randn(self.state_count, self.state_count) if self.effect_matrix is None else self.effect_matrix
        self.state_vector = np.random.randn(self.state_count, 1) if self.state_vector is None else self.state_vector
        self.action_effect_matrix = 0 * np.random.randn(self.state_count, self.action_count)
        self.action_cost_vector = np.random.randn(self.action_count, 1)
        self.state_target_vector = np.random.randn(self.state_count, 1) if self.state_target_vector is None else self.state_target_vector
        self.state_penalty_vector = np.random.randn(self.state_count, 1) if self.state_penalty_vector is None else self.state_penalty_vector

    def insert_effect(self, dynamic: Dynamic):
        self.extra_effects.append(dynamic)

    def calculate_dynamic_effects(self):
        total_delta = np.zeros((self.state_count, 1))
        for dynamic in self.extra_effects:
            total_delta += dynamic.run_effect(self.state_vector)
        return total_delta
