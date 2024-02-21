from dataclasses import dataclass

import numpy as np


@dataclass
class Agent:
    id: int = -1

    def generate_single_random_action(self, action_count):

        action_vector = np.zeros((action_count, 1))
        action_vector[np.random.randint(low=0, high=action_count), 0] = 1
        return action_vector
