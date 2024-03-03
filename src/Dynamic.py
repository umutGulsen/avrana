from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Dynamic:
    dynamic_func: Callable
    recipient: int

    def run_effect(self, state):
        delta = state - state
        delta[self.recipient] = self.dynamic_func(state)
        return delta
