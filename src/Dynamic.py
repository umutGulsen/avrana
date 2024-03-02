from dataclasses import dataclass


@dataclass
class Dynamic:
    dynamic_func: None
    recipient: None

    def run_effect(self, state):
        delta = state - state
        delta[self.recipient] = self.dynamic_func(state)
        return delta
