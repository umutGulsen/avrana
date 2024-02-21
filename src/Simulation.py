import time
from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class Simulation:
    history_interval: int = 1
    sim_run_time: float = 0
    simulation_duration: int = 1
    state_history: np.ndarray = field(default=None, repr=False)
    system_wellness_history: np.ndarray = field(default=None, repr=False)
    action_history: np.ndarray = field(default=None, repr=False)
    clock: int = field(default=0, repr=False)
    agent = None

    def initialize_simulation(self, sys_avr, agent):
        self.state_history = np.zeros((sys_avr.state_count, 1 + self.simulation_duration // self.history_interval))
        self.system_wellness_history = np.zeros((1, 1 + self.simulation_duration // self.history_interval))
        self.action_history = np.zeros((sys_avr.action_count, 1 + self.simulation_duration + sys_avr.action_delay))
        self.agent = agent

    def update_wellness(self, sys_avr):
        penalty_sum = 0
        for i, penalty_func in enumerate(sys_avr.state_penalty_functions):
            if penalty_func is not None:
                penalty_sum += penalty_func(sys_avr.state_vector[i, 0], sys_avr.state_target_vector[i, 0])
        deviation_from_target = np.abs(sys_avr.state_vector - sys_avr.state_target_vector)
        sys_avr.system_wellness = -np.sum(sys_avr.state_penalty_vector * deviation_from_target).astype(float)
        sys_avr.system_wellness -= penalty_sum
        self.system_wellness_history[:, self.clock] = sys_avr.system_wellness

    def run_simulation(self, sys_avr):
        sim_start_time = time.time()
        keep_running = True
        while keep_running:
            keep_running = self.tick(sys_avr)
        sim_end_time = time.time()
        self.sim_run_time = sim_end_time - sim_start_time

    def tick(self, sys_avr):
        self.update_wellness(sys_avr)
        self.state_history[:, self.clock] = sys_avr.state_vector[:, 0]
        self.act(sys_avr)
        if self.clock < self.simulation_duration:
            inherent_change_vector = np.dot(sys_avr.effect_matrix, sys_avr.state_vector)
            action_change_vector = np.dot(sys_avr.action_effect_matrix, self.action_history[:, [self.clock]])
            sys_avr.state_vector += inherent_change_vector + action_change_vector
            self.clock += 1
            return True
        else:
            return False

    def act(self, sys_avr):
        action_vector = self.agent.generate_single_random_action(sys_avr.action_count)
        action_occurrence_time = self.clock + sys_avr.action_delay
        self.action_history[:, [action_occurrence_time]] = action_vector

    def draw_state_history(self):
        fig, ax = plt.subplots(nrows=3, figsize=(14, 6))
        for i, state in enumerate(self.state_history):
            ax[0].plot(state, label=i)

        for i, action in enumerate(self.action_history):
            ax[2].scatter(range(len(action)), action, label=i)

        ax[1].plot(self.system_wellness_history[0], label="Wellness")

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        fig.savefig("../artifacts/state_history.png")
