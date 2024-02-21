import numpy as np

from Agent import Agent
from Simulation import Simulation
from System import System


def main():

    sim_params = {
        "simulation_duration": 1000,
    }

    system_params = {
        "state_count": 3,
        }
    effect_matrix = .1 * np.random.randn(system_params["state_count"], system_params["state_count"])
    np.fill_diagonal(effect_matrix, 0)
    system_matrices = {
        "effect_matrix": effect_matrix,
        "state_vector": np.random.randn(system_params["state_count"], 1),
        "state_penalty_vector": np.ones((system_params["state_count"], 1))
    }
    sim = Simulation(**sim_params)
    s = System(**system_params, **system_matrices)
    a = Agent()
    s.random_init()
    sim.initialize_simulation(s)
    sim.run_simulation(s)
    sim.draw_state_history()
    print(repr(s))
    print(repr(sim))


if __name__ == '__main__':
    main()
