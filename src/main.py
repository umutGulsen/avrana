import numpy as np

from Agent import Agent
from Simulation import Simulation
from System import System


def square_diff(x1, x2):
    return (x1 - x2) ** 2


def main() -> None:

    sim_params = {
        "simulation_duration": 100,
    }
    state_count = 3
    system_params = {
        "state_count": state_count,
        "state_name_list": ["A", "B", "C"],
        "action_count": 2,
        "state_penalty_functions": [square_diff for _ in range(state_count)]
    }

    effect_matrix = .01 * np.random.randn(system_params["state_count"], system_params["state_count"])
    #np.fill_diagonal(effect_matrix, 0)
    system_matrices = {
        "effect_matrix": effect_matrix,
        "state_vector": np.random.randn(system_params["state_count"], 1),
        "state_penalty_vector": np.ones((system_params["state_count"], 1))
    }
    sim = Simulation(**sim_params)
    s = System(**system_params, **system_matrices)
    a = Agent()
    s.random_init()
    sim.initialize_simulation(sys_avr=s, agent=a)
    sim.run_simulation(s)
    sim.draw_state_history(s)
    print(repr(s))
    print(repr(sim))


if __name__ == '__main__':
    main()
