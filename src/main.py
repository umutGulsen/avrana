import pickle

import numpy as np

from Agent import Agent
from Dynamic import Dynamic
from Simulation import Simulation
from System import System


def square_diff(x1, x2):
    return (x1 - x2) ** 2


def load_system(path):
    with open(path, 'rb') as load_path:
        return pickle.load(load_path)


def create_ecosystem():
    def predation(prey, predator, d=.01):
        return prey * predator * d

    def reproduction(x, K=1, growth_rate=.8):
        return growth_rate * x * (1 - x / K)

    def natural_death(x, rate=.1):
        return - x * rate

    def on_grass(state):
        rabbit = state[1]
        grass = state[2]
        grass_effect = reproduction(x=grass, growth_rate=1, K=100)
        rabbit_effect = - predation(prey=grass, predator=rabbit)
        return rabbit_effect + grass_effect

    def on_rabbit(state):
        fox = state[0]
        rabbit = state[1]
        grass = state[2]
        fox_effect = - predation(prey=rabbit, predator=fox)
        grass_effect = reproduction(x=rabbit, growth_rate=.8, K=grass)
        return fox_effect + grass_effect

    def on_fox(state):
        fox = state[0]
        rabbit = state[1]
        rabbit_effect = reproduction(x=fox, growth_rate=.2, K=rabbit)
        fox_effect = natural_death(x=fox)
        return rabbit_effect + fox_effect

    state_count = 3
    system_params = {
        "state_count": state_count,
        "state_name_list": ["Fox", "Rabbit", "Grass"],
        "action_count": 2,
        "state_penalty_functions": [square_diff for _ in range(state_count)]
    }

    effect_matrix = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]).T
    # .01 * np.random.randn(system_params["state_count"], system_params["state_count"])
    np.fill_diagonal(effect_matrix, -.4)
    system_matrices = {
        "effect_matrix": effect_matrix,
        "state_vector": .4 + .1 * np.random.randn(system_params["state_count"], 1),
        "state_penalty_vector": np.ones((system_params["state_count"], 1)),
        "state_target_vector": np.array([[0], [50], [0]])
    }
    s = System(**system_params, **system_matrices)
    s.insert_effect(Dynamic(dynamic_func=on_rabbit, recipient=1))
    s.insert_effect(Dynamic(dynamic_func=on_grass, recipient=2))
    s.insert_effect(Dynamic(dynamic_func=on_fox, recipient=0))

    s.save("../artifacts/eco.json")


def main() -> None:
    sim_params = {
        "simulation_duration": 250,
    }

    sim = Simulation(**sim_params)

    a = Agent()
    # create_ecosystem()
    s = load_system("../artifacts/eco.json")
    s.random_init()
    sim.initialize_simulation(sys_avr=s, agent=a)
    sim.run_simulation(s)
    sim.draw_state_history(s)
    print(repr(s))
    print(repr(sim))


if __name__ == '__main__':
    main()
