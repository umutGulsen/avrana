import numpy as np

from Agent import Agent
from Dynamic import Dynamic
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
        "state_name_list": ["Fox", "Rabbit", "Grass"],
        "action_count": 2,
        "state_penalty_functions": [square_diff for _ in range(state_count)]
    }
    two_var_effect_functions = []



    def prey_on_pred(prey, pred, d=1):
        return prey * pred * d

    def pred_on_prey(prey, pred):
        return - prey * pred

    def reproduction(x):
        return x

    def competition(x, alpha=.5):
        return -x * alpha

    def on_rabbit(state):
        fox = state[0]
        rabbit = state[1]
        grass = state[2]
        fox_effect = pred_on_prey(prey=rabbit, pred=fox)
        rabbit_effect = reproduction(x=rabbit) + competition(x=rabbit, alpha=1.2)
        grass_effect = prey_on_pred(prey=grass, pred=rabbit)
        return fox_effect + rabbit_effect + grass_effect

    def on_grass(state):
        rabbit = state[1]
        grass = state[2]
        rabbit_effect = pred_on_prey(prey=grass, pred=rabbit)
        grass_effect = reproduction(x=grass)
        return rabbit_effect + grass_effect

    def on_fox(state):
        fox = state[0]
        rabbit = state[1]
        rabbit_effect = prey_on_pred(prey=rabbit, pred=fox)
        fox_effect = reproduction(x=fox) + competition(x=fox, alpha=1.5)
        return rabbit_effect + fox_effect

    effect_matrix = np.array([[0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]]).T
    #.01 * np.random.randn(system_params["state_count"], system_params["state_count"])
    np.fill_diagonal(effect_matrix, -.4)
    system_matrices = {
        "effect_matrix": effect_matrix,
        "state_vector": 1+np.random.randn(system_params["state_count"], 1),
        "state_penalty_vector": np.ones((system_params["state_count"], 1)),
        "state_target_vector": np.array([[0], [50], [0]])
    }
    sim = Simulation(**sim_params)
    s = System(**system_params, **system_matrices)
    s.insert_effect(Dynamic(dynamic_func=on_rabbit, recipient=1))
    s.insert_effect(Dynamic(dynamic_func=on_grass, recipient=2))
    s.insert_effect(Dynamic(dynamic_func=on_fox, recipient=0))
    a = Agent()
    s.random_init()
    sim.initialize_simulation(sys_avr=s, agent=a)
    sim.run_simulation(s)
    sim.draw_state_history(s)
    print(repr(s))
    print(repr(sim))


if __name__ == '__main__':
    main()
