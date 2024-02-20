import numpy as np

from System import System


def main():
    system_params = {
        "state_count": 3,
        "simulation_duration": 10,
        }
    effect_matrix = .01 * np.random.randn(system_params["state_count"], system_params["state_count"])
    np.fill_diagonal(effect_matrix, 0)
    system_matrices = {
        "effect_matrix": effect_matrix,
        "state_vector": np.random.randn(system_params["state_count"], 1)
    }
    s = System(**system_params, **system_matrices)

    s.random_init()
    s.run_simulation()
    print(repr(s))


if __name__ == '__main__':
    main()