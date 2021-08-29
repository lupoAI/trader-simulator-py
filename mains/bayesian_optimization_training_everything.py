if __name__ == "__main__":

    from analysis.optimizer_everything import use_bayesian_optimization
    import os
    from glob import glob
    import numpy as np
    import pickle

    if not os.path.exists("../results/bayesian_optimization_training/"):
        os.mkdir("../results/bayesian_optimization_training/")

    tests = glob("../results/bayesian_optimization_training/*/")

    new_test = len(tests) + 1

    # Uncomment next line to alter some tests
    # new_test = 1

    test_path = f"../results/bayesian_optimization_training/test_{new_test}/"

    if not os.path.exists(test_path):
        os.mkdir(test_path)

    res = use_bayesian_optimization([(-np.pi, np.pi),
                                     (0, np.pi),
                                     (0, 0.05),
                                     (10, 100.999),
                                     (0.005, 0.05),
                                     (100, 1000.999),
                                     (100, 1000.999),
                                     (2, 6.999),
                                     (0.005, 0.10)],
                                    acq_func="EI",
                                    n_calls=200,
                                    n_random_starts=120,
                                    noise="gaussian",
                                    random_state=3232,
                                    save_name=test_path + "b_training_1.jpg")

    with open(test_path + 'test_results.pickle', 'wb') as test_results:
        pickle.dump(res, test_results)
