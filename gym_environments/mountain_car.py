MOUNTAIN_CAR_ENV = 'MountainCar-v0'

MOUNTAIN_CAR = {
    'name': "Mountain Car Duelling",
    'env': MOUNTAIN_CAR_ENV,
    'input_shape': (2,),
    'embedding': None,
    'seed': 200,
    'model_path' : 'models/mountain_car_ddqn_dueling.h5',
    'rewards_path': 'rewards/mountain_car_ddqn_dueling.json',
    'plot_path': 'plots/mountain_car_ddqn_dueling.png',
    'layers' : [16, 32, 16],
    'dueling' : True,
    'success_average' : -110,
    'avg_plot' : True,
    'avg_window_plot': 100,
    'memory_config': {
        'max_size': 50000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.99995,
        'epsilon_min' : 0.01,
    },
    'agent_config': {
        'warmup_steps': 10000,
        'target_model_update': 8,
    },
    'train_config': {
        'max_steps': 200000,
        'batch_size': 32,
        'gamma': 0.9,
    },
    'test_config': {
        'n_tests': 1,
    },
}

MOUNTAIN_CAR_NO_DUELLING = {
    'name': "Mountain Car No Duelling",
    'env': MOUNTAIN_CAR_ENV,
    'input_shape': (2,),
    'embedding': None,
    'seed': 200,
    'model_path' : 'models/mountain_car_ddqn.h5',
    'rewards_path': 'rewards/mountain_car_ddqn.json',
    'plot_path': 'plots/mountain_car_ddqn.png',
    'layers' : [16, 32, 16],
    'dueling' : False,
    'success_average' : -110,
    'avg_plot' : True,
    'avg_window_plot': 100,
    'memory_config': {
        'max_size': 50000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.99995,
        'epsilon_min' : 0.01,
    },
    'agent_config': {
        'warmup_steps': 10000,
        'target_model_update': 8,
    },
    'train_config': {
        'max_steps': 200000,
        'batch_size': 32,
        'gamma': 0.9,
    },
    'test_config': {
        'n_tests': 1,
    },
}