TAXI_ENV = 'Taxi-v3'

TAXI = {
    'name': 'Taxi Duelling',
    'env': TAXI_ENV,
    'input_shape': (1,),
    'embedding': [500, 20, (20,)],
    'seed': 200,
    'weight_path' : 'models/taxi_ddqn_dueling.hdf5',
    'model_path' : 'models/taxi_ddqn_dueling.h5',
    'rewards_path': 'rewards/taxi_ddqn_dueling.json',
    'plot_path': 'plots/taxi_ddqn_dueling.png',
    'checkpoint_path': 'checkpoints/taxi_ddqn_dueling.json',
    'layers' : [32, 64, 16],
    'dueling' : True,
    'success_average' : 8,
    'avg_plot' : True,
    'avg_window_plot': 100,
    'memory_config': {
        'max_size': 40000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.9999,
        'epsilon_min' : 0.05,
    },
    'agent_config': {
        'warmup_steps': 500,
        'target_model_update': 10,
    },
    'train_config': {
        'max_steps': 60000,
        'batch_size': 32,
        'gamma': 0.9,
    },
    'test_config': {
        'n_tests': 1,
    },
}

TAXI_NO_DUELLING = {
    'name': 'Taxi NO Duelling',
    'env': TAXI_ENV,
    'input_shape': (1,),
    'embedding': [500, 20, (20,)],
    'seed': 200,
    'weight_path' : 'models/taxi_ddqn.hdf5',
    'model_path' : 'models/taxi_ddqn.h5',
    'rewards_path': 'rewards/taxi_ddqn.json',
    'plot_path': 'plots/taxi_ddqn.png',
    'checkpoint_path': 'checkpoints/taxi_ddqn.json',
    'layers' : [32, 64, 16],
    'dueling' : False,
    'success_average' : 8,
    'avg_plot' : True,
    'avg_window_plot': 100,
    'memory_config': {
        'max_size': 40000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.9999,
        'epsilon_min' : 0.05,
    },
    'agent_config': {
        'warmup_steps': 500,
        'target_model_update': 10,
    },
    'train_config': {
        'max_steps': 60000,
        'batch_size': 32,
        'gamma': 0.9,
    },
    'test_config': {
        'n_tests': 1,
    },
}