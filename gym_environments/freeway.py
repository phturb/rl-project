FREEWAY_ENV = 'minatar-freeway'

FREEWAY = {
    'name' : "Freeway Duelling",
    'env': FREEWAY_ENV,
    'input_shape': (10,10,7),
    'embedding': None,
    'action_space': 6,
    'seed': 200,
    'weight_path' : 'models/freeway_ddqn_dueling.hdf5',
    'model_path' : 'models/freeway_ddqn_dueling.h5',
    'rewards_path': 'rewards/freeway_ddqn_dueling.json',
    'plot_path': 'plots/freeway_ddqn_dueling.png',
    'layers' : [64, 128],
    'dueling' : True,
    'success_average' : 20,
    'avg_plot' : True,
    'avg_window_plot': 1000,
    'memory_config': {
        'max_size': 10000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.9999,
        'epsilon_min' : 0.1,
    },
    'agent_config': {
        'warmup_steps': 10000,
        'target_model_update': 10,
    },
    'train_config': {
        'max_steps': 1_000_000,
        'batch_size': 32,
        'gamma': 0.99,
    },
    'test_config': {
        'n_tests': 100,
    },
}

FREEWAY_NO_DUELLING = {
    'name' : "Freeway No Duelling",
    'env': FREEWAY_ENV,
    'input_shape': (10,10,7),
    'embedding': None,
    'action_space': 6,
    'seed': 200,
    'weight_path' : 'models/freeway_ddqn.hdf5',
    'model_path' : 'models/freeway_ddqn.h5',
    'rewards_path': 'rewards/freeway_ddqn.json',
    'plot_path': 'plots/freeway_ddqn.png',
    'layers' : [64, 128],
    'dueling' : False,
    'success_average' : 20,
    'avg_plot' : True,
    'avg_window_plot': 1000,
    'memory_config': {
        'max_size': 10000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.9999,
        'epsilon_min' : 0.1,
    },
    'agent_config': {
        'warmup_steps': 10000,
        'target_model_update': 10,
    },
    'train_config': {
        'max_steps': 1_000_000,
        'batch_size': 32,
        'gamma': 0.99,
    },
    'test_config': {
        'n_tests': 100,
    },
}