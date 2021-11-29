FROZEN_LAKE_ENV = 'FrozenLake-v1'

FROZEN_LAKE = {
    'name': 'Frozen Lake Duelling',
    'env': FROZEN_LAKE_ENV,
    'input_shape': (1,),
    'embedding': [16, 4, (4,)],
    'seed': 200,
    'model_path' : 'models/frozen_lake_ddqn_dueling.h5',
    'rewards_path': 'rewards/frozen_lake_ddqn_dueling.json',
    'plot_path': 'plots/frozen_lake_ddqn_dueling.png',
    'layers' : [16, 32, 16],
    'dueling' : True,
    'success_average' : 0.7,
    'avg_plot' : True,
    'avg_window_plot': 100,
    'memory_config': {
        'max_size': 5000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.9995,
        'epsilon_min' : 0.10,
    },
    'agent_config': {
        'warmup_steps': 64,
        'target_model_update': 5,
    },
    'train_config': {
        'max_steps': 50000,
        'batch_size': 32,
        'gamma': 0.975,
    },
    'test_config': {
        'n_tests': 1,
    },
}

FROZEN_LAKE_NO_DUELLING = {
    'name': 'Frozen Lake No Duelling',
    'env': FROZEN_LAKE_ENV,
    'input_shape': (1,),
    'embedding': [16, 4, (4,)],
    'seed': 200,
    'model_path' : 'models/frozen_lake_ddqn.h5',
    'rewards_path': 'rewards/frozen_lake_ddqn.json',
    'plot_path': 'plots/frozen_lake_ddqn.png',
    'layers' : [16, 32, 16],
    'dueling' : False,
    'success_average' : 0.7,
    'avg_plot' : True,
    'avg_window_plot': 100,
    'memory_config': {
        'max_size': 5000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.9995,
        'epsilon_min' : 0.10,
    },
    'agent_config': {
        'warmup_steps': 64,
        'target_model_update': 5,
    },
    'train_config': {
        'max_steps': 50000,
        'batch_size': 32,
        'gamma': 0.975,
    },
    'test_config': {
        'n_tests': 1,
    },
}