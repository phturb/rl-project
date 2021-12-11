BLACKJACK_ENV = 'Blackjack-v1'

BLACKJACK = {
    'name' : "Blackjack Duelling",
    'env': BLACKJACK_ENV,
    'input_shape': (3,),
    'embedding': None,
    'seed': 200,
    'weight_path' : 'models/blackjack_ddqn_dueling.hdf5',
    'model_path' : 'models/blackjack_ddqn_dueling.h5',
    'rewards_path': 'rewards/blackjack_ddqn_dueling.json',
    'plot_path': 'plots/blackjack_ddqn_dueling.png',
    'checkpoint_path': 'checkpoints/blackjack_ddqn_dueling.json',
    'layers' : [32, 64, 16],
    'dueling' : True,
    'success_average' : 0,
    'avg_plot' : True,
    'avg_window_plot': 1000,
    'memory_config': {
        'max_size': 1000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.999,
        'epsilon_min' : 0.1,
    },
    'agent_config': {
        'warmup_steps': 1000,
        'target_model_update': 5,
    },
    'train_config': {
        'max_steps': 10000,
        'batch_size': 32,
        'gamma': 0.9,
    },
    'test_config': {
        'n_tests': 1,
    },
}

BLACKJACK_NO_DUELLING = {
    'name' : "Blackjack No Duelling",
    'env': BLACKJACK_ENV,
    'input_shape': (3,),
    'embedding': None,
    'seed': 200,
    'weight_path' : 'models/blackjack_ddqn.hdf5',
    'model_path' : 'models/blackjack_ddqn.h5',
    'rewards_path': 'rewards/blackjack_ddqn.json',
    'plot_path': 'plots/blackjack_ddqn.png',
    'checkpoint_path': 'checkpoints/blackjack_ddqn.json',
    'layers' : [32, 64, 16],
    'dueling' : False,
    'success_average' : 0,
    'avg_plot' : True,
    'avg_window_plot': 1000,
    'memory_config': {
        'max_size': 1000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.999,
        'epsilon_min' : 0.1,
    },
    'agent_config': {
        'warmup_steps': 1000,
        'target_model_update': 5,
    },
    'train_config': {
        'max_steps': 10000,
        'batch_size': 32,
        'gamma': 0.9,
    },
    'test_config': {
        'n_tests': 100,
    },
}