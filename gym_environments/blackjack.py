BLACKJACK_ENV = 'Blackjack-v1'

BLACKJACK = {
    'env': BLACKJACK_ENV,
    'embedding': None,
    'input_shape': (3,),
    'seed': 200,
    'success_average' : -110,
    'model_path' : 'models/blackjack_ddqn_dueling.h5',
    'rewards_path': 'rewards/blackjack_ddqn_dueling.json',
    'plot_path': 'plots/blackjack_ddqn_dueling.png',
    'layers' : [16, 32, 16],
    'dueling' : True,
    'memory_config': {
        'max_size': 1500,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.999,
        'epsilon_min' : 0.01,
    },
    'agent_config': {
        'warmup_steps': 32,
        'target_model_update': 5,
    },
    'train_config': {
        'max_steps': 50000,
        'batch_size': 32,
        'gamma': 0.975,
    },
    'test_config': {
        'n_tests': 100,
    },
}