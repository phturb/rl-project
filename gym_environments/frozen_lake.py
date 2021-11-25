FROZEN_LAKE_ENV = 'FrozenLake-v1'

FROZEN_LAKE = {
    'env': FROZEN_LAKE_ENV,
    'input_shape': (1,),
    'embedding': [16, 4, (4,)],
    'seed': 200,
    'load_path' : 'models/frozen_lake_ddqn_dueling.h5',
    'rewards_path': 'rewards/frozen_lake_ddqn_dueling.json',
    'plot_path': 'plots/mountain_car_ddqn_dueling.png',
    'layers' : [16, 32, 16],
    'dueling' : True,
    'success_average' : 0.9,
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
        'n_tests': 100,
    },
}