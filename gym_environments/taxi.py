TAXI_ENV = 'Taxi-v3'

TAXI = {
    'env': TAXI_ENV,
    'input_shape': (1,),
    'embedding': [500, 6, (6,)],
    'seed': 200,
    'load_path' : 'models/taxi_ddqn_dueling.h5',
    'rewards_path': 'rewards/taxi_ddqn_dueling.json',
    'plot_path': 'plots/taxi_ddqn_dueling.png',
    'layers' : [16, 16, 16],
    'dueling' : True,
    'success_average' : 0, # TODO
    'memory_config': {
        'max_size': 1500,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.999,
        'epsilon_min' : 0.05,
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