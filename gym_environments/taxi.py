TAXI_ENV = 'Taxi-v3'

TAXI = {
    'env': TAXI_ENV,
    'input_shape': (1,),
    'embedding': [500, 20, (20,)],
    'seed': 200,
    'model_path' : 'models/taxi_ddqn_dueling.h5',
    'rewards_path': 'rewards/taxi_ddqn_dueling.json',
    'plot_path': 'plots/taxi_ddqn_dueling.png',
    'layers' : [16, 32, 16],
    'dueling' : True,
    'success_average' : 9, # TODO
    'memory_config': {
        'max_size': 40000,
    },
    'policy_config' : {
        'epsilon' : 1,
        'epsilon_decay' : 0.9999,
        'epsilon_min' : 0.01,
    },
    'agent_config': {
        'warmup_steps': 500,
        'target_model_update': 10,
    },
    'train_config': {
        'max_steps': 100000,
        'batch_size': 32,
        'gamma': 0.9,
    },
    'test_config': {
        'n_tests': 100,
    },
}