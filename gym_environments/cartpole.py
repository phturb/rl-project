CARTPOLE_ENV = 'CartPole-v1'

CARTPOLE = {
    'env': CARTPOLE_ENV,
    'embedding': None,
    'seed': 200,
    'input_shape': (4,),
    'load_path' : 'models/cart_pole_ddqn_dueling.h5',
    'layers' : [16, 32, 16],
    'dueling' : True,
    'success_average' : 195,
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