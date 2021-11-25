MOUNTAIN_CAR_ENV = 'MountainCar-v0'

MOUNTAIN_CAR = {
    'env': MOUNTAIN_CAR_ENV,
    'input_shape': (2,),
    'embedding': None,
    'seed': 200,
    'load_path' : 'models/mountain_car_ddqn_dueling.h5',
    'layers' : [16, 32, 16],
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
        'warmup_steps': 200,
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