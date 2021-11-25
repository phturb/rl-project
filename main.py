
from agent import run_from_config
import keras.backend as K
from gym_environments.blackjack import BLACKJACK, BLACKJACK_ENV
from gym_environments.cartpole import CARTPOLE, CARTPOLE_ENV
from gym_environments.frozen_lake import FROZEN_LAKE, FROZEN_LAKE_ENV
from gym_environments.mountain_car import MOUNTAIN_CAR, MOUNTAIN_CAR_ENV
from gym_environments.taxi import TAXI, TAXI_ENV

configs = {}

configs[CARTPOLE_ENV] = CARTPOLE

configs[BLACKJACK_ENV] = BLACKJACK

configs[MOUNTAIN_CAR_ENV] = MOUNTAIN_CAR

configs[TAXI_ENV] = TAXI

configs[FROZEN_LAKE_ENV] = FROZEN_LAKE

if __name__ == "__main__":

    run_from_config(configs[CARTPOLE_ENV])

    run_from_config(configs[BLACKJACK_ENV])

    run_from_config(configs[MOUNTAIN_CAR_ENV])

    run_from_config(configs[TAXI_ENV])

    run_from_config(configs[FROZEN_LAKE_ENV])