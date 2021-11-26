
from agent import plot_rewards, run_from_config
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

    _, cartpole_reward, _ = run_from_config(configs[CARTPOLE_ENV])
    plot_rewards(cartpole_reward, CARTPOLE_ENV, configs[CARTPOLE_ENV]['plot_path'])

    _, blackjack_reward, _ = run_from_config(configs[BLACKJACK_ENV])
    plot_rewards(blackjack_reward, BLACKJACK_ENV, configs[BLACKJACK_ENV]['plot_path'])

    _, mountain_car_reward, _ = run_from_config(configs[MOUNTAIN_CAR_ENV])
    plot_rewards(mountain_car_reward, MOUNTAIN_CAR_ENV, configs[MOUNTAIN_CAR_ENV]['plot_path'])

    _, taxi_reward, _ = run_from_config(configs[TAXI_ENV])
    plot_rewards(taxi_reward, TAXI_ENV, configs[TAXI_ENV]['plot_path'])

    _, frozen_lake_reward, _ = run_from_config(configs[FROZEN_LAKE_ENV])
    plot_rewards(frozen_lake_reward, FROZEN_LAKE_ENV, configs[FROZEN_LAKE_ENV]['plot_path'])