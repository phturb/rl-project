
from agent import plot_compare_rewards,plot_avg_rewards, run_from_config
import keras.backend as K
from gym_environments.blackjack import BLACKJACK, BLACKJACK_NO_DUELLING
from gym_environments.cartpole import CARTPOLE, CARTPOLE_NO_DUELLING
from gym_environments.frozen_lake import FROZEN_LAKE, FROZEN_LAKE_NO_DUELLING
from gym_environments.mountain_car import MOUNTAIN_CAR, MOUNTAIN_CAR_NO_DUELLING
from gym_environments.freeway import FREEWAY, FREEWAY_NO_DUELLING
from gym_environments.taxi import TAXI, TAXI_NO_DUELLING

if __name__ == "__main__":

    _, freeway_reward, _ = run_from_config(FREEWAY)
    plot_avg_rewards(freeway_reward, FREEWAY)

    _, freeway_no_duelling_reward, _ = run_from_config(FREEWAY_NO_DUELLING)
    plot_avg_rewards(freeway_no_duelling_reward,FREEWAY_NO_DUELLING)

    plot_compare_rewards([freeway_reward, freeway_no_duelling_reward], [FREEWAY, FREEWAY_NO_DUELLING], "./plots/freeway_compare.png")

    _, blackjack_reward, _ = run_from_config(BLACKJACK)
    plot_avg_rewards(blackjack_reward, BLACKJACK)

    _, blackjack_no_duelling_reward, _ = run_from_config(BLACKJACK_NO_DUELLING)
    plot_avg_rewards(blackjack_no_duelling_reward,BLACKJACK_NO_DUELLING)

    plot_compare_rewards([blackjack_reward, blackjack_no_duelling_reward], [BLACKJACK, BLACKJACK_NO_DUELLING], "./plots/blackjack_compare.png")

    _, cartpole_reward, _ = run_from_config(CARTPOLE)
    plot_avg_rewards(cartpole_reward, CARTPOLE)

    _, cartpole_no_duelling_reward, _ = run_from_config(CARTPOLE_NO_DUELLING)
    plot_avg_rewards(cartpole_no_duelling_reward, CARTPOLE_NO_DUELLING)

    plot_compare_rewards([cartpole_reward, cartpole_no_duelling_reward], [CARTPOLE, CARTPOLE_NO_DUELLING], "./plots/cartpole_compare.png")

    # _, mountain_car_reward, _ = run_from_config(MOUNTAIN_CAR)
    # plot_avg_rewards(mountain_car_reward, MOUNTAIN_CAR)

    # _, mountain_car_no_duelling_reward, _ = run_from_config(MOUNTAIN_CAR_NO_DUELLING)
    # plot_avg_rewards(mountain_car_no_duelling_reward, MOUNTAIN_CAR_NO_DUELLING)

    # plot_compare_rewards([mountain_car_reward, mountain_car_no_duelling_reward], [MOUNTAIN_CAR, MOUNTAIN_CAR_NO_DUELLING], "./plots/mountain_car_compare.png")

    _, taxi_reward, _ = run_from_config(TAXI)
    plot_avg_rewards(taxi_reward, TAXI)

    _, taxi_no_duelling_reward, _ = run_from_config(TAXI_NO_DUELLING)
    plot_avg_rewards(taxi_no_duelling_reward, TAXI_NO_DUELLING)

    plot_compare_rewards([taxi_reward, taxi_no_duelling_reward], [TAXI, TAXI_NO_DUELLING], "./plots/taxi_compare.png")

    _, frozen_lake_reward, _ = run_from_config(FROZEN_LAKE)
    plot_avg_rewards(frozen_lake_reward, FROZEN_LAKE)

    _, frozen_lake_no_duelling_reward, _ = run_from_config(FROZEN_LAKE_NO_DUELLING)
    plot_avg_rewards(frozen_lake_no_duelling_reward, FROZEN_LAKE_NO_DUELLING)

    plot_compare_rewards([frozen_lake_reward, frozen_lake_no_duelling_reward], [FROZEN_LAKE, FROZEN_LAKE_NO_DUELLING], "./plots/frozen_lake_compare.png")