
from agent import plot_compare_rewards,plot_avg_rewards, run_from_config
import keras.backend as K
from gym_environments.blackjack import BLACKJACK, BLACKJACK_NO_DUELLING
from gym_environments.cartpole import CARTPOLE, CARTPOLE_NO_DUELLING
from gym_environments.frozen_lake import FROZEN_LAKE, FROZEN_LAKE_NO_DUELLING
from gym_environments.mountain_car import MOUNTAIN_CAR, MOUNTAIN_CAR_NO_DUELLING
from gym_environments.freeway import FREEWAY, FREEWAY_NO_DUELLING
from gym_environments.taxi import TAXI, TAXI_NO_DUELLING


def run_experience(duelling, non_duelling, plot_compare_path):
    _, reward, _ = run_from_config(duelling)
    plot_avg_rewards(reward, duelling)

    _, no_duelling_reward, _ = run_from_config(non_duelling)
    plot_avg_rewards(no_duelling_reward, non_duelling)

    plot_compare_rewards([reward, no_duelling_reward], [duelling, non_duelling], plot_compare_path)

if __name__ == "__main__":
    # run_experience(FREEWAY, FREEWAY_NO_DUELLING,  "./plots/freeway_compare.png")
    run_experience(CARTPOLE, CARTPOLE_NO_DUELLING,  "./plots/cartpole_compare.png")
    run_experience(BLACKJACK, BLACKJACK_NO_DUELLING,  "./plots/blackjack_compare.png")
    run_experience(TAXI, TAXI_NO_DUELLING,  "./plots/taxi_compare.png")
    # run_experience(MOUNTAIN_CAR, MOUNTAIN_CAR_NO_DUELLING,  "./plots/mountain_car_compare.png")
    run_experience(FROZEN_LAKE, FROZEN_LAKE_NO_DUELLING,  "./plots/frozen_lake_compare.png")