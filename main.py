import warnings
warnings.filterwarnings("ignore")

from Train import train, watch
import sys

VALID_REWARDS = ["sparse", "basic", "survival", "pipe"]

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python main.py <reward_type> [watch]")
        print("Reward types:", VALID_REWARDS)
        sys.exit(1)

    reward_type = sys.argv[1]

    if reward_type not in VALID_REWARDS:
        print("Invalid reward type.")
        print("Valid options:", VALID_REWARDS)
        sys.exit(1)

    mode = "train"

    if len(sys.argv) > 2 and sys.argv[2] == "watch":
        mode = "watch"

    if mode == "watch":
        watch(reward_type)
    else:
        train(reward_type)