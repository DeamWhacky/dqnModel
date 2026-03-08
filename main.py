import warnings
warnings.filterwarnings("ignore")
from Train import train, watch
import sys

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch()
    else:
        train()