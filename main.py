from enum import Enum
import sys

from game import Game


class Modes(Enum):
    PLAY = 0
    SOLVE = 1


EPSILON = 1e-4
DEFAULT_PARAMS = (6, 3, 30, 200, 30, 50)
# The default parameters are arbitrary values


def optimal_gain_gld(N, K, W, L, CR, CT):
    """Launch the solve version of the game computing the optimal policy """
    env = Game(N, K, W, L, CR, CT)

    g_star, _ = env.value_iteration(EPSILON)

    return g_star


def play_gld(N, K, W, L, CR, CT):
    """Launch the interactive version of the game"""
    env = Game(N, K, W, L, CR, CT)

    env.play_interactive()


def main(argv):
    """The default parameters are used if no parameter is specified when calling the main function"""
    print(
        "This script can be run using "
        "'$ python3 ./main.py [play, solve] [N K W L CR CT] [epsilon]'.\n"
    )

    if len(argv) < 2:
        mode = Modes.SOLVE
    else:
        if argv[1] == "solve":
            mode = Modes.SOLVE
        elif argv[1] == "play":
            mode = Modes.PLAY
        else:
            raise ValueError(
                "First optional command-line argument should be 'solve' or 'play'."
            )

    if len(argv) >= 8:
        params = tuple(int(arg) for arg in argv[2:8])
    else:
        params = DEFAULT_PARAMS

    if len(argv) == 9:
        epsilon = float(argv[8])
    else:
        epsilon = EPSILON

    print(
        f"Running in {mode.name} mode with parameters {params} "
        + f"and epsilon = {epsilon}"
        if mode == Modes.SOLVE
        else ""
    )

    env = Game(*params)

    if mode == Modes.SOLVE:
        g_star, opt_policy = env.value_iteration(epsilon)

        print(f"Found optimal average gain g* = {g_star}.")

        env.replay_solution(100000, opt_policy, g_star)

    else:
        env.play_interactive()


main(sys.argv)
