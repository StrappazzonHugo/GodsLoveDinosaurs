from enum import Enum
import sys

from game import Game
from creature_action import Action


class Modes(Enum):
    PLAY = 0
    SOLVE = 1


EPSILON = 1e-4
DEFAULT_PARAMS = (6, 3, 30, 200, 30, 50)


def optimal_gain_gld(N, K, W, L, CR, CT):
    env = Game(N, K, W, L, CR, CT)

    g_star, _ = env.value_iteration(EPSILON)

    return g_star


# def check_optimal_gain_gld(N, K, W, L, CR, CT):


def play_gld(N, K, W, L, CR, CT):
    env = Game(N, K, W, L, CR, CT)

    env.play_interactive()


def main(argv):
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
        # epsilon = 1e-2
        g_star, opt_policy = env.value_iteration(epsilon)

        # for i, s in enumerate(env.state_space[:50]):
        #     a = Action(opt_policy[i])
        #     print(f"Optimal policy for state {s} is {a.name}.")

        print(f"Found optimal average gain g* = {g_star}.")

        env.replay_solution(100000, opt_policy, g_star)

    else:
        env.play_interactive()

        # action_list = [
        #     "BR",
        #     *["AR" for _ in range(2)],
        #     *["BT" for _ in range(2)],
        #     "AT",
        #     "AD",
        #     *["BR" for _ in range(2)],
        #     "BT",
        #     "AT",
        #     "AD",
        # ]
        # action_list = [Action[a] for a in action_list]

        # for a in action_list:
        #     print(f"Playing action {a.name}.")
        #     env.play(a)
        #     env.plot_state()


main(sys.argv)
