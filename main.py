from enum import Enum

from game import Game
from creature_action import Action


class Modes(Enum):
    PLAY = 0
    SOLVE = 1


def main():
    mode = Modes.SOLVE

    env = Game(6, 3, 10, 20, 1, 5)

    if mode == Modes.SOLVE:
        epsilon = 1e-5
        res, opt_policy = env.value_iteration(epsilon)

        for i, s in enumerate(env.state_space[:20]):
            a = Action(opt_policy[i])
            print(f"Optimal policy for state {s} is {a.name}.")

        print(f"Found opt avg gain = {res}.")

    else:
        env.plot_state()

        action_list = [
            "BR",
            *["AR" for _ in range(2)],
            *["BT" for _ in range(2)],
            "AT",
            "AD",
            *["BR" for _ in range(2)],
            "BT",
            "AT",
            "AD",
        ]
        action_list = [Action[a] for a in action_list]

        for a in action_list:
            print(f"Playing action {a.name}.")
            env.play(a)
            env.plot_state()


main()
