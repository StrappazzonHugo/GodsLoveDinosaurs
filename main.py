from enum import Enum

from game import Game
from creature_action import Action

class Modes(Enum):
    PLAY = 0
    SOLVE = 1


def main():
    mode = Modes.SOLVE

    env = Game(6, 3, 10, 100, 2, 20)

    if mode == Modes.SOLVE:
        epsilon = 1e-2
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


main()
