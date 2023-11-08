import Nodes


def main():
    env = Nodes.Environement(6, 3, 10, 5, 2, 4)

    env.birthTiger()
    env.birthRabbit()
    env.activeRabbits()
    env.activeTigers()

    for n in env.nodes:
        n.printNode()

    env.printReward()
    env.activeDinosaurs()

    for n in env.nodes:
        n.printNode()

    env.printReward()


main()
