import Nodes


def printNodes(env):
    print("####################")
    for n in env.nodes:
        n.printNode()


def main():
    env = Nodes.Environement(6, 3, 10, 5, 2, 4)
    printNodes(env)

    env.birthRabbit()

    printNodes(env)

    env.activeRabbits()
    env.activeRabbits()
    printNodes(env)

    env.birthTiger()
    env.birthTiger()
    env.birthTiger()
    env.birthTiger()
    printNodes(env)

    env.activeTigers()
    printNodes(env)


main()
