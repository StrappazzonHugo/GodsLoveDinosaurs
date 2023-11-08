import Nodes


def main():
    env = Nodes.Environement(6)

    env.birthTiger()
    env.birthRabbit()

    for n in env.nodes:
        n.printNode()


main()
