from enum import Enum
import random
import math


class State(Enum):
    Empty = 0
    Rabbit = 1
    Tiger = 2

# The class Nodes
# Initialized with no left/right neighborhood
# (automatically set in the __init__of the Environment)


class Node:
    def __init__(self, id, state,):
        self.id = id
        self.state = State(state)
        self.right_nb = None
        self.left_nb = None

# Some setter...

    def set_right_nb(self, nb):
        self.right_nb = nb

    def set_left_nb(self, nb):
        self.left_nb = nb

    def set_state(self, state):
        self.state = State(state)

# Some getter...

    def get_state(self):
        return self.state

    def get_id(self):
        return self.id

    def get_right(self):
        return self.right_nb

    def get_left(self):
        return self.left_nb

# Some printer...

    def printNode(self):
        print("id = ", self.id, " state =", self.state)
        print("right_nb =", self.right_nb.get_id(),
              " left_nb =", self.left_nb.get_id(), "\n")


# Environment class,
# current attribute : nodes -> list of node
#                     nb_nodes -> number of nodes
#                     reward -> current number of points ?? (not used yet)

class Environement:

    def __init__(self, nb_cell):
        self.nodes = []
        self.nb_nodes = nb_cell
        self.reward = 0

        for i in range(nb_cell):
            self.nodes.append(Node(i, 0))
        for i in range(nb_cell):
            self.nodes[i].set_right_nb(self.nodes[(i-1) % nb_cell])
            self.nodes[i].set_left_nb(self.nodes[(i+1) % nb_cell])


# Method need for actions brithTiger and birthRabbit
#


    def check_any_empty_node(self):
        b = False
        for n in self.nodes:
            if n.get_state() == State.Empty:
                b = True
        return b

    def check_empty_node(self, id):
        if self.nodes[id].get_state() == State.Empty:
            return True
        else:
            return False

# Every actions of the Player

    def birthTiger(self):
        if self.check_any_empty_node():
            random_id = math.floor(random.random() * self.nb_nodes)
            self.nodes[random_id].set_state(2)

    def birthRabbit(self):
        if self.check_any_empty_node():
            random_id = math.floor(random.random() * self.nb_nodes)
            self.nodes[random_id].set_state(1)

    def activeRabbits(self):
        rabbit_list = []
        for n in self.nodes:
            if n.get_state() == State.Rabbit:
                rabbit_list.append(n.get_id())
        for i in rabbit_list:
            if self.nodes[i].get_left().get_state() == State.Empty:
                self.nodes[i].get_left().set_state(State.Rabbit)
            if self.nodes[i].get_right()().get_state() == State.Empty:
                self.nodes[i].get_right().set_state(State.Rabbit)

    def activeTigers(self):
        tiger_list = []
        for n in self.nodes:
            if n.get_state() == State.Tiger:
                tiger_list.append(n.get_id())
        for i in tiger_list:
            if self.nodes[i].get_left().get_state() == State.Rabbit:
                self.nodes[i].get_left().set_state(State.Tiger)
            # Ici on vient de se rendre compte que faire get_left et i+1 c'etait pareil =)
            if self.nodes[i+2].get_state() == State.Rabbit:
                self.nodes[i+2].set_state(State.Tiger)

    def activeDinosaurs(self):
        # TODO!
