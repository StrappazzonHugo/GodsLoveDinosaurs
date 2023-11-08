from enum import Enum
import random
import math
import numpy


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

    def __init__(self, N, K, W, L, CR, CT):
        self.nodes = []
        self.reward = 0
        self.nb_nodes = N
        self.K = K
        self.N = N
        self.W = W
        self.L = L
        self.CR = CR
        self.CT = CT

        for i in range(self.N):
            self.nodes.append(Node(i, 0))
        for i in range(self.N):
            self.nodes[i].set_right_nb(self.nodes[(i-1) % self.N])
            self.nodes[i].set_left_nb(self.nodes[(i+1) % self.N])


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
        self.reward -= self.CT
        if self.check_any_empty_node():
            random_id = math.floor(random.random() * self.nb_nodes)
            self.nodes[random_id].set_state(2)

    def birthRabbit(self):
        self.reward -= self.CR
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
            if self.nodes[i].get_right().get_state() == State.Empty:
                self.nodes[i].get_right().set_state(State.Rabbit)

    def activeTigers(self):
        tiger_list = []
        for n in self.nodes:
            if n.get_state() == State.Tiger:
                tiger_list.append(n.get_id())
        for i in tiger_list:
            if self.nodes[i].get_left().get_state() == State.Rabbit:
                self.nodes[i].get_left().set_state(State.Tiger)
            if self.nodes[i].get_left().get_left().get_state() == State.Rabbit:
                self.nodes[i].get_left().get_left().set_state(State.Tiger)

    def activeDinosaurs(self):
        s = numpy.random.uniform(0, self.nb_nodes, self.K)
        s = numpy.asarray(s)
        t = []
        for i in range(len(s)):
            t.append(math.floor(s[i]))
        print("s =", t)
        tigers = 0
        rabbits = 0
        empty = 0
        for i in t:
            curr = self.nodes[i]
            if curr.get_state() == State.Tiger:
                tigers += 1
            if curr.get_state() == State.Rabbit:
                rabbits += 1
            if curr.get_state() == State.Empty:
                empty += 1
            curr.set_state(State.Empty)
        if tigers == 0 and rabbits == 0:
            self.reward -= self.L
        if tigers != 0:
            self.reward += tigers*self.W

    def printReward(self):
        print("reward = ", self.reward)
