""""""

from creature_action import Creature


class State:
    """
    State of the N nodes of the board.
    Wraps a list of nodes and provides some helper functions.
    """

    def __init__(self, nodes):
        self._nodes = nodes

    @property
    def nodes(self):
        """Nodes of the board, indexed and containing a Creature."""
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        from node import Node

        if not isinstance(nodes, list):
            raise TypeError(
                f"'nodes' argument should be of type 'List[Node]' "
                f"but found '{type(nodes).__name__}'."
            )
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError(
                    f"'nodes' argument should be of type 'List[Node]' "
                    f"but found 'List[{type(node).__name__}]'."
                )

        self._nodes = nodes

    @property
    def N(self):
        """Number N of Nodes on the board."""
        return len(self.nodes)

    def __getitem__(self, i):
        return self.nodes[i]

    def copy(self):
        """Copies the State, recursively."""
        return State([node.copy() for node in self.nodes])

    def next_node(self, start_node, steps=1):
        """
        Go to another node 'steps' steps further.
        Positive 'steps' is clockwise.
        """

        from node import Node

        if not isinstance(start_node, Node):
            raise TypeError(
                f"'start_node' argument should be of type 'Node' "
                f"but found '{type(start_node).__name__}'."
            )
        if not isinstance(steps, int):
            raise TypeError(
                f"'steps' argument should be of type 'int' but "
                f"found '{type(steps).__name__}'."
            )

        return self.nodes[(start_node.id + steps + self.N) % self.N]

    def check_empty_node(self, id):
        if not isinstance(id, int):
            raise TypeError(
                f"'id' argument should be of type 'int' but "
                f"found '{type(id).__name__}'."
            )
        if id < 0 or id >= self.N:
            raise ValueError(
                f"'id' argument should be in [0, {self.N - 1}] "
                f"but found '{id}'."
            )

        return self.nodes[id].creature == Creature.Empty

    def get_nodes_containing(self, creature):
        '''Lists all Nodes containing a given Creature.'''
        if not isinstance(creature, Creature):
            raise TypeError(
                f"'creature' argument should be a 'Creature' Enum "
                f"member but found '{type(creature).__name__}'."
            )

        return [n for n in self.nodes if n.creature == creature]

    def get_empty(self):
        '''Lists all Empty Nodes.'''
        return self.get_nodes_containing(Creature.Empty)

    def get_rabbits(self):
        '''Lists all Rabbit Nodes.'''
        return self.get_nodes_containing(Creature.Rabbit)

    def get_tigers(self):
        '''Lists all Tiger Nodes.'''
        return self.get_nodes_containing(Creature.Tiger)

    def check_any_empty_node(self):
        '''Checks if any node is Empty.'''
        for n in self.nodes:
            if n.creature == Creature.Empty:
                return True
        return False

    def count_creature(self, creature):
        '''Counts the nodes containing a given Creature.'''
        if not isinstance(creature, Creature):
            raise TypeError(
                f"'creature' argument should be a 'Creature' Enum "
                f"member but found '{type(creature).__name__}'."
            )
        return len(self.get_nodes_containing(creature))

    def count_empty(self):
        '''Counts the Empty nodes.'''
        return self.count_creature(Creature.Empty)

    def count_rabbit(self):
        '''Counts the Rabbit nodes.'''
        return self.count_creature(Creature.Rabbit)

    def count_tiger(self):
        '''Counts the Tiger nodes.'''
        return self.count_creature(Creature.Tiger)

    # Printing and equality implementations.

    def __str__(self):
        string = "["
        for node in self.nodes:
            string += f"{node.__str__()}, "
        string = string[:-2] + "]"
        return string

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.nodes == other.nodes
