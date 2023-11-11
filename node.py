from creature_action import Creature


class Node:
    """
    Class describing a given Node on the board.
    Has an index and contains a Creature.
    """

    def __init__(self, index, creature):
        # Setters are typechecked
        self._index = index
        self._creature = creature

    # Properties and typechecked setters for attributes

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        if not isinstance(index, int):
            raise TypeError(
                f"'index' argument should be of type 'int' but "
                f"found '{type(index).__name__}'."
            )
        if index < 0:
            raise ValueError(
                f"'index' argument should be non-negative but "
                f"found '{index}'."
            )

        self._index = index

    @property
    def creature(self):
        return self._creature

    @creature.setter
    def creature(self, creature):
        if not isinstance(creature, (int, Creature)):
            raise TypeError(
                f"'creature' argument should be of type 'int' or 'Creature' but "
                f"found '{type(creature).__name__}'."
            )
        if isinstance(creature, int):
            if creature < 0 or creature > len(Creature):
                raise ValueError(
                    f"'creature' argument as an int should be in "
                    f"[0, {len(Creature)}] but found {creature}."
                )
            creature = Creature(creature)

        self._creature = creature

    # Printing functions

    @property
    def creature_symbol(self):
        """Returns symbol ' ', 'R' or 'T' for the creature."""
        if self.creature == Creature.Empty:
            return " "
        elif self.creature == Creature.Rabbit:
            return "R"
        else:
            return "T"

    def __str__(self):
        return f"({self.index}: {self.creature_symbol})"

    def __repr__(self):
        return self.__str__()

    def print_node(self):
        print("index = ", self.index, " creature =", self.creature)

    # Equality and copy implementations

    def __eq__(self, other):
        return self.index == other.index and self.creature == other.creature

    def copy(self):
        return Node(self.index, self.creature)
