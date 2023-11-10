""""""

from enum import Enum


class Creature(Enum):
    """
    Different values the Nodes can take.
    """

    Empty = 0
    Rabbit = 1
    Tiger = 2


class Action(Enum):
    '''
    Different actions.
    '''

    AR = 0
    AT = 1
    AD = 2
    BR = 3
    BT = 4
