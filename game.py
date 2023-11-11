import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from creature_action import Creature, Action
from state import State
from node import Node


class Game:
    def __init__(self, N, K, W, L, CR, CT):
        for arg, arg_name in (
            (N, "N"),
            (K, "K"),
            (W, "W"),
            (L, "L"),
            (CR, "CR"),
            (CT, "CT"),
        ):
            self._typecheck(arg, arg_name, int)
            self._check_non_negative(arg, arg_name)

        # Parameters
        self.N = N
        self.K = K
        self.W = W
        self.L = L
        self.CR = CR
        self.CT = CT

        # State space as a { , R, T}^N product iterated into a list of States

        # All combinations of Creatures as a list [*[C, C, C]]
        creature_combinations = [
            list(s)
            for s in itertools.product(
                *[
                    [Creature.Empty, Creature.Rabbit, Creature.Tiger]
                    for i in range(N)
                ]
            )
        ]

        # Then as a list of Node [*[Node(i, C)]]
        node_combinations = [
            [Node(i, c) for i, c in enumerate(creatures)]
            for creatures in creature_combinations
        ]

        # Then as States
        self.state_space = [State(nodes) for nodes in node_combinations]
        self.card_S = len(self.state_space)  # == 3**N

        # Map from actions to their functions
        # these take a state and return a list [*[next_state, proba, reward]]
        self.actions_map = {
            Action.BR: self.birth_rabbit,
            Action.BT: self.birth_tiger,
            Action.AR: self.activate_rabbits,
            Action.AT: self.activate_tigers,
            Action.AD: self.activate_dinosaurs,
        }
        self.card_A = 5

        # Precompute all rewards and transition probabilities
        # These are (|S|, |A|, |S|) tensors
        #   index by (s, a, s')
        #   slice as [:, a, :] to get the R_a and P_a matrices
        self.rewards, self.probas = self.rewards_and_probas()

        # These are for playing interactively
        self.state = State([Node(i, Creature.Empty) for i in range(N)])
        self.score = 0
        self.state_history = [self.state]
        self.score_history = [self.score]
        self.reward_history = []
        self.action_history = []

        # Create a plot
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = self.fig.subplots()
        self.ax.set_aspect("equal")

        # Coordinates of nodes on the board
        angles = 2 * np.pi * np.array([n.index for n in self.state]) / self.N
        self.x = np.sin(angles)
        self.y = np.cos(angles)

        # for i, s in enumerate(self.state_space[:20]):
        #     for a in Action:
        #         probas = self.probas[i, a.value]
        #         for j, p in enumerate(probas):
        #             if p > 0:
        #                 print(
        #                     f"State: {s}, action: {a.name} =[P = {p}]=> {self.state_space[j]}"
        #                 )
        #         print("_____")
        #     print("============")

    @property
    def time(self):
        """Elpased time = number of states up to this point."""
        return len(self.state_history)

    # Tensors initialization function

    def rewards_and_probas(self):
        """Precompute rewards and transition probabilities for all (s, a, s')."""

        # These are (s, a, s') indexed tensors
        rewards = np.zeros((self.card_S, self.card_A, self.card_S))
        probas = np.zeros((self.card_S, self.card_A, self.card_S))

        # Fill for initial states
        for i, state in enumerate(self.state_space):
            # Fill for actions
            for j, action in enumerate(Action):
                action_function = self.actions_map[action]
                # Get the possible (non-null proba) [next_state, proba, reward]
                possible_next = action_function(state.copy())
                # Get the index of the next states and fill the tensors
                for next_state, proba, reward in possible_next:
                    k = self.state_space.index(next_state)
                    rewards[i, j, k] = reward
                    probas[i, j, k] = proba

        return rewards, probas

    # Helper functions for typechecking and valuechecking

    @staticmethod
    def _typecheck(arg, arg_name, T):
        if not isinstance(arg_name, str):
            raise TypeError(
                f"'arg_name' argument should be of type 'str' but "
                f"found '{type(arg_name).__name__}'."
            )
        if not isinstance(arg, T):
            raise TypeError(
                f"'{arg_name}' argument should be of type "
                f"'{T.__name__}' but found '{type(arg).__name__}'."
            )

    @staticmethod
    def _check_non_negative(arg, arg_name):
        if not isinstance(arg_name, str):
            raise TypeError(
                f"'arg_name' argument should be of type 'str' but "
                f"found '{type(arg_name).__name__}'."
            )
        if arg < 0:
            raise ValueError(
                f"'{arg_name}' argument should be non-negative "
                f"but found '{arg}'."
            )

    def _typecheck_state(self, state):
        self._typecheck(state, "state", State)

    def _typecheck_creature(self, creature):
        self._typecheck(creature, "creature", Creature)

    def _typecheck_action(self, action):
        self._typecheck(action, "action", Action)

    # Action functions, these all return a list [*[next_state, proba, reward]]

    def activate_rabbits(self, state):
        """Apply action AR on state."""
        self._typecheck_state(state)

        reward = 0
        next_state = state.copy()
        for node in state.get_rabbits():
            # Propagate around rabbits
            for step in (+1, -1):
                i = (node.index + step + self.N) % self.N
                if state[i].creature == Creature.Empty:
                    next_state[i].creature = Creature.Rabbit

        return [[next_state, 1, reward]]

    def activate_tigers(self, state):
        """Apply action AT on state."""
        self._typecheck_state(state)

        reward = 0
        next_state = state.copy()
        for node in state.get_tigers():
            # Propagate for 2 clockwise steps
            for step in (+1, +2):
                i = (node.index + step) % self.N
                if state[i].creature == Creature.Rabbit:
                    next_state[i].creature = Creature.Tiger

            # Empty original tiger node
            next_state[node.index].creature = Creature.Empty

        return [[next_state, 1, reward]]

    def activate_dinosaurs(self, state):
        """Apply action AD on state."""
        self._typecheck_state(state)

        result = []

        # Get all combinations of 3 nodes indices amongst N
        for indices in itertools.combinations(range(self.N), self.K):
            next_state = state.copy()

            # Count the eaten rabbits and tigers for rewards
            eaten_rabbits = 0
            eaten_tigers = 0

            # Go through the nodes at these indices
            for i in indices:
                next_state[i].creature = Creature.Empty

                if state[i].creature == Creature.Rabbit:
                    eaten_rabbits += 1
                elif state[i].creature == Creature.Tiger:
                    eaten_tigers += 1

            # Nothing eaten, loss
            if eaten_rabbits == 0 and eaten_tigers == 0:
                reward = -self.L
            # Something eaten, win // tigers
            else:
                reward = eaten_tigers * self.W

            # NOTE: Temporary proba used here, adjusted after
            result.append([next_state, 1, reward])

        ###
        # Adjust the probas
        adjusted_result = []

        # First get unique next states
        unique_results = []
        for res in result:
            if res not in unique_results:
                unique_results.append(res)

        # Count them among results and compute next state proba
        for res in unique_results:
            count = result.count(res)
            s, _, r = res
            adjusted_result.append([s, count / len(result), r])

        return adjusted_result

    def birth(self, state, creature):
        """Apply one of actions BR or BT on state."""
        self._typecheck_state(state)
        self._typecheck_creature(creature)

        if creature == Creature.Empty:
            raise ValueError("'creature' argument can't take value 'Empty'.")

        reward = -self.CR if creature == Creature.Rabbit else -self.CT

        empty_nodes = state.get_empty()
        N_empty = len(empty_nodes)

        # No empty nodes, action does nothing
        if N_empty == 0:
            return [[state.copy(), 1, reward]]

        # All empty nodes have an associated next state
        result = []
        for node in empty_nodes:
            next_state = state.copy()
            next_state[node.index].creature = creature
            result.append([next_state, 1 / N_empty, reward])

        return result

    def birth_rabbit(self, state):
        """Apply one of action BR on state."""
        self._typecheck_state(state)
        return self.birth(state, Creature.Rabbit)

    def birth_tiger(self, state):
        """Apply one of action BR on state."""
        self._typecheck_state(state)
        return self.birth(state, Creature.Tiger)

    # This one picks the next state and associated reward according to probas

    def next_state_and_reward(self, state, action):
        """Apply action on state using probas and returns (next_state, reward)."""
        self._typecheck_state(state)
        self._typecheck_action(action)

        # Get possible next states and probas
        i = self.state_space.index(state)
        j = action.value
        probas = self.probas[i, j, :]

        # Choose next state using probas (uniform)
        k = np.random.choice(list(range(self.card_S)), p=probas)

        next_state = self.state_space[k]
        reward = self.rewards[i, j, k]

        return next_state, reward

    # Functions to play the game step by step

    def replay_solution(self, n, opt_policy, g_star):
        for i in range(n):
            state = self.state
            opt_action = opt_policy[self.state_space.index(self.state)]
            _, reward = self.play_action(opt_action)

            if i % (n // 10) == 0 or i < 20:
                print(
                    f"At t = {i}, in state {state}: \n"
                    f"- playing optimal action {opt_action.name} for reward {reward}\n"
                    f"- current average gain is {np.mean(self.reward_history)}\n"
                    f"               while g* = {g_star}"
                )

    def plot_state(self, colors=None):
        if colors is None:
            colors = ["blue" for _ in range(self.N)]

        self.ax.clear()

        self.ax.scatter(self.x, self.y, s=0)

        for i in range(self.N):
            self.ax.text(
                self.x[i],
                self.y[i],
                self.state[i].creature_symbol,
                bbox=dict(facecolor=colors[i], alpha=0.5),
            )

    def play_interactive(self):
        self.plot_state()

        self.ax.text(-0.2, 0, "Use the buttons to play!")

        axes = [plt.axes([0.3 + i * 0.1, 0.9, 0.1, 0.075]) for i in range(5)]
        AR = Button(axes[0], "AR")
        AR.on_clicked(self.play_AR)
        AT = Button(axes[1], "AT")
        AT.on_clicked(self.play_AT)
        AD = Button(axes[2], "AD")
        AD.on_clicked(self.play_AD)
        BR = Button(axes[3], "BR")
        BR.on_clicked(self.play_BR)
        BT = Button(axes[4], "BT")
        BT.on_clicked(self.play_BT)

        plt.show()

    def play_action(self, action):
        """Play action on current state of the game.
        Updates the board, score and histories."""
        self._typecheck_action(action)

        next_state, reward = self.next_state_and_reward(self.state, action)

        self.score += reward
        self.state = next_state

        self.action_history.append(action)
        self.state_history.append(self.state)
        self.score_history.append(self.score)
        self.reward_history.append(reward)

        return next_state, reward

    def play_and_plot_action(self, action):
        self._typecheck_action(action)

        prev_state = self.state

        next_state, reward = self.play_action(action)

        modified_nodes = [
            i for i in range(self.N) if next_state[i] != prev_state[i]
        ]

        colors = []
        for i in range(self.N):
            if i in modified_nodes:
                colors.append("red")
            else:
                colors.append("blue")

        self.plot_state(colors)

        self.ax.text(
            -0.3,
            0.1,
            f"Score: {self.score}\n"
            f"t = {self.time}\n"
            f"You just played {action.name}\nand won {reward}",
        )

        plt.draw()

    def play_AR(self, b):
        self.play_and_plot_action(Action.AR)

    def play_AT(self, b):
        self.play_and_plot_action(Action.AT)

    def play_AD(self, b):
        self.play_and_plot_action(Action.AD)

    def play_BR(self, b):
        self.play_and_plot_action(Action.BR)

    def play_BT(self, b):
        self.play_and_plot_action(Action.BT)

    # Value iteration

    def value_iteration(self, epsilon):
        """Returns (g*, π*)."""
        self.card_S = 3**self.N

        # Vectors of size |S|
        v = np.zeros(self.card_S)
        v_last = -np.array([i for i in range(self.card_S)]) * 1e6
        opt_policy = np.array([None for _ in range(self.card_S)])

        # Iteration counter
        n = 0

        ###########################################################

        periodic_fix = False

        # Stopping condition on span
        while np.max(v - v_last) - np.min(v - v_last) > epsilon:
            v_last = v.copy()

            # These are results for all (s, a)
            vectors = np.empty((self.card_S, self.card_A))

            for a in Action:
                # (s, s') matrices
                P = self.probas[:, a.value, :]
                R = self.rewards[:, a.value, :]

                # Component-wise mult for P and R, then sum over s'
                vectors[:, a.value] = np.sum(P * R, axis=1)

                if periodic_fix:
                    vectors[:, a.value] += 0.5 * P @ v_last + 0.5 * v_last
                else:
                    vectors[:, a.value] += P @ v_last

            # Component-wise max over a
            v = np.max(vectors, axis=1)

            n += 1

########################
# Old non matrix version
########################
#        else:
#            # Stopping condition on span
#            while np.max(v - v_last) - np.min(v - v_last) > epsilon:
#                v_last = v.copy()
#
#                # Iterate over states
#                for i in range(self.card_S):
#                    v_new = -1e6
#
#                    # Find best action
#                    for a in Action:
#                        # Get transition proba and reward from s under action a
#                        p = self.probas[i, a.value, :]
#                        r = self.rewards[i, a.value, :]
#
#                        # Compute sum over s'
#                        v_temp = np.sum(
#                            [p * (r + v_last)]
#                        )  # + 0.5 * v + 0.5 * p * v])
#
#                        # If better, save it, best action from s is a
#                        if v_temp > v_new:
#                            v_new = v_temp
#
#                    # Update
#                    v[i] = v_new
#
#                # Count
#                n += 1

        ################################################

        print(f"Iterated {n} times.")

        for i in range(self.card_S):
            maxi = -1e6
            for a in Action:
                p = self.probas[i, a.value, :]
                r = self.rewards[i, a.value, :]

                if periodic_fix:
                    gain = np.sum([p * (r + 0.5 * v) + 0.5 * v])
                else:
                    gain = np.sum([p * (r + v)])

                if gain > maxi:
                    maxi = gain
                    opt_policy[i] = a

        return (v - v_last)[0], opt_policy

    # Printing and plotting functions

    def print_score(self):
        print("score = ", self.score)
