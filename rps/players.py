#!/usr/bin/env python
# encoding: utf-8
"""Rock-paper-scissors player classes"""

from __future__ import division, print_function

import sys
from abc import ABCMeta, abstractmethod
from collections import Counter

import ghmm as gh
import numpy as np

from hmm import HMM
from tools.helpers import getch, Progress

ROCK_PAPER_SCISSORS = np.array([[0, -1, 1],
                                [1, 0, -1],
                                [-1, 1, 0]])
ROCK_PAPER_SCISSORS_SYMDICT = ["Rock", "Paper", "Scissors"]
ROCK_PAPER_SCISSORS_KEYDICT = {'r': 0, 'p': 1, 's': 2}

normalize_stoch_map = lambda x: x / np.sum(x, axis=-1, keepdims=True)


def most_common_elems(lst):
    """Returns the most common elements from a list.
    >>> most_common_elems([2, 2, 1])
    [2]
    >>> sorted(most_common_elems([2, 1]))
    [1, 2]
    >>> sorted(most_common_elems([3, 2, 1, 3, 2, 4, 5, 7, 3, 1, 1]))
    [1, 3]
    """
    c = Counter()
    for elem in lst:
        c[elem] += 1
    mc = c.most_common()
    return [elems for elems, count in mc if count == mc[0][1]]


class RPSPlayer(object):
    __metaclass__ = ABCMeta
    """Metaclass for a rock-paper-scissors player."""

    def __init__(self, rules=ROCK_PAPER_SCISSORS):
        """
        :param rules: Anti-hermitian matrix defining the rules of the game. The
                      following values for rules[i][j] are possible:
                          1: i-th symbol beats the j-th symbol
                          0: a tie between the i-th and the j-th symbol
                          -1: i-th symbol looses against j-th symbol
        """
        # All playes this player has done; elements are tuples where the first
        # element is the player's and the second element the opponents symbol
        self._memory = []
        self._rules = rules

    @abstractmethod
    def play(self):
        """
        Returns a symbol in [0,...,len(rules) - 1] indicating the players move
        """
        pass


class RandomPlayer(RPSPlayer):
    """The name says it all -- simply returns a random valid choice"""

    def play(self):
        return np.random.randint(len(self._rules))


class PersistentPlayer(RPSPlayer):
    """A player who always returns the same symbol defined on initialization"""

    def __init__(self, symbol, **kwargs):
        """@todo: to be defined1.

        :param symbol: The symbol to output all the time (should be valid!)

        """
        RPSPlayer.__init__(self, **kwargs)
        self._symbol = symbol

    def play(self):
        return self._symbol


class MemoryPlayer(RPSPlayer):
    """Assumes that the opponent will choose the most common (weighted)
    symbol of his last plays and chooses a symbol which beats it. If none
    such element exists, he just plays a random symbol."""

    def __init__(self, weights=(1, ), **kwargs):
        """
        :param weights: List of integers >= 0, weights put on the last
                        `len(weights)` plays.

        """
        RPSPlayer.__init__(self, **kwargs)
        self._weights = weights

    def play(self):
        recent = [[val[1]] * n
                  for n, val in zip(self._weights, reversed(self._memory))]

        most_common = most_common_elems(sum(recent, []))
        for choice in most_common:
            sel = np.where(self._rules[choice] == -1)[0]
            if len(sel) > 0:
                return sel[0]

        return np.random.randint(len(self._rules))


class HMMPlayer(RPSPlayer, HMM):
    """Docstring for HMMPlayer. """

    def __init__(self, n_hid=10, **kwargs):
        """@todo: to be defined1.

        :param n_hid: @todo

        """
        RPSPlayer.__init__(self, **kwargs)
        HMM.__init__(self, n_obs=self._rules.size, n_hid=n_hid)

    def play(self, **kwargs):
        """@todo: Docstring for play.
        :returns: @todo

        """
        nr_moves = len(self._rules)
        # first we convert the memory to the observable-format
        obs = np.asarray([i + nr_moves * j for i, j in self._memory])
        if len(obs) > 1:
            self.initialize()
            self.fit(obs, max_iter=10000)
            alpha = self._get_forward(self._mkh, self._mko, self._init, obs)[0][-1]
        else:
            alpha = self._init
        next_move_dist = np.dot(self._mko, alpha).reshape(self._rules.shape)
        next_move = np.argmax(np.sum(next_move_dist, axis=1))

        return np.where(self._rules[next_move] == -1)[0][0]


class GHMMPlayer(RPSPlayer):
    """Docstring for GHMMPlayer. """

    def __init__(self, n_hid=10, **kwargs):
        """@todo: to be defined1.

        :param **kwargs: @todo

        """
        RPSPlayer.__init__(self, **kwargs)
        self._n_hid = n_hid
        self._n_sym = len(self._rules)
        sym_pairs = [(i, j) for i in range(self._n_sym) for j in range(self._n_sym)]
        self._alphab = gh.Alphabet(sym_pairs)
        self._conversion_array = np.asarray([[self._alphab.internal((i, j))
                                              for j in range(self._n_sym)]
                                             for i in range(self._n_sym)])
        self._prediction_mode = False

    def _predict_next(self):
        """@todo: Docstring for _predict_next.
        :returns: @todo

        """
        a_init = normalize_stoch_map(np.random.rand(self._n_hid, self._n_hid))
        b_init = normalize_stoch_map(np.random.rand(self._n_hid, self._n_sym**2))
        pi_init = normalize_stoch_map(np.random.rand(self._n_hid))
        hmm = gh.HMMFromMatrices(self._alphab,
                                 gh.DiscreteDistribution(self._alphab),
                                 a_init, b_init, pi_init)
        obs = gh.EmissionSequence(self._alphab, self._memory)
        hmm.baumWelch(obs)

        alpha = hmm.forward(obs)[0][-1]
        trans = hmm.asMatrices()[0]
        alpha = np.dot(alpha, trans)
        next_moves_dist = np.zeros(self._n_sym**2)
        for i in range(self._n_hid):
            next_moves_dist += np.asarray(hmm.getEmission(i)) * alpha[i]
        next_moves_dist = next_moves_dist[self._conversion_array]
        next_move = np.argmax(np.sum(next_moves_dist, axis=0))

        return np.where(self._rules[next_move] == -1)[0][0]

    def play(self):
        if self._prediction_mode:
            return self._predict_next()
        else:
            return np.random.randint(self._n_sym)

class HumanPlayer(RPSPlayer):

    def __init__(self, keydict, **kwargs):
        RPSPlayer.__init__(self, **kwargs)
        self._keydict = keydict

    def play(self):
        while 1:
            c = getch()
            if self._keydict.has_key(c):
                return self._keydict[c]
        sys.exit(-1)




def play(player1, player2, rounds=1, verbose=False, symdict=None):
    """Play a number of `rounds` matches between the two players and return
    the score $S = sum_j a_j$, where

        a_j = 1 if player1 wone --or-- -1 if player2 wone --or-- 0 otherwise.

    """
    if player1 is player2:
        raise AttributeError("Players match...")
    if player1._rules is not player2._rules:
        raise AttributeError("Different rules sets...")
    if symdict is None:
        symdict = range(len(pl1._rules))

    score = [0, 0, 0]
    results = ["Player1 wins.", "Tie.", "Player 2 wins."]
    playiter = xrange(rounds) if verbose else Progress(xrange(rounds))
    for i in playiter:
        res1, res2 = player1.play(), player2.play()
        player1._memory.append((res1, res2))
        player2._memory.append((res2, res1))
        resind = 1 - player1._rules[res1][res2]
        score[resind] += 1
        if verbose:
            print("{} vs {}: {}".format(symdict[res1], symdict[res2],
                                        results[resind]))
            print(score)

    return score

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    pl1 = HumanPlayer(ROCK_PAPER_SCISSORS_KEYDICT)
    pl2 = GHMMPlayer(10)
    score = play(pl1, pl2, rounds=10, verbose=True,
                 symdict=ROCK_PAPER_SCISSORS_SYMDICT)
    pl1._prediction_mode = True
    pl2._prediction_mode = True
    score = play(pl1, pl2, rounds=1000, verbose=True,
                 symdict=ROCK_PAPER_SCISSORS_SYMDICT)
    print(score)

    import matplotlib.pyplot as pl
    # pl.scatter(range(200), [i for i, j in pl1._memory], color='r')
    # pl.scatter(range(200), [j for i, j in pl1._memory], color='b')
    pl.show()

