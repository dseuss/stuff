#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import numpy as np
from itertools import izip
from numpy.random import uniform, normal
from tools.helpers import Progress
from tools.nptools import find


BALLSIZE = 1.0


def dist(v1, v2):
    return np.sqrt(np.sum((v1 - v2) * (v1 - v2), axis=-1))


class BubbleBox(object):
    """Docstring for BubbleBox. """

    def __init__(self, boxsize, number=1, temperature=1.):
        """@todo: to be defined1.

        :param boxsize: @todo
        :param number: @todo

        """
        self._boxsize = boxsize

        self._ballsize = BALLSIZE / 2
        self._pos = np.zeros((number, 2), np.float32)
        for n, pos in Progress(enumerate(self._pos), total=number):
            pos[:] = self._get_nonintersecting_pos()

        self._vel = normal(scale=temperature, size=(number, 2))
        self._last = 0.

    def _get_nonintersecting_pos(self, max_iterations=10000):
        bs = self._ballsize
        for _ in range(max_iterations):
            pos = np.array((uniform(bs, self.width - bs),
                            uniform(bs, self.height - bs)),
                           np.float32)
            try:
                self.find_intersection(pos)
            except StopIteration:
                return pos
        else:
            raise ValueError("Couldnt find nonintersecting new position.")

    def find_intersection(self, pos, sel=slice(None)):
        res = find(self._pos[sel], lambda x: dist(x, pos) <= 2 * self._ballsize)
        return res.next()[0]

    def propagate(self, t):
        boundary = np.asarray(self._boxsize) - self._ballsize
        for i, (pos, vel) in enumerate(izip(self._pos, self._vel)):
            sel = (pos < self._ballsize) * (vel < 0) + (pos > boundary) * (vel > 0)
            vel[sel] *= -1

            try:
                j = self.find_intersection(pos, sel=slice(i))
                x_ij = self._pos[i] - self._pos[j]
                v_ij = self._vel[i] - self._vel[j]
                diff_v = np.dot(x_ij, v_ij)
                if diff_v < 0:
                    diff_v /= np.dot(x_ij, x_ij) * np.sqrt(np.dot(v_ij, v_ij))
                    self._vel[i] -= diff_v * x_ij
                    self._vel[j] += diff_v * x_ij
            except StopIteration:
                pass

        self._pos += (t - self._last) * self._vel
        self._last = t

    def positions(self, t):
        return self._pos + (t - self._last) * self._vel

    @property
    def ballsize(self):
        return self._ballsize

    def __len_(self):
        return 0

    @property
    def boxsize(self):
        return self._boxsize

    @property
    def width(self):
        return self._boxsize[0]

    @property
    def height(self):
        return self._boxsize[1]

