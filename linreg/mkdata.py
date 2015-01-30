#!/usr/bin/env python
# encoding: utf-8
"""Sample Maker
Creates the given number of samples from a random sample (f(x), x) with
x sampled from the uniform distribution on [0, 1]. Here, f is a linear function
with some additive gaussian noise xi (f(x) = mx + n + scale*xi).

Usage:
   mkdata.py [--samples=<ss>] [-m=<m>] [-n=<n>] [--scale=<sc>] [--sigma=<sig>] [--file=<filename>]
   mkdata.py -h | --help
   mkdata.py --version

Options:
   -h --help          Show this screen.
   --samples=<ss>     Number of sample points [default: 10].
   -m=<m>             Slope of linear function [default: 0.5].
   -n=<n>             Offset of linear function [default: 0.0].
   --scale=<sc>       Scale factor in front of gaussian noise [default: 0.05].
   --sigma=<sig>      Standard deviation of gaussian noise [default: 1.0].
   --file=<filename>  File to write to [default: data.csv]
"""

from __future__ import division, print_function

import sys
sys.path.append("/usr/lib/python2.7/dist-packages/")

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as pl
from docopt import docopt


if __name__ == '__main__':
   args = docopt(__doc__, version='mkdata 1.0')
   num_samples = int(args['--samples'])
   m, n = float(args['-m']), float(args['-n'])
   scale, sigma = float(args['--scale']), float(args['--sigma'])

   x = rnd.uniform(size=num_samples)
   y = m*x + n + scale * rnd.normal(loc=0., scale=sigma, size=num_samples)

   pl.scatter(x, y)
   pl.show()

   # TODO Add parameters in comments to file
   np.savetxt(str(args['--file']), np.transpose([y, x]), delimiter=',')
