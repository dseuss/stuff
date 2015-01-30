#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import matplotlib.pyplot as pl

import numpy as np
import scipy.fftpack as fft

from tools.plot import imsshow, rgb2gray


THRESHMAT = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 50, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 255],
                      [59, 64, 78, 87, 103, 255, 255, 255],
                      [72, 91, 95, 98, 112, 255, 255, 255]], dtype=np.uint8)

THRESHMAT = THRESHMAT // 8


def extract_blocks_2D(ary, bs):
    # TODO Check if this is the right continuation mode
    padded = np.pad(ary, ((0, -ary.shape[0] % bs[0]), (0, -ary.shape[1] % bs[1])),
                    mode='edge')
    splits = [xrange(bs[i], padded.shape[i], bs[i]) for i in (0, 1)]
    return np.array([np.split(subimg, splits[1], axis=1)
                     for subimg in np.split(padded, splits[0])])


def blocks2img(blocks):
    return np.vstack([np.hstack(row) for row in blocks])


def quantize(ary, thresh):
    res = thresh * np.floor(ary // thresh)
    return res


if __name__ == '__main__':
    img = rgb2gray(pl.imread('Lenna.png'))
    img = (img * 255).astype(np.uint8)
    pl.gray()
    blocksize = (8, 8)
    blocks = extract_blocks_2D(img, bs=blocksize)
    blockshape = blocks.shape[:2]
    blocks = blocks.reshape((-1, ) + blocksize)

    compressed = np.array([quantize(fft.dct(b.astype(float), norm='ortho'), THRESHMAT) for b in blocks])
    img_c = blocks2img(np.reshape([fft.idct(b.astype(float), norm='ortho') for b in compressed],
                                blockshape + blocksize))

    pl.subplot(121)
    pl.hist(np.ravel(blocks), bins=60)
    pl.subplot(122)
    pl.hist(np.ravel(compressed), bins=60)
    pl.show()

    imsshow((img, img_c))
