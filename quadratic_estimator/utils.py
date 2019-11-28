#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. module:: utils
    :synopsis: collection of convenience functions
.. moduleauthor:: Fabian Koehlinger <fabian.koehlinger@ipmu.jp>
"""
from __future__ import print_function
import sys

# Python 2.x - 3.x compatibility: Always use more efficient range function
try:
    xrange
except NameError:
    xrange = range

def print_ascii_art(text):

    hashtags = get_ascii_art('#', n=len(text))

    print(hashtags)
    print(hashtags + '\n')
    print(text + '\n')
    print(hashtags)
    print(hashtags)

    return

def print_dot():

    sys.stdout.write('.')
    sys.stdout.flush()

    return

def get_ascii_art(symbol, n=80):

    ascii_art = ''

    if n < 80:
        n = 80

    for i in xrange(n):
        ascii_art += symbol

    return ascii_art