# -*- coding: utf-8 -*-
"""Auxiliary functions that support for system."""

from datetime import datetime
from itertools import chain


def get_fullname(o):
    """get the full name of the class."""
    return '%s.%s' % (o.__module__, o.__class__.__name__)


def str2time(string, pattern):
    """convert the string to the datetime."""
    return datetime.strptime(string, pattern)


def make_square(seq, size):
    return zip(*[iter(seq)] * size)


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))
