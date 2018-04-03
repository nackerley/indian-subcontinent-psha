#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:18:33 2018

@author: nick
"""
import numpy as np


def width_from_dip(dip, depth):
    return depth/np.sin(np.deg2rad(dip))


def area_scaling(magnitude, relation='WC1994'):
    if msr == 'StrasserInterface':
        return 10**(-3.99 + 0.98*magnitude)
    else:
        return 10**(-3.476 + 0.952*magnitude)


magnitude = 9.0

for msr, depth, dip in zip(
        ['WC1994'] + ['StrasserInterface']*4,
        [25, 25, 70, 20, 25],
        [78, 78, 78, 10, 12.6]):

    area = area_scaling(magnitude, msr)

    width = width_from_dip(dip, depth)
    length = area/width

    print('M%.1f, %s, %g° dip, %g km depth:' %
          (magnitude, msr, dip, depth))
    print('\t%.0f,000 km^2 area, %.0f x %.0f km' %
          (area/1000, width, length))
