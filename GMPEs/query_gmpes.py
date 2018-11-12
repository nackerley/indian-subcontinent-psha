#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 00:35:18 2018

@author: nick
"""
import importlib
import pkgutil

import pandas as pd

from openquake import hazardlib

GMPE_SUMMARY = 'NT2012_GMPEs.csv'

# %% definitions


def get_gsim(class_name):
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=hazardlib.gsim.__path__,
            onerror=lambda x: None):
        module_name = '.'.join([hazardlib.gsim.__name__, modname])
        module = importlib.import_module(module_name)
        # print('Trying: ', module.__name__)
        try:
            Gsim = getattr(module, class_name)
            return Gsim
        except AttributeError:
            pass
    print('Not found: ', class_name)


def get_sa_periods(Gsim):
    for item in dir(Gsim):
        coeffs = getattr(Gsim, item)
        try:
            sa_coeffs = getattr(coeffs, 'sa_coeffs')
            return sorted([imt.period for imt in sa_coeffs.keys()])
        except AttributeError:
            pass
    print('No SA coeffs found: ', Gsim.__name__)


def has_imt(Gsim, string):
    for item in dir(Gsim):
        coeffs = getattr(Gsim, item)
        try:
            non_sa_coeffs = getattr(coeffs, 'non_sa_coeffs')
            return any(str(imt) == string for imt in non_sa_coeffs.keys())
        except AttributeError:
            pass
    print('No non-SA coeffs found: ', Gsim.__name__)


# %% read table
df = pd.read_csv(GMPE_SUMMARY, index_col='OpenQuake class')

# %% augment table

df['min period [s]'] = [min(get_sa_periods(get_gsim(item)))
                        for item in df.index]
df['max period [s]'] = [max(get_sa_periods(get_gsim(item)))
                        for item in df.index]
df['has PGA'] = [has_imt(get_gsim(item), 'PGA') for item in df.index]
df['has PGV'] = [has_imt(get_gsim(item), 'PGV') for item in df.index]

for attr in [
        'DEFINED_FOR_TECTONIC_REGION_TYPE',
        'DEFINED_FOR_INTENSITY_MEASURE_COMPONENT',
        ]:
    column = attr.replace('_', ' ').lower()
    df[column] = [str(getattr(get_gsim(item), attr)) for item in df.index]

for attr in [
        'DEFINED_FOR_INTENSITY_MEASURE_TYPES',
        ]:
    column = attr.replace('_', ' ').lower()
    df[column] = [
        ', '.join(sorted(item.__name__
                         for item in list(getattr(get_gsim(item), attr))))
        for item in df.index]

for attr in [
        'DEFINED_FOR_STANDARD_DEVIATION_TYPES',
        'REQUIRES_DISTANCES',
        'REQUIRES_RUPTURE_PARAMETERS',
        'REQUIRES_SITES_PARAMETERS',
        ]:
    column = attr.replace('_', ' ').lower()
    df[column] = [
        ', '.join(sorted(str(item)
                         for item in list(getattr(get_gsim(item), attr))))
        for item in df.index]

print('Minimum supported UHS period: ', df['min period [s]'].max())
print('Maximum supported UHS period: ', df['max period [s]'].min())

# %% write table
df.to_csv(GMPE_SUMMARY)
