#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Indian Subcontinent PSHA
# Copyright (C) 2016-2018 Nick Ackerley
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
OpenQuake Implementation of source models for Nath & Thingbaijam (2012)

Read the source description input files from the online supplementary
material and write them to XML.

Note: For imports to work, ../utilities directory must be added to PYTHONPATH
"""

# %% imports

import os
# import sys
from time import time

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from shapely.wkt import dumps
from shapely.geometry import Point
from descartes import PolygonPatch

from obspy.imaging.beachball import aux_plane

from more_itertools import unique_everseen

from source_model_tools import (
    read_polygons, focal_mech, faulting_style, df2nrml, areal2csv,
    SEISMICITY_ALIASES, points2csv, points2nrml, MyPolygon)
from logic_tree_tools import read_tree_tsv, collapse_sources
from toolbox import wrap

from openquake.hazardlib import geo

pd.set_option('mode.chained_assignment', 'raise')


# %% constants
THINNED_ONLY = True

# define the input file names from the original paper
MODEL_PATH = '../Data/nath2012probabilistic'
POLYGON_FORMAT = 'polygonlay%d.txt'
SEISMICITY_FORMAT = 'seismicitylay%d.txt'

SOURCE_TREE_TSV = '../Logic Trees/areal_model_logic_tree_v1.tsv'

# an input file supplies some auxiliary data
AUX_FILE = 'auxiliary_data_v1.csv'
PREFIX = 'nt2012'
VERSION = os.path.splitext(AUX_FILE)[0].split('_')[-1]

# define prefixes for the output file names and models
AREAL_SOURCE_MODEL_BASE = PREFIX + ' areal source model %s' % VERSION

MIN_MAGS = [4.5, 5.5]
LAYERS_DF = pd.read_csv('layers.csv', index_col='layerid')

USE_RECOMPUTED = False

if USE_RECOMPUTED:
    SMOOTHED_MODEL_PATH = '../Smoothed/Recomputed'
else:
    SMOOTHED_MODEL_PATH = MODEL_PATH

SMOOTHED_DATA_TEMPLATE = os.path.join(SMOOTHED_MODEL_PATH,
                                      'lay%dsmooth%.1f.txt')

SMOOTHED_MODEL_BASE = '_'.join((os.path.split(SMOOTHED_MODEL_PATH)[-1],
                                'smoothed source model %s' % VERSION))
SMOOTHED_MODEL_BASE = SMOOTHED_MODEL_BASE.replace(
    os.path.split(MODEL_PATH)[-1], PREFIX)

# %% load electronic supplement for areal zones

print('Reading areal polygons and seismicity statistics for each layer')
areal_dfs = []
for layer_id in LAYERS_DF.index:

    # read seismicity and polygons and join them
    seismicity_file = os.path.join(MODEL_PATH, SEISMICITY_FORMAT % layer_id)
    print('Reading: ' + os.path.abspath(seismicity_file))
    seismicity_df = pd.read_csv(seismicity_file)
    seismicity_df.set_index('zoneid', inplace=True, verify_integrity=True)
    seismicity_df.rename(columns=SEISMICITY_ALIASES, inplace=True)

    # preserve errors in electonic supplement in version v0
    if VERSION == 'v0' and layer_id == 4:
        seismicity_df.loc[169], seismicity_df.loc[170] = \
            seismicity_df.loc[170].copy(), seismicity_df.loc[169].copy()
        print('Swapped seismicity for zones 169 and 170.')

    polygon_file = os.path.join(MODEL_PATH, POLYGON_FORMAT % layer_id)
    print('Reading: ' + os.path.abspath(polygon_file))
    polygon_df = read_polygons(polygon_file)
    polygon_df.set_index('zoneid', inplace=True, verify_integrity=True)

    df = seismicity_df.join(polygon_df, how='outer')

    # add layer info
    df.insert(0, 'layerid', layer_id)
    areal_dfs.append(df)

# put it all together
columns = list(unique_everseen([column for column in df.columns
                                for df in areal_dfs]))
areal_df = pd.concat(areal_dfs)[columns].sort_index()

# %% auxiliary information

print('\nReading: ' + os.path.abspath(AUX_FILE))
aux_df = pd.read_csv(AUX_FILE, index_col='zoneid').sort_index()
assert (areal_df.index == aux_df.index).all()
if 'layerid' in aux_df:
    aux_df.drop(columns='layerid', inplace=True)
areal_df = areal_df.join(aux_df)

# %% augment areal zone description tables

areal_df = areal_df.join(LAYERS_DF, on='layerid')
areal_df['rake'] = wrap(areal_df['rake'])
areal_df['mechanism'] = focal_mech(areal_df['dip'], areal_df['rake'])
areal_df['new style'] = faulting_style(areal_df['strike'], areal_df['dip'],
                                       areal_df['rake'])
areal_df['strike2'], areal_df['dip2'], areal_df['rake2'] = zip(*[
    aux_plane(strike, dip, rake)
    for strike, dip, rake in zip(
        areal_df['strike'], areal_df['dip'], areal_df['rake'])])
areal_df['mechanism2'] = focal_mech(areal_df['dip2'], areal_df['rake2'])
areal_df.loc[areal_df['dip'] == -1, 'dip'] = np.nan
areal_df.loc[areal_df['mechanism'] == 'undefined', 'rake'] = np.nan
areal_df.loc[areal_df['strike'] == -1, 'strike'] = np.nan

areal_df['mmin'] = MIN_MAGS[0]

areal_df['strike2'] = areal_df['strike2'].apply(lambda x: round(x, 1))
areal_df['dip2'] = areal_df['dip2'].apply(lambda x: round(x, 1))
areal_df['rake2'] = areal_df['rake2'].apply(lambda x: round(x, 1))

swap = areal_df['focal plane'] == 'secondary'
print('Treating %d focal planes as secondary' % sum(swap))
for column in ['strike', 'dip', 'rake', 'mechanism']:
    areal_df.loc[swap, [column, column + '2']] = \
        areal_df.loc[swap, [column + '2', column]].values

# %% write areal CSV

areal2csv(areal_df, AREAL_SOURCE_MODEL_BASE)

# %% write areal NRML

mark = time()
areal_source_model = df2nrml(areal_df, AREAL_SOURCE_MODEL_BASE)
print('Finished writing areal model to NRML: %s\n' %
      pd.to_timedelta(time() - mark, unit='s'))

# %% read logic tree description table

print('Logic tree before collapse:')
source_tree_symbolic_df = read_tree_tsv(SOURCE_TREE_TSV)
print(source_tree_symbolic_df)

# %% compute collapsed rates

areal_collapsed_df, reduced_df, all_weights, labels = \
    collapse_sources(areal_df, source_tree_symbolic_df)

print('Logic tree after collapse:')
print(reduced_df)

# %% write areal sources to NRML

mark = time()
df2nrml(areal_collapsed_df, AREAL_SOURCE_MODEL_BASE + ' collapsed')
print('Finished writing collapsed areal model to NRML: %s\n' %
      pd.to_timedelta(time() - mark, unit='s'))

# %% areal zone data

# grab mmax and bvalue from zone above if mmax zero for this zone
check_keys = ['mmax', 'b']
for i, area_series in areal_df[
        (areal_df[check_keys] == 0).any(axis=1)].iterrows():
    alternate_zone = int(area_series.name/10)
    for key in check_keys:
        if area_series[key] == 0:
            print('For zone %d taking %s from zone %d' %
                  (area_series.name, key, alternate_zone))
            areal_df.at[i, key] = areal_df.at[alternate_zone, key]

# in some cases we are only interested in the active zones
active_areal_df = areal_df[areal_df['a'] != 0].reset_index()

active_areal_df.head()

# %% completeness tables

print('Reading completeness tables.')
completeness_df = pd.read_csv(
    '../Data/thingbaijam2011seismogenic/Table1.csv',
    header=[0, 1], index_col=[0, 1])
completeness_df.columns = [' '.join(col).strip()
                           for col in completeness_df.columns.values]

# %% electronic supplement for smoothed-gridded model

print('Reading smoothed seismicity data ...')
mark = time()
smoothed_df_list = []
for i, min_mag in enumerate(MIN_MAGS):
    layer_smoothed_df_list = []
    for layer_id, layer in LAYERS_DF.join(completeness_df,
                                          on=['zmin', 'zmax']).iterrows():

        layer_smoothed_df = pd.read_csv(SMOOTHED_DATA_TEMPLATE %
                                        (layer_id, min_mag))
        nu_mag = 'nu%s' % str(min_mag).replace('.', '_')

        rename_cols = {nu_mag: 'nu', 'lat': 'latitude', 'lon': 'longitude'}
        layer_smoothed_df.rename(columns=rename_cols, inplace=True)

        layer_smoothed_df['layerid'] = layer_id
        layer_smoothed_df['mmin model'] = min_mag
        layer_smoothed_df['mmin'] = min_mag
        layer_smoothed_df['duration'] = (
            layer[str(min_mag) + ' end'] -
            layer[str(min_mag) + ' start'] + 1)
        if USE_RECOMPUTED:
            layer_smoothed_df['lambda'] = layer_smoothed_df['nu']
            layer_smoothed_df['nu'] = (layer_smoothed_df['lambda'] *
                                       layer_smoothed_df['duration'])
        else:
            layer_smoothed_df['lambda'] = (layer_smoothed_df['nu'] /
                                           layer_smoothed_df['duration'])

        layer_smoothed_df_list.append(layer_smoothed_df)

    layer_smoothed_df = pd.concat(layer_smoothed_df_list)
    smoothed_df_list.append(layer_smoothed_df)

smoothed_df = pd.concat(smoothed_df_list)
smoothed_df.sort_values(['layerid', 'mmin model', 'longitude', 'latitude'])
smoothed_df['geometry'] = [Point(longitude, latitude)
                           for longitude, latitude
                           in zip(smoothed_df['longitude'],
                                  smoothed_df['latitude'])]
smoothed_df = gpd.GeoDataFrame(smoothed_df, crs='WGS84')

print('Read %d point sources from %d files: %s\n' %
      (len(smoothed_df), len(MIN_MAGS)*len(LAYERS_DF),
       pd.to_timedelta(time() - mark, unit='s')))

# %% associate points in zones

print('Associate point sources in areal zones with those zones ...')
# quick, requires no transformations
mark = time()

smoothed_df['distance'] = np.inf
smoothed_dfs = []
for layer_id in LAYERS_DF.index:
    smoothed_layer_df = smoothed_df[smoothed_df['layerid'] == layer_id]
    areal_layer_df = gpd.GeoDataFrame(
        areal_df[areal_df['layerid'] == layer_id],
        crs='WGS84')
    smoothed_layer_df = gpd.sjoin(
        smoothed_layer_df, areal_layer_df[['a', 'geometry']],
        how='left', op='within')
    smoothed_dfs.append(smoothed_layer_df)
smoothed_df = pd.concat(smoothed_dfs)
smoothed_df.rename({'index_right': 'zoneid'}, axis=1, inplace=True)

smoothed_df['in zoneid'] = smoothed_df['zoneid'].copy()

assigned = (~np.isnan(smoothed_df['in zoneid'])) & (smoothed_df['a'] != 0)
smoothed_df.loc[assigned, 'distance'] = 0
print('Spatial join accounted for %.2f%% of sources: %s\n' %
      (100*len(smoothed_df[assigned])/len(smoothed_df),
       pd.to_timedelta(time() - mark, unit='s')))

# %% no point should be associated with multiple zones
id_columns = ['latitude', 'longitude', 'layerid', 'mmin']
duplicated_df = smoothed_df[smoothed_df.duplicated(
    subset=id_columns, keep=False)].sort_values(id_columns + ['zoneid'])
if duplicated_df.empty:
    print('SUCCESS: No grid point fell in multiple areal zones')
else:
    duplicated_df.to_csv('smoothed_duplicated.csv')

    point_a = duplicated_df.iloc[0]
    point_b = duplicated_df.iloc[1]
    zone_a = areal_df.at[int(point_a.zoneid), 'geometry']
    zone_b = areal_df.at[int(point_b.zoneid), 'geometry']

    fig, ax = plt.subplots()
    ax.add_patch(PolygonPatch(zone_a, alpha=0.5))
    ax.add_patch(PolygonPatch(zone_b, alpha=0.5))
    ax.scatter(duplicated_df['longitude'], duplicated_df['latitude'])
    ax.set_xlim((point_a.longitude - 5, point_a.longitude + 5))
    ax.set_ylim((point_a.latitude - 5, point_a.latitude + 5))
    ax.set_aspect(1)

    print(int(point_a.zoneid), point_a.layerid,
          dumps(zone_a, rounding_precision=2))
    print(int(point_b.zoneid), point_a.layerid,
          dumps(zone_b, rounding_precision=2))
    print(point_a.longitude, point_a.latitude)
    raise RuntimeError('Points assigned to multiple zones.')

# %% associate points nearest to zones

print('Find nearest areal zones for remaining points ...')
mark = time()
active_areal_df['polygon'] = [
    MyPolygon([geo.point.Point(lat, lon)
               for lat, lon in zip(*zone.geometry.exterior.coords.xy)])
    for _, zone in active_areal_df.iterrows()]

unassigned_df = smoothed_df.loc[~assigned].copy()
distances = np.full((len(unassigned_df),
                     len(active_areal_df)), np.inf)
for i, area_series in active_areal_df.iterrows():
    in_layer = (unassigned_df['layerid'] == area_series['layerid']).values
    mesh = geo.mesh.Mesh(
        unassigned_df.loc[in_layer, 'longitude'].values,
        unassigned_df.loc[in_layer, 'latitude'].values)
    distances[in_layer, i] = area_series['polygon'].distances(mesh)

unassigned_df.loc[:, 'zoneid'] = active_areal_df.loc[
    np.argmin(distances, axis=1), 'zoneid'].values
unassigned_df.loc[:, 'distance'] = np.amin(distances, axis=1)

print('Nearest zone required for %.0f%% of sources: %s\n' %
      (100*len(unassigned_df)/len(smoothed_df),
       pd.to_timedelta(time() - mark, unit='s')))

smoothed_df = pd.concat((smoothed_df[assigned], unassigned_df))

# %% copy parameters of nearest areal zone

print('For each point source, copy parameters of nearest areal zone')
columns_to_copy = ['zoneid', 'zmax', 'zmin', 'tectonic subregion',
                   'a', 'b', 'stdb', 'mmax', 'stdmmax',
                   'rake', 'dip', 'strike', 'aspect ratio', 'msr']
smoothed_df.drop(columns=['a'], inplace=True)
smoothed_df = smoothed_df.merge(active_areal_df[columns_to_copy],
                                on='zoneid')
smoothed_df['a'] = (np.log10(smoothed_df['lambda']) +
                    smoothed_df['b']*smoothed_df['mmin model'])

# %% check for unassigned parameters

display_drop = ['zmax', 'zmin', 'aspect ratio', 'msr',
                'rake', 'dip', 'strike', 'stdb', 'stdmmax']
no_zoneid_df = smoothed_df[smoothed_df['zoneid'].isnull()]
no_mmax_df = smoothed_df[smoothed_df['mmax'] == 0]
no_b_df = smoothed_df[smoothed_df['b'] == 0]
if not no_zoneid_df.empty:
    print(no_zoneid_df.drop(display_drop, axis=1).head())
    print("Leftover points with no assigned zone id")
if not no_mmax_df.empty:
    print(no_mmax_df.drop(display_drop, axis=1).head())
    print("Leftover points with no assigned mmax")
if not no_b_df.empty:
    print(no_b_df.drop(display_drop, axis=1).head())
    print("Leftover points with no assigned b")

if (no_mmax_df.empty and no_b_df.empty and no_zoneid_df.empty):
    print("SUCCESS: No points with unassigned MFD or zone")
else:
    raise RuntimeError('Unassigned parameters remain.')

# %% Thinning of models allows quick testing and git archiving of a sample

res_deg = 1
thinned_df = smoothed_df.loc[
    np.isclose(np.remainder(smoothed_df['latitude'], res_deg), 0) &
    np.isclose(np.remainder(smoothed_df['longitude'], res_deg), 0)].copy()
print('Thinning to %g° spacing reduces number of points from %d to %d.\n'
      % (res_deg, len(smoothed_df), len(thinned_df)))

# %% write thinned models

mark = time()
points2csv(thinned_df, SMOOTHED_MODEL_BASE + ' thinned')
points2nrml(thinned_df, SMOOTHED_MODEL_BASE + ' thinned', by='mmin model')
print('Wrote %d thinned smoothed-gridded sources to CSV & NRML: %s\n' %
      (len(thinned_df), pd.to_timedelta(time() - mark, unit='s')))

# %% write full smoothed-gridded models (~10 minutes)

if not THINNED_ONLY:

    mark = time()
    points2csv(smoothed_df, SMOOTHED_MODEL_BASE)
    points2nrml(smoothed_df, SMOOTHED_MODEL_BASE, by='mmin model')
    print('Wrote %d full smoothed-gridded sources to CSV & NRML: %s\n' %
          (len(smoothed_df), pd.to_timedelta(time() - mark, unit='s')))

    # write collapsed smoothed-gridded sources to NRML (~10 minutes)

    mark = time()
    smoothed_collapsed_df, reduced_df, all_weights, labels = \
        collapse_sources(smoothed_df, source_tree_symbolic_df)

    points2nrml(smoothed_collapsed_df, SMOOTHED_MODEL_BASE + ' collapsed')
    print('Wrote %d collapsed smoothed-gridded sources to CSV & NRML: %s\n' %
          (len(smoothed_collapsed_df),
           pd.to_timedelta(time() - mark, unit='s')))
