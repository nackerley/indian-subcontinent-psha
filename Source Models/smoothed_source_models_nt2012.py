# coding: utf-8
'''
Smoothed Source models for Nath & Thingbaijam (2012)

Read the source description input files from the online supplementary
material and write them to XML. Must be run after areal soruce models are
generated, because zone assignments for points are done on the basis of the
areal source model.

Note: For imports to work, ../utilities directory must be added to PYTHONPATH
'''
import os
# import sys
from time import time
from IPython.display import display

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from openquake.hazardlib import geo

from source_model_tools import (
    sort_and_reindex, points2csv, points2nrml, extract_param, MyPolygon)  # noqa
# from toolbox import limit_precision  # noqa
from toolbox import annotate, logspace, linspace

MIN_MAGS = [4.5, 5.5]
LAYERS_DF = pd.read_csv('layers.csv')
USE_RECOMPUTED = False


# @profile
# def main():

# define the input file names from the original paper
model_path = '../Data/nath2012probabilistic'
if USE_RECOMPUTED:
    smoothed_model_path = '../Smoothed/Recomputed'
else:
    smoothed_model_path = model_path

smoothed_data_template = os.path.join(smoothed_model_path,
                                      'lay%dsmooth%.1f.txt')
smoothed_data_files = [[smoothed_data_template % (layer_id, min_mag)
                        for layer_id in LAYERS_DF['id']]
                       for min_mag in MIN_MAGS]

# define prefixes for the output file names and models
smoothed_source_data_file = 'smoothed_source_model'

# %%

print('Reading (augmented) areal zone descriptions')
areal_df = pd.read_csv('areal_source_model.csv')
areal_df['geometry'] = areal_df['geometry'].apply(loads)
areal_df = gpd.GeoDataFrame(areal_df, crs='WGS84')

# grab mmax and bvalue from zone above if mmax zero for this zone
check_keys = ['mmax', 'b']
for i, area_series in areal_df[
        (areal_df[check_keys] == 0).any(axis=1)].iterrows():
    alternate_zone = int(area_series['zoneid']/10)
    index_alt = np.where(areal_df['zoneid'] == alternate_zone)[0][0]
    for key in check_keys:
        if area_series[key] == 0:
            print('For zone %d taking %s from zone %d' %
                  (area_series['zoneid'], key, alternate_zone))
            areal_df.loc[i, key] = areal_df.loc[index_alt, key]

# in some cases we are only interested in the active zones
active_areal_df = areal_df[areal_df['a'] != 0].reset_index()

active_areal_df.head()

# %%

print('Reading completeness tables.')
completeness_df = pd.read_csv(
    '../Data/thingbaijam2011seismogenic/Table1.csv',
    header=[0, 1], index_col=[0, 1])
completeness_df.columns = [' '.join(col).strip()
                           for col in completeness_df.columns.values]
completeness_df.reset_index(inplace=True)
display(completeness_df)

# %%

print('Reading smoothed seismicity data ...')
mark = time()
smoothed_df_list = []
for i, min_mag in enumerate(MIN_MAGS):
    layer_smoothed_df_list = []
    for j, layer in pd.merge(completeness_df, LAYERS_DF).iterrows():
        layer_smoothed_df = pd.read_csv(smoothed_data_files[i][j])
        nu_mag = 'nu%s' % str(min_mag).replace('.', '_')

        rename_cols = {nu_mag: 'nu', 'lat': 'latitude', 'lon': 'longitude'}
        layer_smoothed_df.rename(columns=rename_cols, inplace=True)

        layer_smoothed_df['layerid'] = layer['id']
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
smoothed_df = sort_and_reindex(smoothed_df)
mark = time()
smoothed_df['geometry'] = [Point(longitude, latitude)
                           for longitude, latitude
                           in zip(smoothed_df['longitude'],
                                  smoothed_df['latitude'])]
smoothed_df = gpd.GeoDataFrame(smoothed_df, crs='WGS84')

display(pd.concat((smoothed_df.head(), smoothed_df.tail())))
print('Read %d point sources from %d files [%.0f s]\n' %
      (len(smoothed_df), len(MIN_MAGS)*len(LAYERS_DF), time() - mark))

# %%

prop = 'lambda'
smoothed_hist_pdf = os.path.join(smoothed_model_path,
                                 'smoothed_%s_histograms.png' % prop)
print('Plotting histograms of %s:\n\t %s' %
      (prop, os.path.abspath(smoothed_hist_pdf)))
fig, axes = plt.subplots(nrows=len(MIN_MAGS), ncols=1,
                         figsize=(6, 3*len(MIN_MAGS)), sharex=True)
fig.subplots_adjust(hspace=0.1)
for min_mag, ax in zip(MIN_MAGS, axes):
    groups = smoothed_df[
        smoothed_df['mmin model'] == min_mag].groupby('layerid')
    data = [np.log10(group[prop]).values for _, group in groups]
    labels = ['layer %d' % id for id, _ in groups]
    ax.hist(data, label=labels, stacked=True, bins=np.arange(-7, 0, 0.5))
    ax.set_ylabel(('%s%g' % (prop, min_mag)).replace('.', '_'))
axes[0].legend(loc='upper left')
fig.savefig(smoothed_hist_pdf, dpi=300, transparent=True,
            bbox_inches='tight', pad_inches=0.1)
plt.close(fig)  # comment out line to view

# %%

print('Associate point sources in areal zones with those zones ...')
# quick, requires no transformations
start = time()

smoothed_df['distance'] = np.inf
smoothed_dfs = []
for layer_id in LAYERS_DF['id']:
    smoothed_layer_df = smoothed_df[smoothed_df['layerid'] == layer_id]
    areal_layer_df = areal_df[areal_df['layerid'] == layer_id]
    smoothed_layer_df = gpd.sjoin(
        smoothed_layer_df, areal_layer_df[['zoneid', 'a', 'geometry']],
        how='left')
    smoothed_dfs.append(smoothed_layer_df)
smoothed_df = pd.concat(smoothed_dfs)
smoothed_df.drop(columns='index_right', inplace=True)

smoothed_df['in zoneid'] = smoothed_df['zoneid'].copy()

assigned = (~np.isnan(smoothed_df['in zoneid'])) & (smoothed_df['a'] != 0)
smoothed_df.loc[assigned, 'distance'] = 0
print('Spatial join accounted for %.0f%% of sources [%.0f s]' %
      (100*len(smoothed_df[assigned])/len(smoothed_df), time() - start))

# In[]:

print('Find nearest areal zones for remaining points ...')
start = time()
active_areal_df['polygon'] = [
    MyPolygon([geo.point.Point(lat, lon)
               for lat, lon in zip(*zone.geometry.exterior.coords.xy)])
    for _, zone in active_areal_df.iterrows()]

unassigned_df = smoothed_df.loc[~assigned]
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

print('Nearest zone required for %.0f%% of sources [%.0f s]' %
      (100*len(unassigned_df)/len(smoothed_df), time() - start))

smoothed_df = pd.concat((smoothed_df[assigned], unassigned_df))

# %%

zone_id = 118
min_mag = 4.5
n_points_zone = ((smoothed_df['mmin model'] == min_mag) &
                 (smoothed_df['in zoneid'] == zone_id)).sum()
fmd_cols = ['zoneid', 'layerid', 'a', 'b', 'stdb', 'mmax', 'stdmmax',
            'mmin', 'geometry']

display(active_areal_df[
    active_areal_df['zoneid'] == zone_id][fmd_cols].squeeze())
print('Points in zone %d: %d' % (zone_id, n_points_zone))

# In[]:

# The truncated Gutenberg-Richter magnitude-frequency distribution in
# OpenQuakeimplements
# $$\lambda(m \geq M) = 10^{a - b m} = e^{\alpha - \beta m}$$
# where, since $\lambda$ is an annual rate, $10^a$ is too. If we ignore
# events below some threshold $m_{min}$ then the annual rate becomes
# $$\lambda(m \geq m_{min}) = e^{\alpha - \beta m_{min}} e^{-\beta (m - m_{min})} = \nu e^{-\beta (m - m_{min})} $$
# Thus to compute the $a$ value required by OpenQuake from the activity
# rate $\lambda$ for a given magnitude threshold, we must also take into
# account the $b$ value for the zone:
# $$a = \log_{10}(\lambda) + b m_{min}$$
# If instead what is provided is event counts $\nu$ over some catalog
# duration $T$, then one simply computes $\lambda = \nu/T$

# %%

print('For each point source, copy parameters of nearest areal zone')
columns_to_copy = ['zoneid', 'zmax', 'zmin', 'tectonic subregion',
                   'a', 'b', 'stdb', 'mmax', 'stdmmax',
                   'rake', 'dip', 'strike', 'aspect ratio', 'msr']
smoothed_df.drop(columns=['a'], inplace=True)
smoothed_df = smoothed_df.merge(active_areal_df[columns_to_copy],
                                on='zoneid')

pd.concat((smoothed_df.head(), smoothed_df.tail()))

# %%

# computing the a-value for each point is now easy
smoothed_df['a'] = (np.log10(smoothed_df['lambda']) +
                    smoothed_df['b']*smoothed_df['mmin model'])

# for each area in the areal model: count the number of points and
# sum the activity rates in the smoothed model. from the latter estimate
# an equivalent a-value
for min_mag, min_mag_df in smoothed_df.groupby('mmin model'):
    zone_dfs = min_mag_df.groupby('in zoneid')
    stats_df = zone_dfs.agg({'mmin': len, 'lambda': np.sum})

    smoothed_n = 'smoothed N ' + str(min_mag)
    smoothed_lambda = 'smoothed lambda ' + str(min_mag)

    stats_df.rename(columns={'mmin': smoothed_n,
                             'lambda': smoothed_lambda},
                    inplace=True)
    stats_df['zoneid'] = stats_df.index
    areal_df = areal_df.join(stats_df, on='zoneid', rsuffix='_stats')
    areal_df.drop(columns=['zoneid_stats'], inplace=True)

    areal_df['smoothed a ' + str(min_mag)] = (
        np.log10(areal_df[smoothed_lambda]) +
        areal_df['b']*min_mag).round(2)
    areal_df['equiv a ' + str(min_mag)] = (
        areal_df['a']/areal_df[smoothed_n]).round(2)
    areal_df['areal lambda ' + str(min_mag)] = (
        10**(areal_df['a'] - areal_df['b']*min_mag)).round(4)
    areal_df[smoothed_lambda] = areal_df[smoothed_lambda].round(4)

# %%

display_drop = ['zmax', 'zmin', 'aspect ratio', 'msr',
                'rake', 'dip', 'strike', 'stdb', 'stdmmax']
no_zoneid_df = smoothed_df[smoothed_df['zoneid'].isnull()]
no_mmax_df = smoothed_df[smoothed_df['mmax'] == 0]
no_b_df = smoothed_df[smoothed_df['b'] == 0]
if not no_zoneid_df.empty:
    print("Leftover points with no assigned zone id")
    display(no_zoneid_df.drop(display_drop, axis=1).head())
if not no_mmax_df.empty:
    print("Leftover points with no assigned mmax")
    display(no_mmax_df.drop(display_drop, axis=1).head())
if not no_b_df.empty:
    print("Leftover points with no assigned b")
    display(no_b_df.drop(display_drop, axis=1).head())
if not (no_mmax_df.empty and no_b_df.empty and no_zoneid_df.empty):
    print("SUCCESS: No points with unassigned MFD or zone")

# %%

display(pd.concat((smoothed_df.head(),
                   smoothed_df.tail())).drop(display_drop, axis=1))

# %%

this_lon_lat = ((smoothed_df['longitude'] == 98.0) &
                (smoothed_df['latitude'] == 3.8))
display(smoothed_df[this_lon_lat].drop(display_drop, axis=1))

# %%

areal_display_drop = display_drop + \
    ['polygon', 'tectonic zone', 'concerns', 'layerid']
pd.concat((areal_df.head(), areal_df.tail())).drop(areal_display_drop,
                                                   axis=1)

# %%

activity_file = os.path.join(
    smoothed_model_path, 'activity_rates_by_zone_areal_vs_smoothed.csv')
print('Writing activity summary to:\n\t%s' % activity_file)
activity_df = areal_df[[
    'zoneid', 'layerid', 'areal lambda 4.5', 'areal lambda 5.5',
    'smoothed lambda 4.5', 'smoothed lambda 5.5', 'smoothed N 4.5',
    'smoothed N 5.5']]
activity_df = activity_df.rename(
    columns={'areal lambda 4.5': 'areal 4.5',
             'smoothed lambda 4.5': 'smoothed 4.5',
             'areal lambda 5.5': 'areal 5.5',
             'smoothed lambda 5.5': 'smoothed 5.5',
             'smoothed N 4.5': 'N 4.5',
             'smoothed N 5.5': 'N 5.5'})

activity_df = activity_df.sort_values('zoneid').reset_index(drop=True)

for layer_id in LAYERS_DF['id']:
    series = pd.Series(activity_df[
        activity_df['layerid'] == layer_id].sum(axis=0))
    series['layerid'] = layer_id
    series['zoneid'] = 'All'
    activity_df = activity_df.append(series, ignore_index=True)

series = pd.Series(activity_df[activity_df['zoneid'] == 'All'].sum(axis=0))
series['layerid'] = 'All'
series['zoneid'] = 'All'
activity_df = activity_df.append(series, ignore_index=True)
activity_df['ratio 4.5'] = (activity_df['smoothed 4.5'] /
                            activity_df['areal 4.5']).round(2)
activity_df['ratio 5.5'] = (activity_df['smoothed 5.5'] /
                            activity_df['areal 5.5']).round(2)

activity_df.to_csv(activity_file)

# %%

# try computing layers directly
layer_activity_df = pd.DataFrame()
layer_activity_df.index.name = 'layerid'

for layer_id in LAYERS_DF['id']:
    in_areal_layer = areal_df['layerid'] == layer_id
    in_smoothed_layer = smoothed_df['layerid'] == layer_id
    in_a_zone = smoothed_df['distance'] == 0

    layer_series = pd.Series()
    for min_mag in MIN_MAGS:

        this_model = smoothed_df['mmin model'] == min_mag
        layer_series = layer_series.append(pd.Series({
            'areal ' + str(min_mag):
                (10**(areal_df[in_areal_layer]['a'] -
                      areal_df[in_areal_layer]['b'] *
                      min_mag)).sum().round(1),
            'smoothed ' + str(min_mag):
                smoothed_df[in_smoothed_layer &
                            this_model &
                            in_a_zone]['lambda'].sum().round(1),
            }, name=layer_id))
    layer_activity_df = layer_activity_df.append(layer_series,
                                                 ignore_index=True)

layer_activity_df = layer_activity_df.append(pd.Series(
    layer_activity_df.sum(axis=0), name='Total'))
for min_mag in MIN_MAGS:
    layer_activity_df['ratio ' + str(min_mag)] = (
        layer_activity_df['smoothed ' + str(min_mag)] /
        layer_activity_df['areal ' + str(min_mag)]).round(2)

display(layer_activity_df)

# %%

catalogue_df = pd.read_csv('../Catalogue/SACAT1900_2008v2.txt', sep='\t')
fig, ax = plt.subplots(figsize=(3, 3))
catalogue_df['SHOCK_TYPE'].value_counts().plot(kind='pie', ax=ax)
fig.savefig('ShockTypes.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1)
plt.close(fig)  # comment out line to view

# %%

# augment catalogue with zone and layer info
catalogue_df['geometry'] = [Point(lon, lat)
                            for lon, lat in zip(catalogue_df['LON'],
                                                catalogue_df['LAT'])]
layer_catalogue_gdfs = []
for _, layer in LAYERS_DF.iterrows():
    layer_catalogue_gdf = gpd.GeoDataFrame(catalogue_df[
        (catalogue_df['DEPTH'] >= layer['zmin']) &
        (catalogue_df['DEPTH'] < layer['zmax'])], crs='WGS84')
    layer_areal_gdf = gpd.GeoDataFrame(
        areal_df[areal_df['layerid'] == layer['id']]
        [['geometry', 'zoneid', 'layerid']], crs='WGS84')
    layer_catalogue_gdfs.append(gpd.sjoin(layer_catalogue_gdf,
                                          layer_areal_gdf, how='left'))
catalogue_df = pd.concat(layer_catalogue_gdfs).drop('geometry')

# %%

display(pd.concat((catalogue_df.head(),
                   catalogue_df.tail())))

# %%

# for each minimum magnitude and layer work out the activity rates
catalogue_activity_df = pd.DataFrame()
for _, layer in pd.merge(completeness_df, LAYERS_DF).iterrows():
    layer_results = pd.Series()
    for min_mag in reversed(MIN_MAGS):
        above_thresh = catalogue_df['MAG_MW'] >= min_mag
        start = layer[str(min_mag) + ' start']
        end = layer[str(min_mag) + ' end']
        at_depth = ((catalogue_df['DEPTH'] >= layer['zmin']) &
                    (catalogue_df['DEPTH'] < layer['zmax']))
        in_years = ((catalogue_df['YEAR'] >= start) &
                    (catalogue_df['YEAR'] <= end))
        in_a_zone = catalogue_df['zoneid'] != -1
        is_mainshock = catalogue_df['SHOCK_TYPE'] == 'Mainshock'
        subcat_df = catalogue_df[
            above_thresh & at_depth & in_years & in_a_zone & is_mainshock]
        layer_results = layer_results.append(pd.Series({
            'catalogue ' + str(min_mag):
                round(float(len(subcat_df))/(end - start + 1), 1),
            }, name=layer_id))
    catalogue_activity_df = catalogue_activity_df.append(layer_results,
                                                         ignore_index=True)
catalogue_activity_df = catalogue_activity_df.append(pd.Series(
    catalogue_activity_df.sum(axis=0), name='Total'))

# %%

summary_tex = os.path.join(
    smoothed_model_path,
    'activity_rates_by_layer_areal_vs_smoothed_vs_catalogue.tex')
print('Writing activity rates to:\n\t%s' % summary_tex)
activity_df = layer_activity_df.join(catalogue_activity_df)
model_type = [item.split()[0] for item in activity_df.columns]
model_type = ['catalogue' if item == 'cat' else item
              for item in model_type]
min_mag = [float(item.split()[1]) for item in activity_df.columns]
activity_df.columns = [min_mag, model_type]
multi_cols = pd.MultiIndex.from_tuples([
    (4.5, 'areal'), (4.5, 'smoothed'), (4.5, 'catalogue'),
    (5.5, 'areal'), (5.5, 'smoothed'), (5.5, 'catalogue'),
    ], names=['source', 'minimum magnitude'])
# activity_df[('layerid', 'minimum magnitude')] = activity_df.index
# activity_df.index = range(len(activity_df))

activity_df = activity_df[multi_cols]
activity_df.to_latex(summary_tex, index_names=True,)

# %%

PLOT_PARAMS = ['a', 'b', 'nu', 'lambda']
LOG_PARAMS = ['nu', 'lambda']

for param in PLOT_PARAMS:
    file_name = ("Histogram_%s.png" % param)
    print('Plotting histograms of %s:\n\t%s' % (param, file_name))

    all_data = smoothed_df[param].values
    if param in LOG_PARAMS:
        bins = logspace(np.nanmin(all_data), np.nanmax(all_data), 3)
    else:
        bins = linspace(np.nanmin(all_data), np.nanmax(all_data))

    fig, axes = plt.subplots(len(LAYERS_DF), len(MIN_MAGS),
                             figsize=(6.5, 8.5), squeeze=False,
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for layer_id, row_axes in zip(LAYERS_DF['id'], axes):
        for min_mag, ax in zip(MIN_MAGS, row_axes):
            annotate('layer %d mmin %g' % (layer_id, min_mag),
                     loc='upper right', ax=ax, prop={'size': 8})

            subset_indices = (
                (smoothed_df['mmin model'] == min_mag) &
                (smoothed_df['layerid'] == layer_id))

            ax.hist(smoothed_df[subset_indices][param],
                    bins=bins)
            if param in LOG_PARAMS:
                ax.set_xscale('log')

    for ax in axes[:, -1]:
        ax.set_xlabel(param)

    fig.savefig(file_name, dpi=150, transparent=True,
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # comment out line to view

# %%

for param in PLOT_PARAMS:

    file_name = ("SmoothedEquivalentMap_%s.png" % param)
    print('Plotting map of %s:\n\t%s' % (param, file_name))

    all_data, longitudes, latitudes = extract_param(smoothed_df, param)
    if param in LOG_PARAMS:
        ticks = logspace(np.nanmin(all_data), np.nanmax(all_data), 3)
        norm = LogNorm(vmin=ticks[0], vmax=ticks[-1])
    else:
        ticks = linspace(np.nanmin(all_data), np.nanmax(all_data), 7)
        norm = Normalize(vmin=ticks[0], vmax=ticks[-1])

    extent = (longitudes[0], longitudes[-1], latitudes[0], latitudes[-1])

    fig, axes = plt.subplots(len(LAYERS_DF), len(MIN_MAGS),
                             figsize=(6.5, 8.5), squeeze=False,
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for layer_id, row_axes in zip(LAYERS_DF['id'], axes):
        for min_mag, ax in zip(MIN_MAGS, row_axes):
            annotate('layer %d mmin %g' % (layer_id, min_mag),
                     loc='lower left', ax=ax)
            subset_indices = (
                (smoothed_df['mmin model'] == min_mag) &
                (smoothed_df['layerid'] == layer_id))

            data = extract_param(smoothed_df[subset_indices], param)[0]
            image = ax.imshow(
                data, cmap='jet', origin='lower', aspect='equal',
                extent=extent, norm=norm)
            ax.grid()

    for ax in axes[:, 0]:
        ax.set_ylabel(u'Latitude (°N)')
    for ax in axes[-1, :]:
        ax.set_xlabel(u'Longitude (°E)')

    fig.colorbar(image, ax=axes.ravel().tolist(), label=param,
                 shrink=1/len(LAYERS_DF), ticks=ticks)

    fig.savefig(file_name, dpi=300, transparent=True,
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # comment out line to view

# %% Thinning of models allows quick testing and git archiving of a sample

res_deg = 1
thinned_df = smoothed_df.loc[
    np.isclose(np.remainder(smoothed_df['latitude'], res_deg), 0) &
    np.isclose(np.remainder(smoothed_df['longitude'], res_deg), 0)]
model_basename = ' '.join((os.path.split(smoothed_model_path)[1],
                           smoothed_source_data_file))
print('Thinning to %g° spacing reduces number of points from %d to %d.'
      % (res_deg, len(smoothed_df), len(thinned_df)))

# %%

mark = time()
points2csv(thinned_df, model_basename + ' thinned', MIN_MAGS,
           LAYERS_DF['id'])
points2csv(smoothed_df, model_basename, MIN_MAGS, LAYERS_DF['id'])

print('Finished writing CSV source models [%.0f s]' % (time() - mark))

# %%

# write point source models without twinning to TSV file
# for min_mag in MIN_MAGS:
#    smoothed_output_df = smoothed_df[smoothed_df['mmin model'] == min_mag]
#    smoothed_output_df = sort_and_reindex(
#        add_name_id(smoothed_output_df))
#    smoothed_output_df.drop(['mmin model'], axis=1, inplace=True)
#    smoothed_output_df['lambda'] = limit_precision(
#        smoothed_output_df['lambda'].values)
#    smoothed_output_df['a'] = limit_precision(
#        smoothed_output_df['a'].values)
#    smoothed_tsv = '%s_%s_mmin_%g.tsv' % (
#        os.path.split(smoothed_model_path)[1],
#        smoothed_source_data_file, min_mag)
#    smoothed_output_df.to_csv(smoothed_tsv, sep='\t', index=False)

# %%

mark = time()
points2nrml(thinned_df, model_basename + ' thinned', MIN_MAGS)
points2nrml(smoothed_df, model_basename, MIN_MAGS)
print('Finished writing NRML source models [%.0f s]' % (time() - mark))

# %%


#if __name__ == '__main__':
#    sys.exit(main())
