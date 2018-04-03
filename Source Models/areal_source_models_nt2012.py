# coding: utf-8
'''
Areal source models for Nath & Thingbaijam (2012)

Read the source description input files from the online supplementary
material and write them to XML.

Note: For imports to work, ../utilities directory must be added to PYTHONPATH
'''
# pylint: disable=invalid-name

# %% imports

import os
# import sys
from time import time

import numpy as np
import pandas as pd
import geopandas as gpd
from IPython.display import display

import matplotlib.pyplot as plt

from shapely.geometry import Point
from more_itertools import unique_everseen

from obspy.imaging.beachball import beachball, aux_plane

from source_model_tools import (
    read_polygons, focal_mech, faulting_style, df2nrml, areal2csv,
    SEISMICITY_ALIASES)
from toolbox import wrap

pd.set_option('mode.chained_assignment', 'raise')


# %% constants

MIN_MAGS = [4.5, 5.5]

LAYER_DEPTHS_KM = [0., 25., 70., 180., 300.]

LAYERS_DF = pd.DataFrame(
    list(zip(LAYER_DEPTHS_KM[:-1],
             LAYER_DEPTHS_KM[1:])),
    index=[1, 2, 3, 4],
    columns=['zmin', 'zmax'])
LAYERS_DF.index.name = 'layerid'
LAYERS_DF.to_csv('layers.csv')
display(LAYERS_DF)

# define the input file names from the original paper
MODEL_PATH = '../Data/nath2012probabilistic'
POLYGON_FORMAT = 'polygonlay%d.txt'
SEISMICITY_FORMAT = 'seismicitylay%d.txt'

# an input file supplies some auxiliary data
AUX_FILE = 'auxiliary data.csv'
AUX2_FILE = 'TRT_assignments_KKST.csv'
SIGNIFICANT_EVENTS_FILE = '../Data/nath2011peak/Table6.csv'
COMPLETENESS_FILE = '../Data/thingbaijam2011seismogenic/Table1.csv'
CATALOGUE_FILE = '../Catalogue/SACAT1900_2008v2.txt'

# define prefixes for the output file names and models
AREAL_SOURCE_MODEL_BASE = 'areal_source_model'

# %% load electronic supplement

print('Reading areal polygons and seismicity statistics for each layer')
areal_dfs = []
for layer_id in LAYERS_DF.index:

    # read seismicity and polygons and join them
    seismicity_file = os.path.join(MODEL_PATH, SEISMICITY_FORMAT % layer_id)
    print('Reading: ' + os.path.abspath(seismicity_file))
    seismicity_df = pd.read_csv(seismicity_file)
    seismicity_df.set_index('zoneid', inplace=True, verify_integrity=True)
    seismicity_df.rename(columns=SEISMICITY_ALIASES, inplace=True)

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
aux_df = aux_df.drop(['dip', 'rake', 'mechanism'], axis=1)
assert (areal_df.index == aux_df.index).all()
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

# %% query a zone

print('\nA typical zone:')
areal_df.loc[936]

# %% add information provided by Kiran Thingbaijam

print('\nReading: ' + os.path.abspath(AUX2_FILE))
aux2_df = pd.read_csv(AUX2_FILE, na_values=['nan'], keep_default_na=False)
aux2_df.set_index('zoneid', inplace=True, verify_integrity=True)
aux2_df.sort_index(inplace=True)
assert (areal_df.index == aux2_df.index).all()
assert (areal_df['layerid'] == aux2_df['layerid']).all()

areal2_df = areal_df.join(aux2_df, rsuffix='2', how='outer')
areal2_df.fillna('', inplace=True)

print('\nThese zones have identical focal mechanisms:')
duplicated_df = areal2_df[
    areal2_df.duplicated(['strike', 'dip', 'rake'], keep=False) &
    ~pd.isnull(areal2_df['dip']) &
    (areal2_df['mechanism'] != 'undefined')].copy()
duplicated_df.sort_values('dip', inplace=True)
display(duplicated_df[
    ['layerid', 'faulting style', 'new style',
     'strike', 'dip', 'rake', 'mechanism',
     'strike2', 'dip2', 'rake2', 'mechanism2']
])

# %% bring terminologies in line for comparison

areal2_df['tectonic subregion'] = (
    areal2_df['tectonic subregion']
    .str.lower()
    .str.replace('himalayas', '')
    .str.replace('strike-slip reverse', '')
    .str.replace('normal', '')
    .str.strip()
)
areal2_df['tectonic zone'] = (
    areal2_df['tectonic zone']
    .str.lower()
)
areal2_df['mechanism'] = (
    areal2_df['mechanism']
    .replace('dextral', 'strike-slip')
    .replace('sinistral', 'strike-slip')
)
areal2_df['mechanism2'] = (
    areal2_df['mechanism2']
    .replace('dextral', 'strike-slip')
    .replace('sinistral', 'strike-slip')
)

different_mechanism_df = areal2_df[
    (areal2_df['mechanism'] != areal2_df['faulting style']) &
    (areal2_df['tectonic subregion'] != 'no seismicity') &
    (areal2_df['dip'].apply(np.isreal))
]
different_trt_df = areal2_df[
    (areal2_df['tectonic subregion'] !=
     areal2_df['tectonic region type']) &
    (areal2_df['tectonic subregion'] != 'no seismicity')
]

print('%d/%d (%d%%) mechanisms different' %
      (len(different_mechanism_df), len(areal_df),
       100*len(different_mechanism_df)/len(areal_df)))
print('%d/%d (%d%%) TRTs different' %
      (len(different_trt_df), len(areal_df),
       100*len(different_trt_df)/len(areal_df)))

# %% plot focal plane alternatives

subset_df = areal2_df[
    areal2_df['dip'].apply(np.isreal) &
    (areal2_df['dip'] != -1) &
    (areal2_df['faulting style'] != '')]
wrong_df = different_mechanism_df[
    different_mechanism_df['dip'].apply(np.isreal) &
    (different_mechanism_df['dip'] != -1) &
    (different_mechanism_df['faulting style'] != '')]
colours = {
    'normal': 'red',
    'reverse': 'green',
    'strike-slip': 'blue'
}

fig, ax = plt.subplots()
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

for name, group in subset_df.groupby('faulting style'):
    ax.plot(group['rake'], group['dip'], color=colours[name],
            marker='o', markersize=15, linestyle='', linewidth=2,
            label=name)
    for x, y, zone_id in zip(group['rake'], group['dip'], group.index):
        ax.annotate(s=zone_id, xy=(x, y),
                    xytext=(0, 0), textcoords='offset points',
                    fontsize=6, fontweight='bold', color='white',
                    horizontalalignment='center',
                    verticalalignment='center')
for _, zone in wrong_df.iterrows():
    ax.plot([zone['rake'], zone['rake2']], [zone['dip'], zone['dip2']],
            linestyle=':', color='black', linewidth=0.5, label=None)
    ax.plot(zone['rake'], zone['dip'], color=colours[zone['mechanism']],
            marker='o', markersize=10, markeredgecolor='black',
            linestyle='', linewidth=2, label=None)
    ax.plot(zone['rake2'], zone['dip2'], color=colours[zone['mechanism2']],
            marker='o', markersize=10, markeredgecolor='black',
            linestyle='', linewidth=2, label=None)

ax.legend(loc='lower left')
for threshhold in [-150, -30, 30, 150]:
    ax.axvline(x=threshhold, linestyle='--', color='grey', linewidth=0.5)
ax.set_xlim((-180, 180))
ax.set_xlabel(('Rake [°]'))
ax.set_xticks(range(-180, 181, 45))
ax.set_ylim((0, 90))
ax.set_ylabel(('Dip [°]'))
ax.set_yticks(range(0, 91, 45))

fig.savefig('FocalMechanisms.png', transparent=True,
            bbox_inches='tight', pad_inches=0.1)
plt.close(fig)  # uncomment to view

# %% summarize discrepancies
print('\nZoes wrong faulting style in the initial implementation')
display(wrong_df[
    ['layerid', 'faulting style', 'new style',
     'strike', 'dip', 'rake', 'mechanism',
     'strike2', 'dip2', 'rake2', 'mechanism2']])


print('\nZones with different tectonic region types')
display(different_trt_df[[
    'layerid', 'strike', 'dip', 'rake', 'a',
    'mechanism', 'faulting style',
    'tectonic zone', 'tectonic subregion', 'tectonic region type'
]])

print('\nZones with different faulting styles')
display(different_mechanism_df[[
    'layerid', 'strike', 'dip', 'rake',
    'mechanism', 'faulting style',
]])

print('\nZones assigned different faulting styles')

keep_columns = ['layerid', 'strike', 'dip', 'rake',
                'mechanism', 'mechanism2', 'faulting style', 'new style']
display(different_mechanism_df[
    different_mechanism_df['faulting style'] !=
    different_mechanism_df['new style']][keep_columns])

print('\nNon-zero b-values:')
print(areal_df[areal_df['stdb'] != 0]['stdb'].describe())

print('\nHighest and lowest activity rates')
drop_columns = ['tectonic zone', 'region', 'concerns', 'polygon', 'geometry',
                'aspect ratio', 'dip', 'rake', 'strike']
temp_df = areal_df[areal_df.a != 0].drop(drop_columns, axis=1).sort_values('a')
display(pd.concat([temp_df.head(), temp_df.tail()]))

print('\nZones without seismicity:')
areal_df[areal_df['tectonic subregion'] == 'no seismicity'].drop(
    drop_columns, axis=1)

print('\nTectonic subregions:', set(areal_df['tectonic subregion']))


# %% histograms of FMD parameters

props = ['a', 'b', 'mmax']
ranges = [np.arange(-0.5, 8, 1),
          np.arange(-0.1, 1.7, 0.2),
          np.arange(-0.5, 10, 1)]
groups = areal_df.groupby('layerid')
fig, axes = plt.subplots(nrows=len(props), ncols=1,
                         figsize=(6, 3*len(props)))
for prop, ax, bins in zip(props, axes, ranges):
    data = [group[prop] for _, group in groups]
    labels = ['layer %d' % layer_id for layer_id, _ in groups]
    ax.hist(data, label=labels, stacked=True, bins=bins)
    ax.set_ylabel(prop)
axes[0].legend(loc='upper left')
fig.savefig('ArealModelFmds.png', dpi=300,
            transparent=True, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)  # uncomment to view


# %% load catalogue

print('\nReading catalogue: ' + os.path.abspath(CATALOGUE_FILE))
catalogue_df = pd.read_csv(CATALOGUE_FILE, sep='\t')

fig, ax = plt.subplots(figsize=(6, 6))
catalogue_df['SHOCK_TYPE'].value_counts().plot(kind='pie', ax=ax)
fig.savefig('ShockTypes.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1)
plt.close(fig)  # uncomment to view

# %% categorize catalogue by zone

print('\nAssociating catalogue events wtih zones')
mark = time()
catalogue_df['geometry'] = [Point(lon, lat)
                            for lon, lat in zip(catalogue_df['LON'],
                                                catalogue_df['LAT'])]
layer_catalogue_gdfs = []
for layer_id, layer in LAYERS_DF.iterrows():
    layer_catalogue_gdf = gpd.GeoDataFrame(catalogue_df[
        (catalogue_df['DEPTH'] >= layer['zmin']) &
        (catalogue_df['DEPTH'] < layer['zmax'])], crs='WGS84')
    layer_areal_gdf = gpd.GeoDataFrame(
        areal_df[areal_df['layerid'] == layer_id].reset_index()
        [['geometry', 'zoneid', 'layerid']], crs='WGS84')
    layer_catalogue_gdfs.append(
        gpd.sjoin(layer_catalogue_gdf, layer_areal_gdf,
                  how='left', op='intersects'))
catalogue_df = pd.concat(layer_catalogue_gdfs).drop('geometry')
print('Associated %d events with %d zones (%.0f s)' %
      (len(catalogue_df), len(areal_df), time() - mark))
display(pd.concat((catalogue_df.head(), catalogue_df.tail())))

# %% completeness tables

print('\nReading completeness: ' + os.path.abspath(COMPLETENESS_FILE))
completeness_df = pd.read_csv(COMPLETENESS_FILE,
                              header=[0, 1], index_col=[0, 1])
# completeness_df.reset_index(inplace=True)
completeness_df.columns = [' '.join(col).strip()
                           for col in completeness_df.columns.values]
display(completeness_df)

# %% areal zone activity rates

print('\nCompute areal zone activity rates from catalogue')
catalogue_activity_df = pd.DataFrame()
for layer_id, layer in LAYERS_DF.join(completeness_df,
                                      on=['zmin', 'zmax']).iterrows():
    layer_results = pd.Series()
    for mag in reversed(MIN_MAGS):
        above_thresh = catalogue_df['MAG_MW'] >= mag
        start = layer[str(mag) + ' start']
        end = layer[str(mag) + ' end']
        at_depth = ((catalogue_df['DEPTH'] >= layer['zmin']) &
                    (catalogue_df['DEPTH'] < layer['zmax']))
        in_years = ((catalogue_df['YEAR'] >= start) &
                    (catalogue_df['YEAR'] <= end))
        in_a_zone = catalogue_df['zoneid'] != -1
        is_mainshock = catalogue_df['SHOCK_TYPE'] == 'Mainshock'
        subcat_df = catalogue_df[
            above_thresh & at_depth & in_years & in_a_zone & is_mainshock]
        layer_results = layer_results.append(pd.Series({
            'catalogue ' + str(mag):
                round(float(len(subcat_df))/(end - start + 1), 1),
            }, name=layer_id))
    catalogue_activity_df = catalogue_activity_df.append(layer_results,
                                                         ignore_index=True)
catalogue_activity_df = catalogue_activity_df.append(pd.Series(
    catalogue_activity_df.sum(axis=0), name='Total'))

# %% write CSV

areal2csv(areal_df, AREAL_SOURCE_MODEL_BASE)

# %% write NRML

mark = time()
areal_source_model = df2nrml(areal_df, AREAL_SOURCE_MODEL_BASE)
print('Finished writing NRML source models [%.0f s]' % (time() - mark))


# %% write SVG baechballs for QGIS

bb_dir = './beachballs/'
print('\nGenerating SVG beachballs for areal zones for display in QGIS:\n\t' +
      os.path.abspath(bb_dir))

mark = time()

if not os.path.isdir(bb_dir):
    os.mkdir(bb_dir)

for zoneid, zone in areal_df.iterrows():
    focal_mechanism = (zone['strike'], zone['dip'], zone['rake'])
    file_name = os.path.join(bb_dir, str(zoneid) + '.svg')
    beachball(focal_mechanism, outfile=file_name, facecolor='black')
    plt.close()
print('Wrote %d beachballs (%.0f s)' % (len(areal_df), time() - mark))

print('Reading significant events: ' +
      os.path.abspath(SIGNIFICANT_EVENTS_FILE))
df_events = pd.read_csv(SIGNIFICANT_EVENTS_FILE,
                        parse_dates=['Date'], na_values='-')
df_events.sort_values('H (km)', ascending=True, inplace=True)
print('Deep events:')
display(df_events[(df_events['H (km)'] >= 180) & (df_events['H (km)'] < 600)])

bb_dir = '../Data/nath2011peak/'
print('Generating SVG beachballs for significant events '
      'for display in QGIS:\n\t' + os.path.abspath(bb_dir))

mark = time()

if not os.path.isdir(bb_dir):
    os.mkdir(bb_dir)
for _, event in df_events.iterrows():
    focal_mechanism = (event['φ'], event['δ'], event['Λ'])
    file_name = os.path.join(
        bb_dir, str(event['Date'].date()).replace('-', '') + '.svg')
    beachball(focal_mechanism, outfile=file_name, facecolor='blue')
    plt.close()
print('Wrote %d beachballs (%.0f s)' % (len(df_events), time() - mark))
