# coding: utf-8
'''
Areal Source models for Nath & Thingbaijam (2012)

Read the source description input files from the online supplementary
material and write them to XML.

Note: For imports to work, ../utilities directory must be added to PYTHONPATH
'''
# pylint: disable=invalid-name

# %%

import os
# import sys
from time import time

import numpy as np
import pandas as pd
import geopandas as gpd
from IPython.display import display

import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon

from obspy.imaging.beachball import beachball, aux_plane

from openquake.hazardlib import geo
from openquake.hmtk.sources.source_model import mtkSourceModel
from openquake.hmtk.plotting.mapping import HMTKBaseMap
from openquake.hmtk.parsers.source_model.nrml04_parser import \
    nrmlSourceModelParser

from source_model_tools import (
    PseudoCatalogue, read_polygons, MyPolygon, focal_mech, faulting_style,
    sort_and_reindex, add_name_id, add_binwise_rates, twin_source_by_magnitude,
    source_df_to_kml, source_df_to_list,
    )
from toolbox import wrap, annotate

# %%
# define some lists needed at different stages
MIN_MAGS = [4.5, 5.5]

LAYER_IDS = [1, 2, 3, 4]
LAYER_DEPTHS_KM = [0., 25., 70., 180., 300.]

LAYERS_DF = pd.DataFrame(list(zip(LAYER_IDS,
                                  LAYER_DEPTHS_KM[:-1],
                                  LAYER_DEPTHS_KM[1:])),
                         columns=['id', 'zmin', 'zmax'])
LAYERS_DF['id'] = LAYERS_DF['id'].astype(int)
LAYERS_DF.to_csv('layers.csv', index=False)
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

# # @profile
# def main():

# %%

print('Reading areal polygons and seismicity statistics for each layer')
areal_df = pd.DataFrame()
areal_polygons_df = pd.DataFrame()
for i, layer in LAYERS_DF.iterrows():

    # read seismicity and polygons
    layer_seis_df = pd.read_csv(os.path.join(
        MODEL_PATH, SEISMICITY_FORMAT % layer['id']))
    layer_seis_df.rename(columns={'avalue': 'a',
                                  'bvalue': 'b',
                                  'stdbvalue': 'stdb'}, inplace=True)
    layer_poly_df = read_polygons(os.path.join(
        MODEL_PATH, POLYGON_FORMAT % layer['id']))

    # fill in depths, specify source mechanisms, clean up dip & rake
    n_zones = len(layer_seis_df)
    idx = layer_seis_df.index
    layer_seis_df['zmin'] = pd.Series(np.full(n_zones, layer['zmin']),
                                      index=idx)
    layer_seis_df['zmax'] = pd.Series(np.full(n_zones, layer['zmax']),
                                      index=idx)
    layer_seis_df['layerid'] = pd.Series(np.full(n_zones, layer['id']),
                                         index=idx)

    # put it all together
    layer_source_df = pd.merge(layer_seis_df, layer_poly_df, on='zoneid')
    areal_df = pd.concat([areal_df, layer_source_df], ignore_index=True)
    areal_polygons_df = pd.concat([areal_polygons_df, layer_poly_df],
                                  ignore_index=True)

# %%

print('Merge with auxiliary data (crucially, tectonic zones)')
aux_df = pd.read_csv(AUX_FILE)
aux_df = aux_df.drop(['zmin', 'zmax', 'dip', 'rake', 'mechanism'], axis=1)
areal_df = pd.merge(areal_df, aux_df, on='zoneid')

# convert polygons coordinates to objects
areal_df['polygon'] = [
    MyPolygon([geo.point.Point(lat, lon)
               for lat, lon in area_series['polygon coordinates']])
    for _, area_series in areal_df.iterrows()]

areal_df['geometry'] = [Polygon(coordinates)
                        for coordinates in areal_df['polygon coordinates']]
(areal_df['centroid longitude'],
 areal_df['centroid latitude']) = zip(*[
     [item[0] for item in geometry.centroid.coords.xy]
     for geometry in areal_df['geometry']])

# convert zoneid to string - sort_and_reindex takes care of "human" sorting
areal_df['zoneid'] = [str(item) for item in areal_df['zoneid']]
areal_df = sort_and_reindex(areal_df)

# %%

# add some information to tables
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

# %%

areal_df[areal_df['zoneid'] == '936'].squeeze()

# %%

# read in additional information provided by Kiran Thingbaijam
aux2_df = pd.read_csv(AUX2_FILE, na_values=['nan'],
                      keep_default_na=False, )
aux2_df['zoneid'] = aux2_df['zoneid'].astype(str)

# %%

# merge this info with the areal zone table
areal2_df = pd.merge(areal_df, aux2_df,
                     on=['zoneid', 'layerid'], how='left')
areal2_df.fillna('', inplace=True)
areal2_df.sort_values(['layerid', 'zoneid'],
                      ascending=[False, True], inplace=True)

# %%

print('These zones are duplicated')
duplicated_df = areal2_df[
    areal2_df.duplicated(['strike', 'dip', 'rake'], keep=False) &
    ~pd.isnull(areal2_df['dip']) &
    (areal2_df['mechanism'] != 'undefined')].copy()
duplicated_df.sort_values('dip', inplace=True)
display(duplicated_df[
    ['zoneid', 'layerid', 'faulting style', 'new style',
     'strike', 'dip', 'rake', 'mechanism',
     'strike2', 'dip2', 'rake2', 'mechanism2']
])

# %%

# bring terminologies in line for comparison
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

# %%

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
    for x, y, zone_id in zip(group['rake'], group['dip'], group['zoneid']):
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

# %%
print('These zones had the wrong faulting style in the initial implementation')
display(wrong_df[
    ['zoneid', 'layerid', 'faulting style', 'new style',
     'strike', 'dip', 'rake', 'mechanism',
     'strike2', 'dip2', 'rake2', 'mechanism2']])

# In[]:
bb_dir = './beachballs/'
print('Generating SVG beachballs for areal zones for display in QGIS:\n\t' +
      os.path.abspath(bb_dir))

mark = time()

if not os.path.isdir(bb_dir):
    os.mkdir(bb_dir)

for _, zone in areal_df.iterrows():
    fig = plt.figure(1)
    focal_mechanism = (zone['strike'], zone['dip'], zone['rake'])
    file_name = os.path.join(bb_dir, zone['zoneid'] + '.svg')
    beachball(focal_mechanism, outfile=file_name, fig=fig,
              facecolor='black')
print('Wrote %d beachballs (%.0f s)' % (len(areal_df), time() - mark))

# In[]:
print('Read significant events from Nath (2011)')

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
    fig = plt.figure(1)
    focal_mechanism = (event['φ'], event['δ'], event['Λ'])
    file_name = os.path.join(
        bb_dir, str(event['Date'].date()).replace('-', '') + '.svg')
    beachball(focal_mechanism, outfile=file_name, fig=fig,
              facecolor='blue')
print('Wrote %d beachballs (%.0f s)' % (len(df_events), time() - mark))

# %%

print('These zones have different tectonic region types')
display(different_trt_df[[
    'zoneid', 'layerid', 'zmin', 'zmax', 'strike', 'dip', 'rake', 'a',
    'mechanism', 'faulting style',
    'tectonic zone', 'tectonic subregion', 'tectonic region type'
]])

# %%

print('These zones have different faulting styles')
display(different_mechanism_df[[
    'zoneid', 'layerid', 'zmin', 'zmax', 'strike', 'dip', 'rake',
    'mechanism', 'faulting style',
]])

# %%

keep_columns = ['zoneid', 'layerid', 'zmin', 'zmax',
                'strike', 'dip', 'rake',
                'mechanism', 'mechanism2', 'faulting style', 'new style']
display(different_mechanism_df[
    different_mechanism_df['faulting style'] !=
    different_mechanism_df['new style']][keep_columns])

# %%

print('Zones without seismicity')
display(aux_df[aux_df['tectonic subregion'] == 'no seismicity'])

# %%

areal_df[areal_df['stdb'] != 0]['stdb'].describe()

# %%

# show a summary including megathrust zones and bin statistics
drop_columns = ['tectonic zone', 'region', 'concerns', 'zmax', 'zmin',
                'polygon coordinates', 'polygon', 'geometry',
                'aspect ratio', 'dip', 'rake', 'strike']
display(pd.concat([areal_df.drop(drop_columns, axis=1).head(),
                   areal_df.drop(drop_columns, axis=1).tail()]))

# %%

areal_df[areal_df['tectonic subregion'] == 'no seismicity'].drop(
    drop_columns, axis=1)

# %%

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

# %%

print(set(areal_df['tectonic subregion']))
areal_df[areal_df['tectonic subregion'] == 'no seismicity'].drop(
    drop_columns, axis=1)

# %%

catalogue_df = pd.read_csv(CATALOGUE_FILE, sep='\t')

fig, ax = plt.subplots(figsize=(6, 6))
catalogue_df['SHOCK_TYPE'].value_counts().plot(kind='pie', ax=ax)
fig.savefig('ShockTypes.png', transparent=True, bbox_inches='tight',
            pad_inches=0.1)
plt.close(fig)  # uncomment to view

# %%

print('Associating catalogue events wtih zones')
mark = time()
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
    layer_catalogue_gdfs.append(
        gpd.sjoin(layer_catalogue_gdf, layer_areal_gdf,
                  how='left', op='intersects'))
catalogue_df = pd.concat(layer_catalogue_gdfs).drop('geometry')
print('Associated %d events with %d zones (%.0f s)' %
      (len(catalogue_df), len(areal_df), time() - mark))
display(pd.concat((catalogue_df.head(), catalogue_df.tail())))

# %%

print('Read completeness tables')
completeness_df = pd.read_csv(COMPLETENESS_FILE,
                              header=[0, 1], index_col=[0, 1])
# completeness_df.reset_index(inplace=True)
completeness_df.columns = [' '.join(col).strip()
                           for col in completeness_df.columns.values]
completeness_df.reset_index(inplace=True)
display(completeness_df)

# %%

print('Compute areal zone activity rates from catalogue')
catalogue_activity_df = pd.DataFrame()
for _, layer in pd.merge(completeness_df, LAYERS_DF).iterrows():
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
            }, name=layer['id']))
    catalogue_activity_df = catalogue_activity_df.append(layer_results,
                                                         ignore_index=True)
catalogue_activity_df = catalogue_activity_df.append(pd.Series(
    catalogue_activity_df.sum(axis=0), name='Total'))

# %%
print('Writing areal source model CSV table: ' +
      AREAL_SOURCE_MODEL_BASE + '.csv')
areal2_df.to_csv(AREAL_SOURCE_MODEL_BASE + '.csv', index=False)

# %%

# print('Writing areal source model TSV table with zones twinned by '
#      'magnitude: ' + AREAL_SOURCE_MODEL_BASE + '.tsv')
# areal_output_df = sort_and_reindex(add_name_id(
#    twin_source_by_magnitude(areal_df)).drop(
#        ['polygon', 'polygon coordinates'], axis=1))
# areal_output_df.to_csv(AREAL_SOURCE_MODEL_BASE + '.tsv', sep='\t',
#                       index=False, float_format='%.12g')
# areal_output_df = sort_and_reindex(add_name_id(
#    areal_df).drop(['polygon', 'geometry'], axis=1))
# areal_output_df.to_csv(AREAL_SOURCE_MODEL_BASE + '_no_twin.tsv',
#                       sep='\t', index=False, float_format='%.12g')

# %%

print('Writing areal source model KML with added binwise rates: ' +
      AREAL_SOURCE_MODEL_BASE + '.kml')
areal_kml_df = add_name_id(
    add_binwise_rates(areal_df).drop(['polygon', 'geometry'], axis=1))
for layer_id in LAYERS_DF['id']:
    this_layer = areal_kml_df['layerid'] == layer_id
    temp_df = areal_kml_df.drop(['layerid'], axis=1)
    source_df_to_kml(temp_df.loc[this_layer, :],
                     '%s layer %d' % (AREAL_SOURCE_MODEL_BASE, layer_id))

# %%

print('Writing NRML areal source model with sources twinned by magnitude: ' +
      AREAL_SOURCE_MODEL_BASE + '.xml')
areal_source_list = source_df_to_list(
    add_name_id(twin_source_by_magnitude(areal_df)))
model_name = os.path.split(MODEL_PATH)[1] + ' areal'
areal_source_model = mtkSourceModel(identifier='1', name=model_name,
                                    sources=areal_source_list)
mark = time()
areal_source_model.serialise_to_nrml(AREAL_SOURCE_MODEL_BASE + '.xml')
print('Serialized %d sources to NRML (%.0f s)' %
      (len(areal_source_list), time() - mark))

# %%

print('Printing layer-by-layer maps of dominant mechanisms in each zone')
mark = time()
map_config = {'min_lon': 60, 'max_lon': 105,
              'min_lat': 0, 'max_lat': 40, 'resolution': 'l'}
parser = nrmlSourceModelParser(AREAL_SOURCE_MODEL_BASE + '.xml')

catalogue = PseudoCatalogue(areal_source_model)
for depth in sorted(list(set(catalogue.data['depth']))):
    basemap = HMTKBaseMap(map_config, '', lat_lon_spacing=5)

    source_model_read = parser.read_file('Areal Source Model')
    selected_sources = [source for source in source_model_read.sources
                        if source.hypo_depth_dist.data[0][1] == depth]
    source_model_read.sources = selected_sources
    selected_catalogue = PseudoCatalogue(source_model_read)

    basemap.add_source_model(source_model_read, overlay=True,
                             area_border='0.5')
    basemap.add_focal_mechanism(selected_catalogue, magnitude=False)

    ax = basemap.fig.gca()
    annotate('%g km' % depth, loc='lower left', ax=ax)
    for _, zone in selected_catalogue.data.iterrows():
        x, y = basemap.m(zone.longitude, zone.latitude)
        ax.annotate(s=zone.id.replace('z', ''), xy=(x, y), color='green',
                    xytext=(0, 10), textcoords='offset points', zorder=100,
                    fontsize=8, fontweight='bold',
                    horizontalalignment='center')
    basemap.fig.savefig('ArealModelFocalMechanisms%gkm.png' % depth,
                        dpi=300, transparent=False,
                        bbox_inches='tight', pad_inches=0.1)

plt.close('all')  # uncomment to view
print('Generated maps of %d areal zones at %d depths (%.0f s)' %
      (len(areal_source_model.sources), len(set(catalogue.data['depth'])),
       time() - mark))


# if __name__ == '__main__':
#     sys.exit(main())
