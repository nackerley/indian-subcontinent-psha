# coding: utf-8
'''
Collapsing of frequency magnitude distributions for Nath & Thingbaijam (2012)

OpenQuake isn't up to the task of enumerating $3^{222} \approx 0.83$ million
googols of logic tree branches, so we will collapse all $m_{max}$ and $b$
variation for each areal zone into one discrete distribution.

Note: For imports to work, ../utilities directory must be added to PYTHONPATH
'''

# %% imports
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display

from toolbox import annotate
from logic_tree_tools import read_tree_tsv, collapse_sources
from source_model_tools import csv2areal, csv2points, df2nrml, points2nrml

# %% definitions


def display_and_save(name, df):
    print('\n%s:' % name)
    display(df)
    df.to_csv(name.replace(' ', '_') + '.csv')


# %% constants

LAYERS_DF = pd.read_csv('layers.csv', index_col='layerid')
USE_RECOMPUTED = False

model_path = '../Data/nath2012probabilistic'
if USE_RECOMPUTED:
    smoothed_model_path = '../Smoothed/Recomputed'
else:
    smoothed_model_path = model_path

smoothed_source_data_file = 'smoothed_source_model'


# %% inputs

areal_source_model_csv = 'areal_source_model.csv'
min_mags = [4.5, 5.5]
smoothed_source_models_csv = [
    'nath2012probabilistic_smoothed_source_model_mmin_%g.csv' % min_mag
    for min_mag in min_mags]
source_tree_tsv = '../Logic Trees/areal_model_logic_tree.tsv'


# %% areal zone data

areal_df = csv2areal(areal_source_model_csv)

display_drop = [column for column in areal_df if column in
                ['polygon', 'geometry', 'aspect ratio', 'centroid',
                 'source_name', 'id', 'concerns', 'mmin']]
display_drop += [column for column in areal_df if column.endswith('2')]

display(pd.concat([areal_df.drop(display_drop, axis=1).head(),
                   areal_df.drop(display_drop, axis=1).tail()]))

# %% read logic tree description table

print('Logic tree before collapse:')
source_tree_symbolic_df = read_tree_tsv(source_tree_tsv)
display(source_tree_symbolic_df)

# %% compute collapsed rates

areal_collapsed_df, reduced_df, all_weights, labels = \
    collapse_sources(areal_df, source_tree_symbolic_df)

print('Logic tree after collapse:')
display(reduced_df)

# %% plot collapsed rates by zone

fig, axes = plt.subplots(len(LAYERS_DF.index), 1,
                         figsize=(8, 4*len(LAYERS_DF)),  sharex=True)
fig.subplots_adjust(hspace=0)
for layer_id, ax in zip(LAYERS_DF.index, axes):
    annotate('layer %d' % layer_id, ax=ax)
    layer_df = areal_collapsed_df[areal_collapsed_df['layerid'] == layer_id]
    for zoneid, zone in layer_df.iterrows():
        num_bins = zone['occurRates'].size
        mags = zone['mmin'] + zone['magBin']*np.arange(num_bins + 1)
        rates = zone['occurRates'][[0] + list(range(num_bins))]
        ax.step(mags, rates, label=zoneid)
    ax.set_yscale('log')
    ax.set_ylabel('Annual Occurence Rate')
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
              frameon=False, labelspacing=0, ncol=3)

axes[-1].set_xlabel('Moment Magnitude, $M_w$')
fig.savefig('MeanOccurrenceRatesAllZones.pdf',
            transparent=True, bbox_inches='tight', pad_inches=0.1)

# %% investigate zones

zoneids = [119, 912]
# zoneids = [1, 924, 93, 137, 913, 915, 132]
zones_df = areal_collapsed_df.loc[zoneids]

fig, axes = plt.subplots(len(zoneids), 1, figsize=(4, 3*len(zoneids)),
                         sharex=True)
fig.subplots_adjust(hspace=0)
for (zoneid, zone), ax in zip(zones_df.iterrows(), axes):
    annotate(r'zone %s: $b = %g \pm %g$, $m_{max} = %g \pm %g$' %
             (zoneid, zone['b'], zone['stdb'], zone['mmax'], zone['stdmmax']),
             ax=ax)
    zone_rates = zone.all_rates
    num_bins = zone.all_rates.shape[0]
    mags = zone['mmin'] + zone['magBin']*np.arange(zone.all_rates.shape[0] + 1)
    repeat_first = [0] + list(range(zone.all_rates.shape[0]))
    rates = zone.all_rates[repeat_first, :]
    for rate, label in zip(rates.T, labels):
        ax.step(mags, rate, label=label)
    ax.step(mags, zone['occurRates'][repeat_first],
            linewidth=2, color='black', label='collapsed')
    ax.set_yscale('log')
    ax.set_ylabel('Annual Occurence Rate')

axes[0].legend(bbox_to_anchor=(1, 0.5), loc='center left',
               frameon=False, labelspacing=0, fontsize='small')
axes[-1].set_xlabel('Moment Magnitude, $M_w$')

file_name = ('Mean_Occurrence_Rates_Zones_%s.pdf' %
             '_'.join([str(zoneid) for zoneid in zoneids]))
fig.savefig(file_name, transparent=True, bbox_inches='tight', pad_inches=0.1)

# %% summarize

summary_columns = ['layerid', 'a', 'b', 'stdb', 'mmax', 'stdmmax',
                   'new style', 'tectonic subregion', 'region']

display_and_save('Selected zones of interest',
                 zones_df[summary_columns])

display_and_save('Top 3 b-value uncertainty',
                 areal_collapsed_df.sort_values('stdb', ascending=False)
                 [summary_columns].head(3))

display_and_save('Top 3 mmax uncertainty',
                 areal_collapsed_df.sort_values('stdmmax', ascending=False)
                 [summary_columns].head(3))

display_and_save('Top 3 mmax',
                 areal_collapsed_df.sort_values('mmax', ascending=False)
                 [summary_columns].head(3))

display_and_save('Bottom 3 mmax',
                 areal_collapsed_df.sort_values('mmax', ascending=True)
                 [summary_columns].head(3))

display_and_save('Top 3 a-value',
                 areal_collapsed_df.sort_values('a', ascending=False)
                 [summary_columns].head(3))

display_and_save('Bottom 3 a-value',
                 areal_collapsed_df.sort_values('a', ascending=True)
                 [summary_columns].head(3))

display_and_save('Top 3 b-value',
                 areal_collapsed_df.sort_values('b', ascending=False)
                 [summary_columns].head(3))

display_and_save('Bottom 3 b-value',
                 areal_collapsed_df.sort_values('b', ascending=True)
                 [summary_columns].head(3))

# %% write areal sources to NRML

df2nrml(areal_collapsed_df, 'areal_collapsed.xml')

# %% write point sources to NRML

model_basename = ' '.join((os.path.split(smoothed_model_path)[1],
                           smoothed_source_data_file)) + ' thinned'
smoothed_df = csv2points(model_basename)

smoothed_collapsed_df, reduced_df, all_weights, labels = \
    collapse_sources(smoothed_df, source_tree_symbolic_df)

model_basename += ' collapsed'
points2nrml(smoothed_collapsed_df, model_basename)
