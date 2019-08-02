#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Given a Canadian National Earthquake Database pick file (NPF) name, produces
a validated QuakeML file.

By default the output_file name is the input_file name with '.xml' appended,
but in the current working directory.

This is simply a wrapper around catalogue_tools.npf2cat.npf2cat().
'''
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator

import osr
import gdal

from toolbox import convert_probability, logspace, read_hazard_map_csv

FILE_NAME = os.path.basename(__file__)
IMT_POE_FMT = '%s, %.3g%%/%gy'


def _label(i, edges):
    if i == 0:
        return '<= %g' % edges[i]
    if i == len(edges) - 1:
        return '> %g' % edges[i - 1]
    return '%g - %g' % (edges[i - 1], edges[i])


def csv2raster(input_file, raster_fmt='GTiff', crs='EPSG:4326',
               investigation_time=50, limits=[0.005, 5], bins_per_decade=6,
               plot_fmt='png', colormap='jet', axsize=[3, 3], symmetric=False,
               sites_table='../Data/nath2012probabilistic/Table 3.csv'):
    '''
    Arguments
    ---------
    input_file: str
        NPF input file name
    raster_fmt: str
        raster format supported by GDAL
    investigation_time: bool
        time period of interest (nominal)
    encoding: str
        input file character encoding
    raise_errors: bool
        ... instead of logging them, for troubleshooting
    one_solution_per_event: bool
        linefeed and/or dummy line can be omitted between events
    validate: bool
        validate QuakeML produced against schema
    log_level: str
        console logging level (verbosity)

    Returns
    -------
    str:
        QuakeML output file name
    '''
    df_all, config = read_hazard_map_csv(input_file)
    t_inv = config['investigation_time']

    imts = df_all.columns.levels[0].values
    poes_inv = df_all.columns.levels[1].values
    poes = convert_probability(poes_inv, t_inv, investigation_time)

    lons = np.array(sorted(df_all.index.levels[0].values))
    lats = np.array(sorted(df_all.index.levels[1].values))

    grid_step = np.mean(np.diff(lons))

    srs = osr.SpatialReference()
    srs.SetFromUserInput(crs)
    projection_wkt = srs.ExportToWkt()

    if raster_fmt.lower() == 'gtiff':
        raster_ext = 'tif'
    else:
        raster_ext = raster_fmt

    raster_file = os.path.splitext(input_file)[0] + '.' + raster_ext
    print('Saving: ' + raster_file)

    ds = gdal.GetDriverByName(raster_fmt).Create(
        raster_file, len(lons), len(lats), df_all.shape[1], gdal.GDT_Float64)
    ds.SetProjection(projection_wkt)
    ds.SetGeoTransform([lons.min(), grid_step, 0, lats.min(), 0, grid_step])

    band_number = 1
    for imt in imts:
        for poe_inv, poe in zip(poes_inv, poes):
            df = df_all[(imt, poe_inv)].unstack().T
            band = ds.GetRasterBand(band_number)
            band.WriteArray(df.values)
            band.SetDescription(IMT_POE_FMT % (imt, 100*poe,
                                               investigation_time))
            band_number += 1

    ds.FlushCache()

    if not plot_fmt:
        return raster_file

    image_file = os.path.splitext(input_file)[0] + '.' + plot_fmt
    print('Saving: ' + image_file)

    if sites_table:
        df_cities = pd.read_csv(sites_table, skiprows=1, index_col='City')

    x = np.hstack((lons - grid_step/2, lons[-1] + grid_step/2))
    y = np.hstack((lats - grid_step/2, lats[-1] + grid_step/2))

    if not symmetric:
        boundaries = logspace(limits[0], limits[1], bins_per_decade)
    else:
        boundaries = logspace(limits[0], limits[1], 2*bins_per_decade)
        boundaries = np.array(sorted(
            set(logspace(limits[0], limits[1], 2*bins_per_decade)) -
            set(logspace(limits[0], limits[1], bins_per_decade))))

    cmap = plt.cm.get_cmap(colormap, len(boundaries) - 1)
    cmap.set_over('0.7')
    cmap.set_under('0.3')
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    fig, axes = plt.subplots(
        len(imts), len(poes_inv), sharex=True, sharey=True,
        figsize=(axsize[0]*len(poes_inv), axsize[1]*len(imts)),
        subplot_kw=dict(aspect='equal', adjustable='box'))

    for imt, row_axes in zip(imts, axes):
        for poe_inv, poe, ax in zip(poes_inv, poes, row_axes):

            ax.annotate(IMT_POE_FMT % (imt, 100*poe, investigation_time),
                        xy=(0.05, 0.05), xycoords='axes fraction')

            df = df_all[(imt, poe_inv)].unstack().T
            im = ax.pcolormesh(x, y, df.values, cmap=cmap, norm=norm)

            if sites_table:
                ax.plot(df_cities['Longitude (°E)'],
                        df_cities['Latitude (°N)'],
                        'ko', markersize=2)

    fig.subplots_adjust(hspace=0, wspace=0)

    if sites_table:
        for city, info in df_cities.iterrows():
            axes[0, 0].annotate(
                city, (info['Longitude (°E)'], info['Latitude (°N)']),
                va='bottom', fontsize=6)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5,
                        label='Acceleration [g]', extend='both')
    cbar.set_ticks(boundaries)
    cbar.ax.set_yticklabels(['%g' % item for item in boundaries])

    for ax in axes[-1, :]:
        ax.set_xlabel(u'Longitude [°]')
        ax.xaxis.set_major_locator(MultipleLocator(base=10.))
    for ax in axes[:, 0]:
        ax.set_ylabel(u'Latitude [°]')
        ax.yaxis.set_major_locator(MultipleLocator(base=10.))
    for ax in axes[:-1, :].flatten():
        ax.set_xticks([])
    for ax in axes[:, 1:].flatten():
        ax.set_yticks([])

    fig.savefig(image_file, dpi=300, bbox_inches='tight')

    colormap_file = 'colormap_%s_%g-%g.txt' % (colormap, boundaries[0],
                                               boundaries[-1])
    print('Saving: ' + colormap_file)

    cmap_df = pd.DataFrame(
        np.vstack((
            cmap._rgba_under,
            np.vstack(cmap(i) for i in range(cmap.N)),
            cmap._rgba_over)),
        columns=['R', 'G', 'B', 'A'],
        index=np.arange(cmap.N + 2))
    cmap_df = (256*cmap_df).astype(int)
    # cmap_df.drop(cmap_df.index[-1], inplace=True)
    # cmap_df.drop(cmap_df.index[-1], inplace=True)
    cmap_df.insert(0, 'upper', np.hstack((boundaries, np.Inf)))
    cmap_df['label'] = [_label(i, cmap_df['upper']) for i in cmap_df.index]

    if os.path.isfile(colormap_file):
        os.remove(colormap_file)
    with open(colormap_file, 'w') as file:
        file.write('# %s Generated Color Map Export File\n' % colormap)
        file.write('INTERPOLATION:DISCRETE\n')
        cmap_df.to_csv(file, index=False, header=False, float_format='%g')

    return raster_file


def _argparser():
    '''
    Command-line arguments for main
    '''
    parser = ArgumentParser(prog=os.path.splitext(FILE_NAME)[0],
                            description=__doc__)
    parser.add_argument(
        'input_file',
        help='NPF input file name ')
    parser.add_argument(
        '-r', '--raster_fmt', default='GTiff',
        help='raster file format (see GDAL documentation)')
    parser.add_argument(
        '-c', '--crs', default='EPSG:4326',
        help='output coordinate reference system')
    parser.add_argument(
        '-i', '--investigation_time', type=float, default=50,
        help='nominal investigation time')
    parser.add_argument(
        '-l', '--limits', type=float, nargs=2, default=[0.005, 5],
        help='nominal investigation time')
    parser.add_argument(
        '-a', '--axsize', type=float, nargs=2, default=[3, 3],
        help='target width x height of a subplot axes [inches]')
    parser.add_argument(
        '-b', '--bins_per_decade', type=int, default=6,
        help='number of bins per decade, for plotting')
    parser.add_argument(
        '--colormap', default='jet',
        help='colormap for plotting')
    parser.add_argument(
        '-p', '--plot_fmt', default='png',
        help='image file format (see matplotlib.pyplot.Figure.savefig())')
    parser.add_argument(
        '-s', '--symmetric', action='store_true',
        help='force bins to be symmetric around unity.')
    parser.add_argument(
        '--sites_table',
        default='../Data/nath2012probabilistic/Table 3.csv',
        help='tables of sites to plot')
    return parser


def main(argv=()):
    '''
    Run command line prog to generate QuakeML.

    Returns zero for successful termination, one otherwise.
    '''
    parser = _argparser()
    if len(argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args(argv[1:])
    kwargs = vars(args)
    input_file = kwargs.pop('input_file')

    output_file = csv2raster(input_file, **kwargs)

    return len(output_file) == 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
