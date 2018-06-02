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
'''
Helper functions for OpenQuake source modeling.

Most of these functions are utilities specific to Nath & Thingbaijam (2012).
'''
# pylint: disable=superfluous-parens
import os
import re
from io import StringIO
from copy import deepcopy
from numbers import Number
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

import numpy as np
import pandas as pd
from shapely.wkt import loads, dumps
import geopandas as gpd
from natsort import natsorted, order_by_index, index_natsorted

from obspy.imaging.beachball import aux_plane

from openquake.hazardlib import geo, mfd, pmf

from openquake.hmtk.sources.area_source import mtkAreaSource
from openquake.hmtk.sources.point_source import mtkPointSource
from openquake.hmtk.sources.source_model import mtkSourceModel

from openquake.hmtk.seismicity.completeness.comp_stepp_1971 import Stepp1971

from openquake.hmtk.plotting.mapping import HMTKBaseMap
from openquake.hmtk.plotting.seismicity.completeness import plot_stepp_1972
from openquake.hmtk.plotting.seismicity.catalogue_plots \
    import plot_magnitude_time_density


from toolbox import wrap, annotate

SEISMICITY_ALIASES = {
    'avalue': 'a',
    'bvalue': 'b',
    'stdbvalue': 'stdb'}

COORDINATE_ALIASES = {
    'Latitude': 'latitude',
    'lat': 'latitude',
    'Lat': 'latitude',
    'Longitude': 'longitude',
    'Long': 'longitude',
    'Lon': 'longitude',
    'long': 'longitude',
    'lon': 'longitude',
    }

ALL_REQUIRED = ['id', 'source_name', 'zmin', 'zmax', 'tectonic subregion',
                'msr', 'strike', 'dip', 'rake', 'aspect ratio']

COORDINATES = ['longitude', 'latitude']
AREA_REQUIRED = ['geometry']

GR_REQUIRED = ['mmin', 'mmax', 'a', 'b']
DISCRETE_REQUIRED = ['mmin', 'occurRates', 'magBin']


class MyPolygon(geo.polygon.Polygon):
    # pylint: disable=no-member,no-init,too-few-public-methods
    '''
    Enables fast caclulation of point-to-polygon distances,
    based on openquake.hazardlib.geo.
    '''

    def distances(self, mesh):
        '''
        Compute distances to each point of mesh.

        Mesh coordinate values and results are in decimal degrees.

        :param mesh:
            :class:`openquake.hazardlib.geo.mesh.Mesh` instance.
        :returns:
            Numpy array of `float` values in the same shapes in the input
            coordinate arrays consisting of the distance to each point in
            those arrays. Points inside or on edge of polygon return zero.
        '''
        self._init_polygon2d()
        pxx, pyy = self._projection(mesh.lons, mesh.lats)
        return geo.utils.point_to_polygon_distance(self._polygon2d, pxx, pyy)


def read_polygons(file_name, rename=(('polygon coordinates', 'polygon'),)):
    """
    Read polygon descriptions from text file into pandas.DataFrame.

    File format as per Nath & Thingbaijam (2012):

    zoneid,[polygon coordinates]
    1,[67.81,37.55; 68.30,38.82; 72.95,39.99; 73.00,39.06; 70.69,38.27; ]
    2,[72.95,39.99; 76.53,39.99; 76.10,38.89; 75.85,37.89; 75.85,37.89; 74.94,38.35; 73.00,39.06; ]
    """  # noqa
    with open(file_name) as file:
        line = file.readline()
        columns = [item.strip('[]') for item in line.strip().split(',')]

        rows = []
        for line in file.readlines():
            fields = line.split(',', len(columns) - 1)
            series = pd.Series()
            for column, field in zip(columns, fields):
                if 'id' in column:
                    value = int(field)
                elif 'polygon' in column:
                    # TODO: create shapely Polygon directly
                    value = MyPolygon([
                        geo.point.Point(float(row.split(',')[0]),
                                        float(row.split(',')[1]))
                        for row in field.strip().strip('[]').split(';')
                        if row.strip()])
                else:
                    raise ValueError('Unrecognized column: ' + column)

                series[column] = value

            rows.append(series)

    df = pd.DataFrame(rows).rename(columns=dict(rename))

    df['geometry'] = [loads(polygon.wkt) for polygon in df['polygon']]

    return df


def add_name_id(df):
    '''
    Add columns with short names and ids appropriate for NRML source models.
    '''
    source_class = get_source_class(df)
    df = df.copy()

    if source_class is mtkAreaSource:
        df['source_name'] = df.apply(_areal_source_name, axis=1)
        df['id'] = df.apply(_areal_source_id, axis=1)

    elif source_class is mtkPointSource:
        df['source_name'] = df.apply(_point_source_name, axis=1)
        df['id'] = df.apply(_point_source_id, axis=1)

    else:
        raise ValueError(
            'Source class %s not supported' % source_class.__name__)

    return df


def natural_sort(df, by='id', index=False):
    '''
    Sort a pandas dataframe "naturally" by column or by index.
    '''
    if index:
        return df.reindex(index=natsorted(df.index))
    else:
        return df.reindex(index=order_by_index(df.index,
                                               index_natsorted(df['id'])))


def make_source(series, source_class, mag_bin_width=0.1):
    '''
    Make a source from a pandas Series.
    '''
    if source_class is mtkPointSource:
        geometry = geo.point.Point(series.longitude, series.latitude)

    elif source_class is mtkAreaSource:
        if isinstance(series.geometry, str):
            series.geometry = loads(series.geometry)
        coords = list(zip(*series.geometry.exterior.coords.xy))
        points = [geo.point.Point(lon, lat) for lon, lat in coords]
        geometry = geo.polygon.Polygon(points + [points[0]])

    else:
        raise ValueError('Source class %s not supported' %
                         source_class.__name__)

    if 'occurRates' in series:
        mag_freq_dist = mfd.EvenlyDiscretizedMFD(
            series.mmin + series.magBin/2, series.magBin,
            series.occurRates.tolist())
    else:
        mag_freq_dist = mfd.TruncatedGRMFD(
            series.mmin, series.mmax, mag_bin_width, series.a, series.b)

    nodal_plane_pmf = pmf.PMF(
        [(1.0, geo.NodalPlane(series.strike, series.dip, series.rake))])

    hypo_depth_pmf = pmf.PMF(
        [(1.0, (series.zmin + series.zmax)/2.0)])

    return source_class(
        series.id,
        series.source_name,
        geometry=geometry,
        trt=series['tectonic subregion'],
        upper_depth=series.zmin,
        lower_depth=series.zmax,
        rupt_aspect_ratio=series['aspect ratio'],
        mag_scale_rel=series.msr,
        mfd=mag_freq_dist,
        nodal_plane_dist=nodal_plane_pmf,
        hypo_depth_dist=hypo_depth_pmf)


def source_df_to_list(df):
    '''
    Converts source description table into list of source objects.

    The result list can then be serialized to NRML using
    `openquake.hmtk.sources.source_model.mtkSourceModel`

    Arguments
    ---------
    df: :class:`pandas.DataFrame`
        table of descriptions of sources

    Returns
    -------
    sources: list
        list of e.g. :class:`openquake.hmtk.sources.area_source.mtkAreaSource`
    '''  # noqa

    df = df[(df['a'] != 0) &
            (df['mmax'] != 0) &
            (df['dip'] != -1) &
            ~pd.isnull(df['dip'])]

    source_class = get_source_class(df)

    return df.apply(lambda series: make_source(series, source_class),
                    axis=1).tolist()


def get_source_class(df):
    '''
    Determine source class based on presence of geometry vs. point coordinates.
    '''
    if all(key in df.columns for key in COORDINATES):
        source_class = mtkPointSource
    elif all(key in df.columns for key in AREA_REQUIRED):
        source_class = mtkAreaSource
    else:
        raise ValueError(
            'Only area [%s] and point [%s] sources currently supported' %
            (', '.join(COORDINATES), ', '.join(AREA_REQUIRED)))

    return source_class


def _areal_source_name(series):
    return 'zone %s' % series.name


def _areal_source_id(series):
    return 'z%s' % series.name


def _point_source_name(series):
    return '%gN %gE %g-%g km depth M%g' % (series.latitude, series.longitude,
                                           series.zmin, series.zmax,
                                           series.mmin)


def _point_source_id(series):
    result = '%gN_%gE_L%d_M%.1f' % (series.latitude, series.longitude,
                                    series.layerid, series.mmin)
    # For the source IDs OpenQuake only accepts a-zA-z0-9_-
    return result.replace('.', 'p')


def _check_columns(df):
    missing = [item for item in ALL_REQUIRED if item not in df.columns]
    if missing:
        raise ValueError(
            'Missing required columns: ' + ', '.join(missing))
    missing_point = [item for item in COORDINATES if item not in df.columns]
    missing_area = [item for item in AREA_REQUIRED if item not in df.columns]
    if missing_point and missing_area:
        raise ValueError(
            'Missing geometry: %s or %s' %
            (', '.join(missing_point), ', '.join(missing_area)))
    missing_gr = [item for item in GR_REQUIRED if item not in df.columns]
    missing_discrete = [item for item in DISCRETE_REQUIRED
                        if item not in df.columns]
    if missing_gr and missing_discrete:
        raise ValueError(
            'Missing FMD: %s or %s' %
            (', '.join(missing_gr), ', '.join(missing_discrete)))


def df2nrml(df, model_name):
    '''
    Write pandas DataFrame of source models to NRML. Sources are twinned by
    magnitude to support alternative tectonic region types for large-magnitude
    events.
    '''
    if model_name.endswith('.xml'):
        model_name = model_name[:-4]
    nrml_file = model_name.replace(' ', '_') + '.xml'

    df = add_name_id(df)
    df = twin_source_by_magnitude(df)
    _check_columns(df)
    df = natural_sort(df, by='id')

    source_list = source_df_to_list(df)
    source_model = mtkSourceModel(identifier='1',
                                  name=model_name,
                                  sources=source_list)

    print('Writing: %s' % os.path.abspath(nrml_file))
    source_model.serialise_to_nrml(nrml_file)

    return source_model


def points2nrml(df, base_name, by=['mmin model'], fmt='mmin%g'):
    '''
    Write multiple pandas DataFrame of point source models to NRML.
    '''
    df.sort_values(['mmin model', 'layerid'] + COORDINATES, inplace=True)

    for index, group_df in df.groupby(by):
        model_name = base_name + ' ' + fmt % index
        df2nrml(group_df, model_name)


def points2csv(df, base_name, by=['mmin model', 'layerid'],
               fmt='mmin%g layer%d'):
    '''
    Write grouped data with added names, ids and binwise rates.
    '''
    if base_name.endswith('.csv'):
        base_name = base_name[:-4]

    # TODO: select columns of interest, or at least control column order?
    df.drop('geometry', axis=1, inplace=True)

    df = add_name_id(df)
    df = add_binwise_rates(df)
    _check_columns(df)
    df = natural_sort(df, by='id')

    df.sort_values(by + COORDINATES, inplace=True)

    for index, group_df in df.groupby(by):
        model_name = base_name + ' ' + fmt % index
        csv_file = model_name.replace(' ', '_') + '.csv'
        print('Writing: ' + os.path.abspath(csv_file))
        group_df.drop(columns=by).to_csv(csv_file, index=False,
                                         float_format='%.5g')


def csv2points(base_name, by=['mmin model', 'layerid'],
               fmt='mmin%g layer%d', ranges=((4.5, 5.5), (1, 2, 3, 4))):
    '''
    Write grouped data with added names, ids and binwise rates.
    '''
    dfs = []
    for values in product(*ranges):
        model_name = base_name + ' ' + fmt % values
        csv_file = model_name.replace(' ', '_') + '.csv'
        print('Reading: ' + os.path.abspath(csv_file))
        df = pd.read_csv(csv_file)
        for column, value in zip(by, values):
            df[column] = value
        dfs.append(df)

    df = pd.concat(dfs)

    _check_columns(df)

    if 'geometry' in df:
        df['geometry'] = df.geometry.apply(loads)

    df.sort_values(by + COORDINATES, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def areal2csv(df, model_name):
    '''
    Write areal model with names, ids and geometry.
    '''
    if model_name.endswith('.csv'):
        model_name = model_name[:-4]
    csv_file = model_name.replace(' ', '_') + '.csv'

    df = df.drop(columns=['polygon'])

    if not csv_file.endswith('.csv'):
        csv_file += '.csv'
    print('Writing: ' + os.path.abspath(csv_file))

    df['centroid'] = df.geometry.apply(  # facilitates plotting symbols in QGIS
        lambda polygon: dumps(polygon.centroid, rounding_precision=2))
    df['geometry'] = df.geometry.apply(
        lambda polygon: dumps(polygon, rounding_precision=2))
    df = add_name_id(df)
    _check_columns(df)

    df.to_csv(csv_file)


def csv2areal(csv_file):
    '''
    read areal model, returning a geopandas DataFrame.
    '''
    if not csv_file.endswith('.csv'):
        csv_file += '.csv'
    print('Reading: ' + os.path.abspath(csv_file))
    df = pd.read_csv(csv_file, index_col='zoneid')
    df['geometry'] = df['geometry'].apply(loads)
    df = gpd.GeoDataFrame(df, crs='WGS84')
    _check_columns(df)

    return df


def csv2df(csv_file):
    if 'areal' in csv_file:
        return csv2areal(csv_file)
    elif 'smoothed' in csv_file:
        return csv2points(csv_file)
    else:
        raise RuntimeError('Not sure what kind of source file this is:' +
                           csv_file)


def focal_mech(dip, rake, threshold=30):
    '''
    Associate a focal mechanism with a dip & rake. This is based on the
    formulae used in GMPEs to infer a style of faulting from focal plane angles
    but is flawed because it takes no notice if the auxiliary plane is actually
    more physically plausible.

    Note that different GMPEs use different threshold values, e.g.:
    30°: Boore and Atkinson (2008), Campbell and Bozorgnia (2008)
    45°: Zhao et al. (2006)
    '''
    if isinstance(dip, Number) and isinstance(rake, Number):
        rake = wrap(rake)
        if 0 <= wrap(dip) <= 90:
            if threshold < rake < 180 - threshold:
                return 'reverse'  # dip-slip
            elif threshold < -rake < 180 - threshold:
                return 'normal'  # dip-slip
            elif rake < threshold:
                return 'sinistral'  # strike-slip

            return 'dextral'  # strike-slip

        return 'undefined'

    return [focal_mech(d, r) for d, r in zip(dip, rake)]


FAULTING_STYLES = pd.read_fwf(StringIO('''\
faulting style dip rake
sinistral      90  0
dextral        90  180
dextral        90  -180
reverse        45  90
normal         45  -90
'''))
CANONICAL_PLANES = FAULTING_STYLES[['dip', 'rake']].as_matrix()
DIP_RAKE_CANONICAL = CANONICAL_PLANES.copy()


def faulting_style(strike, dip, rake):
    '''
    Assign most physically plausible faulting style given the angles
    defining one of the nodal planes.

    Method: find euclidian distances from both nodal planes to the planes of
    each of the canonical faulting styles, and choose the minimum.

    Returns
    =======
    faulting_style: str
        most physically plausible faulting style
    strike, dip, rake: tuple of float
        angles defining most physically plausible fault plane
    '''
    try:
        rake = wrap(rake)

        if 0 <= wrap(dip) <= 90:
            candidates = [
                focal_mech(dip, rake),
                focal_mech(*(aux_plane(strike, dip, rake)[1:]))
                ]

            return next(
                (faulting_style for faulting_style in candidates
                 if faulting_style in ['normal', 'reverse']),
                'strike-slip')

        return 'undefined'

    except ValueError:
        return [faulting_style(s, d, r) for s, d, r in zip(strike, dip, rake)]


def twin_source_by_magnitude(df, column='tectonic subregion',
                             select_type='subduction interface',
                             type_suffix=' megathrust',
                             mag_thresh=7.5, twin_zone_suffix='m'):
    '''
    Twin selected sources by magnitude.

    Assumes an 'id' has been generated for each source e.g. using
    add_name_id(). The twin_zone_suffix is appended to this id for the new
    zones.

    The point here is to take a source and make two sources out of it, with
    the new source given a new tectonic region type.
    '''

    oq_valid_id = '[A-Za-z0-9_]*$'
    assert re.match(oq_valid_id, twin_zone_suffix), \
        '"%s" does not match "%s"' % (twin_zone_suffix, oq_valid_id)

    df = df.copy()

    new_type = select_type + type_suffix
    # don't re-twin
    if any(df[column] == new_type):
        return df

    # create twinned sources
    indices = ((df[column] == select_type) &
               (df['mmax'] - df['stdmmax'] > mag_thresh))
    twinned_df = df[indices].copy()
    twin_start_index = int(10**(np.ceil(np.log10(twinned_df.index.max()))))
    twinned_df.index += twin_start_index + twinned_df.index
    df.loc[indices, 'mmax'] = mag_thresh
    twinned_df['mmin'] = mag_thresh
    twinned_df['id'] += twin_zone_suffix
    twinned_df[column] = new_type

    # prune bins above/below maximum magnitude
    if 'occurRates' in df.columns:
        above_rates, below_rates = [], []
        for _, zone in df.loc[indices].iterrows():

            num_bins = zone['occurRates'].size
            mags = zone['mmin'] + zone['magBin']*(np.arange(num_bins) + 0.5)
            above_rates += [zone['occurRates'][mags > mag_thresh]]
            below_rates += [zone['occurRates'][mags < mag_thresh]]

        twinned_df['occurRates'] = above_rates
        for zone, rates in zip(df.index[indices], below_rates):
            df.at[zone, 'occurRates'] = rates

    df = pd.concat([df, twinned_df])

    return df


def add_binwise_rates(df, mag_start=5, mag_stop=8, mag_step=1):
    '''
    Add binwise sesismicity rates for comparison
    '''
    for mag in range(mag_start, mag_stop, mag_step):
        log_n_m_lo = df['a'] - df['b']*np.maximum(df['mmin'], mag)
        log_n_m_hi = df['b'] - df['b']*np.minimum(df['mmax'], mag + 1)

        with np.errstate(invalid='ignore', divide='ignore'):
            df['logN_%.1f-%.1f' % (mag, mag + 1)] = \
                np.log10(10**log_n_m_lo - 10**log_n_m_hi).round(2)

    return df


def extract_param(df, param, x='longitude', y='latitude'):
    '''
    Extract a matrix of parameter grid values from a dataframe, for plotting.
    '''
    pivot_df = df.pivot_table(index=y, columns=x, values=param)
    ordinate = pivot_df.columns.values
    abscissa = pivot_df.index.values
    data = pivot_df.as_matrix()
    return data, ordinate, abscissa


def plot_mag_time_density_slices(
        catalogue, completeness_tables, slice_key, slice_ids,
        mag_bin=0.1, time_bin=1):
    """
    Magnitude-time density plots on sub-catalogues, where `slice_key` and
    `slice_ids` determine how the sub-catalouges are formed.
    """

    fig, axes = plt.subplots(len(slice_ids), 1,
                             figsize=(8, 2*len(slice_ids)), sharex=True)
    fig.subplots_adjust(hspace=0)
    for ax, slice_id, completeness_tables_slice \
            in zip(axes, slice_ids, completeness_tables):

        annotate('%s %d' % (slice_key, slice_id), loc='upper left', ax=ax)

        catalogue_slice = deepcopy(catalogue)
        in_slice = catalogue_slice.data[slice_key] == slice_id
        catalogue_slice.select_catalogue_events(in_slice)

        plot_magnitude_time_density(
            catalogue_slice, mag_bin, time_bin,
            completeness=completeness_tables_slice, ax=ax)

    return fig


def plot_completeness_slices(catalogue, slice_key, slice_ids,
                             mag_bin=0.5, time_bin=1.,
                             deduplicate=True, mag_range=(4., None),
                             year_range=None):
    """
    Stepp (1971) analysis on sub-catalogues, where `slice_key` and
    `slice_ids` determine how the sub-catalouges are formed.
    """
    comp_config = {'magnitude_bin': mag_bin,
                   'time_bin': time_bin,
                   'increment_lock': True}

    fig, axes = plt.subplots(len(slice_ids), 1,
                             figsize=(6, 2*len(slice_ids)), sharex=True)
    fig.subplots_adjust(hspace=0)

    slice_completeness_tables = []
    for ax, slice_id in zip(axes, slice_ids):

        catalogue_slice = deepcopy(catalogue)
        in_slice = catalogue_slice.data[slice_key] == slice_id
        catalogue_slice.select_catalogue_events(in_slice)

        model = Stepp1971()
        model.completeness(catalogue_slice, comp_config)
        model.simplify(deduplicate, mag_range, year_range)
        slice_completeness_tables.append(model.completeness_table.tolist())

        annotate('%s %d' % (slice_key, slice_id), loc='upper left', ax=ax)
        plot_stepp_1972.create_stepp_plot(model, ax=ax)

    return fig, slice_completeness_tables


def plot_depth_distance(catalogue, coordinate_limits, ordinate, name=None,
                        colour='black', size=4, ax=None):
    """
    Produces a "side-view" of a portion of a catalogue. Subcatalogue selection
    is currently a simple rectangle of latitudes and longitudes. Ordinates
    supported are currently 'latitude' or 'longitude'.

    :param catalogue: instance of :class:`hmtk.seismicity.catalogue.Catalogue`
    :param tuple coordinate_limits: lat_min, lat_max, lon_min, lon_max
    :param string ordinate: distance to plot on x-axis
    """

    assert ordinate in ['latitude', 'longitude']

    subcatalogue = deepcopy(catalogue)

    lat_min, lat_max, lon_min, lon_max = coordinate_limits
    subcatalogue.purge_catalogue(
        (subcatalogue.data['latitude'] >= coordinate_limits[0]) &
        (subcatalogue.data['latitude'] <= coordinate_limits[1]) &
        (subcatalogue.data['longitude'] >= coordinate_limits[2]) &
        (subcatalogue.data['longitude'] <= coordinate_limits[3]))

    if colour == 'magnitude':
        colour = subcatalogue.data['magnitude']
        cmap = 'jet'
    else:
        cmap = 'none'

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(subcatalogue.data[ordinate], subcatalogue.data['depth'],
               c=colour, s=size, cmap=cmap, edgecolor='none')

    if ordinate == 'latitude':
        ax_label = u'Longitude: %g°-%g°' % (lon_min, lon_max)
        ax.set_xlabel(u'Latitude (°)')
        ax.set_xlim(lat_min, lat_max)
    else:
        ax_label = u'Latitude: %g°-%g°' % (lat_min, lat_max)
        ax.set_xlabel(u'Longitude (°)')
        ax.set_xlim(lon_min, lon_max)
    if name is not None:
        ax_label = name + '\n' + ax_label
    annotate(ax_label, loc='lower left', frameon=False, ax=ax)


def locations(fig, axes, func, attribute):
    '''
    Helper for selecting axes among subplots.

    Example
    -------
    >>> fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
    >>> for i, ax in enumerate(locations(fig, axes[:2, :2], max, 'ymax')):
    >>>     ax.set_title(str(i))
    '''
    pos = func(getattr(ax.get_position(), attribute) for ax in fig.axes)
    return (ax for ax in axes.ravel()
            if getattr(ax.get_position(), attribute) == pos)


def plot_smoothed_maps(file_template, layer_ids, min_mags, grid_step,
                       coordinate_limits, value_limits, axes=None):
    '''
    Plot smoothed seismicity models by layer and minimum magnitude.

    Data is projected over an HMTKBaseMap.
    '''
    map_config = {
        'min_lon': coordinate_limits[0],
        'max_lon': coordinate_limits[1],
        'min_lat': coordinate_limits[2],
        'max_lat': coordinate_limits[3],
        'parallel_meridian_spacing': 10,
        'resolution': 'l',
        }

    if axes is None:
        fig, axes = plt.subplots(len(layer_ids), len(min_mags),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0.3)

    for i, layer_id in enumerate(layer_ids):
        for j, min_mag in enumerate(min_mags):
            annotate('Layer %d: M > %g' % (layer_id, min_mag),
                     'lower left', ax=axes[i, j])

            small_csv = file_template % (min_mag, layer_id)

            df = pd.read_csv(small_csv)
            df.rename(columns=COORDINATE_ALIASES, inplace=True)
            basemap = HMTKBaseMap(map_config, ax=axes[i, j], lat_lon_spacing=5)
            basemap.add_colour_scaled_points(
                df['longitude'].values, df['latitude'].values,
                df['Smoothed Rate'].values, size=(grid_step/0.2)**2, alpha=1,
                norm=LogNorm(*value_limits), overlay=True)


def plot_smoothed(file_template, layer_ids, min_mags, grid_step,
                  coordinate_limits, value_limits, axes=None):
    '''
    Plot smoothed seismicity models by layer and minimum magnitude.

    Data is plotted on a square latitude-longitude grid.
    '''
    if axes is None:
        fig, axes = plt.subplots(len(layer_ids), len(min_mags),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0.3)

    for layer_id, row_axes in zip(layer_ids, axes):
        for min_mag, ax in zip(min_mags, row_axes):

            df = pd.read_csv(file_template % (layer_id, min_mag))
            df.rename(columns=COORDINATE_ALIASES, inplace=True)

            rate = df[df.columns[-1]].values
            image = ax.scatter(df[COORDINATES[0]], df[COORDINATES[1]],
                               c=rate, s=(grid_step/0.2)**2, edgecolor='none',
                               cmap='jet', norm=LogNorm(*value_limits))

    fig = axes.ravel()[0].get_figure()
    for ax, min_mag in zip(locations(fig, axes, max, 'ymax'), min_mags):
        if ax in axes.ravel():
            ax.set_title('M > %g' % min_mag)
    for ax, min_mag in zip(locations(fig, axes, min, 'ymin'), min_mags):
        ax.set_xlabel(u'Longitude [°]')
        ax.xaxis.set_major_locator(MultipleLocator(base=10.))
        ax.set_xlim(coordinate_limits[:2])
    for ax, layer_id in zip(locations(fig, axes, min, 'xmin'), layer_ids):
        annotate('Layer %d' % layer_id, loc='lower left', ax=ax)
        ax.set_ylabel(u'Latitude [°]')
        ax.yaxis.set_major_locator(MultipleLocator(base=10.))
        ax.set_ylim(coordinate_limits[2:])

    return image


def try_again(success=False):
    '''
    If at first you don't succeed, try, try again.
    '''
    while not success:
        try_again()
