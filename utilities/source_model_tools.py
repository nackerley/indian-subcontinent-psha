# -*- coding: utf-8 -*-
# pylint: disable=superfluous-parens
'''
Helper functions for OpenQuake source modeling.

Most of these functions are utilities specific to Nath & Thingbaijam (2012).
'''
import os
import re
import ast
import gzip
import shutil
from io import StringIO
from copy import deepcopy
from numbers import Number
from collections import OrderedDict
from itertools import product
from lxml import etree

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

import scipy
import numpy as np
import pandas as pd
from shapely.wkt import loads, dumps
import geopandas as gpd

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

COORDINATES = ['longitude', 'latitude']


class PseudoCatalogue:
    """
    ugly hack for plotting source mechanisms:
    construct pseudo-catalogue from pandas.DataFrame
    """
    def __init__(self, source_model, select=None, exclude_id_endswith='m'):
        rows = []
        sources = [source for group in source_model for source in group]
        for source in sources:

            row = OrderedDict((
                ('id', source.source_id),
                ('depth', source.hypocenter_distribution.data[0][1]),
                ('upper_depth', source.upper_seismogenic_depth),
                ('lower_depth', source.lower_seismogenic_depth),
                ('longitude', np.mean(source.polygon.lons)),
                ('latitude', np.mean(source.polygon.lats)),
                ('strike1', source.nodal_plane_distribution.data[0][1].strike),
                ('dip1', source.nodal_plane_distribution.data[0][1].dip),
                ('rake1', source.nodal_plane_distribution.data[0][1].rake),
                ('magnitude', 2.5*source.get_min_max_mag()[1]),
                ))

            if row['id'].endswith(exclude_id_endswith):
                continue

            if select is None or \
                    all(row[key] == value for key, value in select.items()):
                rows.append(row)

        self.data = pd.DataFrame(rows)

    def get_number_tensors(self):
        return len(self.data.magnitude)


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


class Kml(object):
    '''Keyhole Markup Language object tree'''

    def __init__(self, name=None):
        self.root = etree.Element('kml',
                                  xmlns='http://www.opengis.net/kml/2.2')
        self.doc = etree.SubElement(self.root, 'Document')
        self.write_extended = True
        self.schema_id = None
        if name is not None:
            etree.SubElement(self.doc, 'name').text = name

    def add_schema(self, df_table, schema_name, schema_id=None):
        '''Adds a schema for ExtendedData based on a pandas.DataFrame'''
        if schema_id is None:
            schema_id = schema_name + 'Id'
        schema = etree.SubElement(self.doc, 'Schema',
                                  name=schema_name, id=schema_id)
        field_names = df_table.columns.values.tolist()
        kml_types = [self._KML_TYPES[dtype.name]
                     for dtype in df_table.dtypes.values.tolist()]
        for field_name, kml_type in zip(field_names, kml_types):
            etree.SubElement(schema, 'SimpleField',
                             type=kml_type, name=field_name)
        self.schema_id = schema_id

    def _add_placemark(self, name, desc, extra_data=None):
        '''Adds a placemark to a KML tree'''
        placemark = etree.SubElement(self.doc, 'Placemark')
        etree.SubElement(placemark, 'name').text = str(name)
        etree.SubElement(placemark, 'description').text = desc
        if extra_data is not None and self.write_extended:
            extended = etree.SubElement(placemark, 'ExtendedData')
            if self.schema_id:
                schema = etree.SubElement(extended, 'SchemaData',
                                          schemaUrl='#' + self.schema_id)
                for item in extra_data.keys():
                    datum = etree.SubElement(schema, 'SimpleData', name=item)
                    datum.text = str(extra_data[item])
            else:
                for item in extra_data.keys():
                    datum = etree.SubElement(extended, 'Data', name=item)
                    etree.SubElement(datum,
                                     'value').text = str(extra_data[item])
        return placemark

    def add_point(self, name, desc, coord, extra_data=None):
        '''Adds a point to a KML tree'''
        coord = np.array(coord)
        np.pad(coord, (0, 3 - coord.size),
               'constant', constant_values=0)

        placemark = self._add_placemark(name, desc, extra_data)
        point = etree.SubElement(placemark, 'Point')
        coord_string = ','.join(str(x) for x in coord)
        etree.SubElement(point, 'coordinates').text = coord_string

    def add_polygon(self, name, desc, coords, extra_data=None):
        '''Adds a closed polygon to a KML tree'''
        coords = np.array(coords)
        np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])),
               'constant', constant_values=0)
        if not all(coords[0] == coords[-1]):
            coords = np.vstack((coords, coords[0]))
        assert coords.shape[0] > 3

        placemark = self._add_placemark(name, desc, extra_data)
        polygon = etree.SubElement(placemark, 'Polygon')
        boundary = etree.SubElement(polygon, 'outerBoundaryIs')
        ring = etree.SubElement(boundary, 'LinearRing')
        etree.SubElement(ring, 'tessellate').text = '1'
        coord_string = ' '.join(','.join(str(x) for x in coord) + ',0'
                                for coord in coords)
        etree.SubElement(ring, 'coordinates').text = coord_string

    def write(self, output_file, compress=False, compress_large=False):
        '''
        Writes tree to a KML file.

        If resultin KML file is larger than 5 MiB it is compressed using GZIP
        and '.gz' is appended to the file name.
        '''
        tree = etree.ElementTree(self.root)
        tree.write(output_file, encoding='UTF-8', pretty_print=True,
                   xml_declaration=True, with_tail=True)

        if compress or (compress_large and
                        (os.path.getsize(output_file) >
                         0.5*scipy.constants.mebi)):
            with open(output_file, 'rb') as f_in, \
                    gzip.open(output_file + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(output_file)
            output_file = output_file + '.gz'

        return output_file

    # more numpy dtypes are possible but these seem to be the only ones
    # automatically generated by pandas.read_csv
    _KML_TYPES = {
        'int64': 'int',
        'float64': 'double',
        'object': 'string'
        }


def source_df_to_kml_tree(df, layer_name, schema_name='SourceModelSchema'):
    '''
    Converts source description table to Keyhole Markup Language tree.

    :param df: table of descriptions of sources
    :type df: :class:`pandas.DataFrame`
    :param str layer_name: used for layer naming in KML tree
    :param str schema_name: name of schema to be declared in KML file
    :returns: source description tree
    :rtype: :class:`lxml.etree.ElementTree`
    '''
    kml = Kml(layer_name.replace('_', ' '))
    kml.add_schema(df, schema_name)

    point_source = get_source_class(df) is mtkPointSource

    for zoneid, series in df.iterrows():

        name = str(zoneid)
        description = series.pop('tectonic subregion')
        if point_source:
            coords = [series.pop('longitude'), series.pop('latitude')]
            kml.add_point(name, description, coords, series)
        else:
            geometry = series.pop('geometry')
            if isinstance(geometry, str):
                geometry = loads(geometry)
            coords = list(zip(*geometry.exterior.coords.xy))
            kml.add_polygon(name, description, coords, series)

    return kml


def source_df_to_kml(df, name, schema_name='SourceModelSchema'):
    '''
    Converts source description table to Keyhole Markup Language file.

    File name is constructed from name by converting spaces to
    underscores and appending '.kml'.

    :param df: table of descriptions of sources
    :type df: :class:`pandas.DataFrame`
    :param str name: name of KML tree
    :param str schema_name: name of schema to be declared in KML file
    :returns str: file name
    '''
    kml = source_df_to_kml_tree(df, name, schema_name=schema_name)
    file_name = name.replace(' ', '_') + '.kml'
    print('Writing:\n\t' + os.path.abspath(file_name))
    return kml.write(file_name)


def df2kml(df, base_name, by='layerid'):
    '''
    Write groups of data to KML with added binwise rates.
    '''
    df = df.copy()
    df = add_binwise_rates(df)
    df = add_name_id(df)
    for group_id, group_df in df.groupby(by):
        source_df_to_kml(df=group_df.drop(['layerid'], axis=1),
                         name='%s layer %d' % (base_name, group_id))


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


ALL_REQUIRED = ['id', 'source_name', 'zmin', 'zmax', 'tectonic subregion',
                'msr', 'strike', 'dip', 'rake', 'aspect ratio']

POINT_REQUIRED = ['longitude', 'latitude']
AREA_REQUIRED = ['geometry']

GR_REQUIRED = ['mmin', 'mmax', 'a', 'b']
DISCRETE_REQUIRED = ['mmin', 'occurRates', 'magBin']


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
    if all(key in df.columns for key in POINT_REQUIRED):
        source_class = mtkPointSource
    elif all(key in df.columns for key in AREA_REQUIRED):
        source_class = mtkAreaSource
    else:
        raise ValueError(
            'Only area [%s] and point [%s] sources currently supported' %
            (', '.join(POINT_REQUIRED), ', '.join(AREA_REQUIRED)))

    return source_class


def _areal_source_name(series):
    return 'zone %s' % series.name


def _areal_source_id(series):
    return 'z%s' % series.name


def _point_source_name(series):
    return '%gN %gE %g-%g km depth' % (series.latitude, series.longitude,
                                       series.zmin, series.zmax)


def _point_source_id(series):
    result = '%gN_%gE_%d' % (series.latitude, series.longitude, series.layerid)
    # For the source IDs OpenQuake only accepts a-zA-z0-9_-
    return result.replace('.', 'p')


def make_source_geometry(series, source_class):
    '''
    Given a source description, returns the required class and geometry
    as well as an appropriate name for the source.

    :param series: source description
    :type series: :class:`pandas.Series` or dict

    :returns: tuple of (source_class, geometry, name)
    :type source_class: e.g. :class:`openquake.hmtk.sources.area_source.mtkAreaSource`
    :type geometry: e.g. :class:`openquake.hazardlib.geo.point.Point` instance
    :type name: str
    '''  # noqa

    if source_class is mtkPointSource:
        geometry = geo.point.Point(series['longitude'],
                                   series['latitude'])

    elif source_class is mtkAreaSource:
        list_points = series['geometry']  # FIXME
        if isinstance(list_points, str):
            list_points = ast.literal_eval(list_points)
        points = [geo.point.Point(lon, lat) for lon, lat in list_points]
        geometry = geo.polygon.Polygon(points + [points[0]])

    else:
        raise ValueError('Source class %s not supported' %
                         source_class.__name__)

    return geometry


def _check_columns(df):
    missing = [item for item in ALL_REQUIRED if item not in df.columns]
    if missing:
        raise ValueError(
            'Missing required columns: ' + ', '.join(missing))
    missing_point = [item for item in POINT_REQUIRED if item not in df.columns]
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

    source_list = source_df_to_list(df)
    source_model = mtkSourceModel(identifier='1',
                                  name=model_name,
                                  sources=source_list)

    print('Writing:\n\t%s' % os.path.abspath(nrml_file))
    source_model.serialise_to_nrml(nrml_file)

    return source_model


def points2nrml(df, base_name, by=['mmin model'], fmt='mmin%g'):
    '''
    Write multiple pandas DataFrame of point source models to NRML.
    '''
    df.sort_values(['mmin model', 'layerid'] + POINT_REQUIRED, inplace=True)

    for index, group_df in df.groupby(by):
        model_name = base_name + ' ' + fmt % index
        df2nrml(group_df, model_name)


def points2csv(df, base_name, by=['mmin model', 'layerid'],
               fmt='mmin%g layer%d'):
    '''
    Write grouped data with added names, ids and binwise rates.
    '''
    # TODO: select columns of interest, or at least control column order?
    df.drop('geometry', axis=1, inplace=True)

    df = add_name_id(df)
    df = add_binwise_rates(df)

    _check_columns(df)

    df.sort_values(by + POINT_REQUIRED, inplace=True)

    for index, group_df in df.groupby(by):
        model_name = base_name + ' ' + fmt % index
        csv_file = model_name.replace(' ', '_') + '.csv'
        print('Writing:\n\t' + os.path.abspath(csv_file))
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
        print('Reading:\n\t' + os.path.abspath(csv_file))
        df = pd.read_csv(csv_file)
        for column, value in zip(by, values):
            df[column] = value
        dfs.append(df)

    df = pd.concat(dfs)

    _check_columns(df)

    df['geometry'] = df['geometry'].apply(loads)

    df.sort_values(by + POINT_REQUIRED, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def areal2csv(df, csv_file):
    '''
    Write areal model with names, ids and geometry.
    '''
    df = df.drop(columns=['polygon'])

    if not csv_file.endswith('.csv'):
        csv_file += '.csv'
    print('Writing:\n\t' + os.path.abspath(csv_file))
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
    print('Reading:\n\t' + os.path.abspath(csv_file))
    df = pd.read_csv(csv_file, index_col='zoneid')
    df['geometry'] = df['geometry'].apply(loads)
    df = gpd.GeoDataFrame(df, crs='WGS84')
    _check_columns(df)

    return df


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


def natural_keys(text):
    '''
    alist.sort_values(by=natural_keys) sorts strings in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    FIXME: Delete as deprecated if no longer needed.
    '''
    def atoi(text):
        '''Convert alphanumeric to integer if numeric'''
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


def sort_and_reindex(df_in, sort_columns=None):
    '''
    Standardize ordering, including by numeric string id.

    FIXME: Delete as deprecated if no longer needed.
    '''
    # identify columns of interest
    if sort_columns is None:
        sort_columns = ['id', 'zoneid', 'layerid', 'longitude', 'latitude',
                        'tectonic subregion', 'mmin']
    all_columns = df_in.columns.tolist()
    sort_columns = [item for item in sort_columns if item in all_columns]
    other_columns = [item for item in all_columns if item not in sort_columns]
    other_columns = sorted(other_columns)

    # sort in different ways depending on columns present
    if 'id' in sort_columns:
        df_in['natural_key'] = [natural_keys(item) for item in df_in['id']]
        df_out = df_in.sort_values('natural_key')
        df_out.drop('natural_key', axis=1, inplace=True)
    else:
        df_out = df_in.sort_values(sort_columns)

    # reindex and reorder columns
    df_out.index = range(len(df_out))
    df_out = df_out[sort_columns + other_columns]
    df_out.is_copy = False

    return df_out


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
