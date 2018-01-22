# -*- coding: utf-8 -*-
# pylint: disable=superfluous-parens
'''
Helper functions for OpenQuake source modeling.

Module exports:

:class:`Kml`
:func:`df_to_kml`
:func:`source_df_to_list`
:func:`make_source_geometry`
:func:`focal_mechanisms`
:func:`write_source_df_to_kml`
'''
import os
import re
import ast
import gzip
import shutil
from numbers import Number
from string import Template
from io import StringIO

import scipy
import numpy as np
import pandas as pd
from lxml import etree

from obspy.imaging.beachball import aux_plane

from openquake.hmtk.sources.area_source import mtkAreaSource
from openquake.hmtk.sources.point_source import mtkPointSource
from openquake.hazardlib import geo, mfd, pmf
from openquake.baselib.general import deprecated

import toolbox as tb


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


def source_df_to_kml_tree(source_df, layer_name,
                          schema_name='SourceModelSchema'):
    '''
    Converts source description table to Keyhole Markup Language tree.

    :param source_df: table of descriptions of sources
    :type source_df: :class:`pandas.DataFrame`
    :param str layer_name: used for layer naming in KML tree
    :param str schema_name: name of schema to be declared in KML file
    :returns: source description tree
    :rtype: :class:`lxml.etree.ElementTree`
    '''
    kml = Kml(layer_name.replace('_', ' '))
    kml.add_schema(source_df, schema_name)

    for _, source_series in source_df.iterrows():
        name = source_series.pop('zoneid')
        description = source_series.pop('tectonic subregion')
        if 'polygon coordinates' in source_series.keys():
            poly_coords = source_series.pop('polygon coordinates')
            kml.add_polygon(name, description, poly_coords, source_series)
        elif all((item in source_series.keys())
                 for item in ['longitude', 'latitude']):
            lon = source_series.pop('longitude')
            lat = source_series.pop('latitude')
            kml.add_point(name, description, [lon, lat], source_series)
        else:
            print("Unkwown source type, doing nothing:")
            print(source_series)

    return kml


def source_df_to_kml(source_df, layer_name, schema_name='SourceModelSchema'):
    '''
    Converts source description table to Keyhole Markup Language file.

    File name is constructed from layer name by converting spaces to
    underscores and appending '.kml'.

    :param source_df: table of descriptions of sources
    :type source_df: :class:`pandas.DataFrame`
    :param str layer_name: used for layer naming in KML tree
    :param str schema_name: name of schema to be declared in KML file
    :returns str: file name
    '''
    kml = source_df_to_kml_tree(source_df, layer_name, schema_name=schema_name)
    output_file = layer_name.replace(' ', '_') + '.kml'
    return kml.write(output_file)


@deprecated('Use df_to_kml instead')
def write_source_df_to_kml(source_df, layer_name):
    '''
    Takes a pandas dataframe and writes the data to KML (deprecated)

    Deprecated in favour of :class:Kml.
    '''
    coord_format = "\t\t\t\t\t\t\t%g,%g\n"

    polygon_template = Template(POLYGON_TEMPLATE)
    document_template = Template(KML_TEMPLATE)

    placemarks_string = ''
    for _, source in source_df.iterrows():
        coord_string = [(coord_format % tuple(coords))
                        for coords in source['polygon coordinates']]
        coord_string = (''.join(coord_string))[:-1]
        polygon_string = polygon_template.substitute(
            ZONEID=source['zoneid'],
            TRT=source['tectonic subregion'],
            COORDINATES=coord_string)
        placemarks_string += polygon_string

    document_string = document_template.substitute(
        LAYERNAME=layer_name,
        PLACEMARKS=placemarks_string)

    output_file = layer_name.replace(' ', '_') + '.kml'
    with open(output_file, 'w') as f:
        f.write(document_string)

    return output_file


def source_df_to_list(source_df, mag_bin_width=0.1):
    '''
    Converts source description table into list of source objects.

    :param source_df: table of descriptions of sources
    :type source_df: :class:`pandas.DataFrame`
    :param float mag_bin_width: magnitude frequency distribution discretization
    :returns: list of sources
    :rtype: list of e.g. :class:`openquake.hmtk.sources.area_source.mtkAreaSource`
    '''  # noqa

    source_list = []
    for _, source_series in source_df.iterrows():

        if ((source_series['a'] == 0) or
                (source_series['mmax'] == 0) or
                (source_series['dip'] == -1) or
                pd.isnull(source_series['dip'])):
            continue

        source_class, geometry = make_source_geometry(source_series)

        if 'occurRates' in source_series.keys():
            if isinstance(source_series['occurRates'], float):
                continue
            mag_freq_dist = mfd.EvenlyDiscretizedMFD(
                source_series['mmin'] + mag_bin_width/2, mag_bin_width,
                source_series['occurRates'].tolist())
        else:
            mag_freq_dist = mfd.TruncatedGRMFD(
                source_series['mmin'], source_series['mmax'], mag_bin_width,
                source_series['a'], source_series['b'])

        nodal_plane_pmf = pmf.PMF(
            [(1.0, geo.NodalPlane(strike=source_series['strike'],
                                  dip=source_series['dip'],
                                  rake=source_series['rake']))])

        hypo_depth_pmf = pmf.PMF(
            [(1.0, (source_series['zmin'] + source_series['zmax'])/2.0)])

        source = source_class(
            source_series['id'], source_series['name'], geometry=geometry,
            trt=source_series['tectonic subregion'],
            upper_depth=source_series['zmin'],
            lower_depth=source_series['zmax'],
            rupt_aspect_ratio=source_series['aspect ratio'],
            mag_scale_rel=source_series['msr'],
            mfd=mag_freq_dist,
            nodal_plane_dist=nodal_plane_pmf,
            hypo_depth_dist=hypo_depth_pmf)
        source_list.append(source)

    return source_list


def get_source_class(source_series):
    '''
    Determine source class based on presence of polygon or simple lat/lon.
    '''
    if 'polygon coordinates' in source_series.keys():
        source_class = mtkAreaSource
    elif all(key in source_series.keys() for key in ['longitude', 'latitude']):
        source_class = mtkPointSource
    else:
        raise ValueError('Only area and point sources currently supported')

    return source_class


def add_name_id(in_df):
    '''
    Add columns with short names and ids appropriate for NRML source models
    '''

    out_df = in_df.copy()
    names = []
    ids = []
    for _, source_series in out_df.iterrows():

        source_class = get_source_class(source_series)

        if source_class.__name__ == 'mtkPointSource':
            names += ['%gN %gE %g-%g km depth' % (
                source_series['latitude'], source_series['longitude'],
                source_series['zmin'], source_series['zmax'])]
            ids += ['%gN%gEz%s' % (
                source_series['latitude'], source_series['longitude'],
                str(source_series['zoneid']))]
        elif source_class.__name__ == 'mtkAreaSource':
            names += ['zone %s' % str(source_series['zoneid'])]
            ids += ['z%s' % str(source_series['zoneid'])]
        else:
            raise ValueError('Source class %s not supported' %
                             source_class.__name__)

    # For the source IDs OpenQuake only accepts a-zA-z0-9_-
    ids = [item.replace('.', 'p') for item in ids]

    out_df['name'] = names
    out_df['id'] = ids
    return out_df


def make_source_geometry(source_series):
    '''
    Given a source description, returns the required class and geometry
    as well as an appropriate name for the source.

    :param source_series: source description
    :type source_series: :class:`pandas.Series` or dict

    :returns: tuple of (source_class, geometry, name)
    :type source_class: e.g. :class:`openquake.hmtk.sources.area_source.mtkAreaSource`
    :type geometry: e.g. :class:`openquake.hazardlib.geo.point.Point` instance
    :type name: str
    '''  # noqa

    source_class = get_source_class(source_series)

    if source_class.__name__ == 'mtkPointSource':
        geometry = geo.point.Point(source_series['longitude'],
                                   source_series['latitude'])
    elif source_class.__name__ == 'mtkAreaSource':
        list_points = source_series['polygon coordinates']
        if isinstance(list_points, str):
            list_points = ast.literal_eval(list_points)
        points = [geo.point.Point(lon, lat) for lon, lat in list_points]
        geometry = geo.polygon.Polygon(points + [points[0]])
    else:
        raise ValueError('Source class %s not supported' %
                         source_class.__name__)

    return (source_class, geometry)


def focal_mechanism(dip, rake, threshold=30):
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
        rake = tb.wrap(rake)
        if 0 <= tb.wrap(dip) <= 90:
            if threshold < rake < 180 - threshold:
                return 'reverse'  # dip-slip
            elif threshold < -rake < 180 - threshold:
                return 'normal'  # dip-slip
            elif rake < threshold:
                return 'sinistral'  # strike-slip

            return 'dextral'  # strike-slip

        return 'undefined'

    return [focal_mechanism(d, r) for d, r in zip(dip, rake)]


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
        rake = tb.wrap(rake)

        if 0 <= tb.wrap(dip) <= 90:
            candidates = [
                focal_mechanism(dip, rake),
                focal_mechanism(*(aux_plane(strike, dip, rake)[1:]))
                ]

            return next(
                (faulting_style for faulting_style in candidates
                 if faulting_style in ['normal', 'reverse']),
                'strike-slip')

        return 'undefined'

    except ValueError:
        return [faulting_style(s, d, r) for s, d, r in zip(strike, dip, rake)]


KML_TEMPLATE = '''
<?xml version='1.0' encoding='UTF-8'?>
<kml xmlns='http://www.opengis.net/kml/2.2'>
  <Document>
    <name>$LAYERNAME</name>
$PLACEMARKS
  </Document>
</kml>

'''

POLYGON_TEMPLATE = '''
    <Placemark>
      <name>$ZONEID</name>
      <description>$TRT</description>
      <ExtendedData>
      </ExtendedData>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <tessellate>1</tessellate>
            <coordinates>
$COORDINATES
                        </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
'''


def twin_source_by_magnitude(source_df,
                             column='tectonic subregion',
                             select_type='subduction interface',
                             new_type='subduction interface megathrust',
                             mag_thresh=7.5,
                             twin_zone_suffix='m'):
    '''
    Twin selected sources by magnitude
    '''

    oq_valid_id = '[A-Za-z0-9_]*$'
    assert re.match(oq_valid_id, twin_zone_suffix), \
        '"%s" does not match "%s"' % (twin_zone_suffix, oq_valid_id)

    source_df = source_df.copy()

    # don't re-twin
    if any(source_df[column] == new_type):
        return source_df

    # cast zone id to string if not already
    source_df['zoneid'] = [str(item) for item in source_df['zoneid']]

    # create twinned sources
    indices = ((source_df[column] == select_type) &
               (source_df['mmax'] - source_df['stdmmax'] > mag_thresh))
    twinned_df = source_df[indices].copy()
    source_df.loc[indices, 'mmax'] = mag_thresh
    twinned_df['mmin'] = mag_thresh
    twinned_df['zoneid'] = [item + twin_zone_suffix
                            for item in twinned_df['zoneid']]
    twinned_df[column] = new_type

    # prune bins above/below maximum magnitude
    if 'occurRates' in source_df.columns:
        above_rates, below_rates = [], []
        for _, zone in source_df.loc[indices].iterrows():
            num_bins = zone['occurRates'].size
            mags = zone['mmin'] + zone['magBin']*(np.arange(num_bins) + 0.5)
            above_rates += [zone['occurRates'][mags > mag_thresh]]
            below_rates += [zone['occurRates'][mags < mag_thresh]]

        twinned_df['occurRates'] = above_rates
        source_df.loc[indices, 'occurRates'] = pd.Series(below_rates)

    source_df = pd.concat([source_df, twinned_df])
    source_df.index = range(len(source_df))

    return source_df


def add_binwise_rates(source_df, mag_start=5, mag_stop=8, mag_step=1):
    '''
    Add binwise sesismicity rates for comparison
    '''
    source_df = source_df.copy()
    for mag in range(mag_start, mag_stop, mag_step):
        m_lo = np.maximum(source_df['mmin'], mag)
        m_hi = np.minimum(source_df['mmax'], mag + 1)
        a_vals = source_df['a']
        b_vals = source_df['b']

        log_n_m_lo = a_vals - b_vals*m_lo
        log_n_m_hi = a_vals - b_vals*m_hi
        log_n_bin = np.log10(10**log_n_m_lo - 10**log_n_m_hi).round(2)
        series_name = 'logN_%.1f-%.1f' % (mag, mag + 1)

        source_df = source_df.join(
            pd.DataFrame(log_n_bin,
                         columns=[series_name],
                         index=source_df.index))

    return source_df


def natural_keys(text):
    '''
    alist.sort_values(by=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    def atoi(text):
        '''Convert alphanumeric to integer if numeric'''
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


def sort_and_reindex(df_in, sort_columns=None):
    '''An attempt to standardize ordering'''

    # identify columns of interest
    if sort_columns is None:
        sort_columns = ['id', 'longitude', 'latitude',
                        'zoneid', 'layerid',
                        'tectonic subregion', 'mmin']
    all_columns = df_in.columns.tolist()
    sort_columns = [item for item in sort_columns if item in all_columns]
    other_columns = [item for item in all_columns if item not in sort_columns]
    other_columns = sorted(other_columns)

    # sort in different ways depending on columns present
    unique_ids = [item for item in ['id', 'zoneid'] if item in sort_columns]
    if unique_ids:
        df_in['natural_key'] = [natural_keys(item)
                                for item in df_in[unique_ids[0]]]
        df_out = df_in.sort_values('natural_key')
        df_out.drop('natural_key', axis=1, inplace=True)
    else:
        df_out = df_in.sort_values(sort_columns)

    # reindex and reorder columns
    df_out.index = range(len(df_out))
    df_out = df_out[sort_columns + other_columns]
    df_out.is_copy = False

    return df_out


class MyPolygon(geo.polygon.Polygon):
    # pylint: disable=no-member,no-init,too-few-public-methods
    '''
    Wrapper to add a way to calculate distances.
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


def try_again(success=False):
    '''
    If at first you don't succeed, try, try again.
    '''
    while not success:
        try_again()
