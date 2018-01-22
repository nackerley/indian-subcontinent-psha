# -*- coding: utf-8 -*-
"""
Helper functions for OpenQuake logic trees.

Module exports:

:func:`is_imt`
:func:`get_imt`
:func:`compute_gmpe`
:func:`df_massage`
"""

import os
import numpy as np
import pandas as pd
import toolbox as tb

from openquake.hazardlib import const, gsim, imt  # pylint: disable=E0611


def is_imt(value):
    """Check if value is castable to an intensity measure type."""
    if tb.is_numeric(value):
        return True
    else:
        try:
            imt.from_string(value.upper())
            return True
        except ValueError:
            return False


def get_imt(value):
    """Get intensity measure type corresponding to value."""
    if tb.is_numeric(value):
        return imt.from_string('SA(%g)' % float(value))
    else:
        try:
            return imt.from_string(value.upper())
        except ValueError:
            return None


def choose_attribute(prefix, preferred, required):
    """
    Utility function for picking GMPE parameters from lists.
    """
    attribute = next(iter([item for item in preferred
                           if item in required]), None)
    if attribute is not None:
        column = prefix + attribute
    else:
        column = None
    return (attribute, column)


def compute_gmpe(gmpe, mags, ruptures, distances, sites, im_types,
                 std_type=const.StdDev.TOTAL):
    # pylint: disable=too-many-arguments, too-many-locals
    """
    Compute mean ground motion and associated uncertainty for GMPE.

    :param gmpe: A ground-motion prediction equation
    :type gmpe: :class:`openquake.hazardlib.gsim.base.GMPE`
    :param list mags: Rupture magnitudes
    :param list ruptures: Rake angles in degrees or hypocentral depths in km
    :param list distances: Propagation distances in km
    :param list sites: Site vs30s in m/s or z1pt0 in km
    :param list im_types: List of :class:`openquake.hazardlib.imt._IMT`
    :param std_type: Uncertainty type of interest
    :type std_type: :class:`openquake.hazardlib.const.StdDev`

    :returns: Table of results
    :rtype: :class:`pandas.DataFrame`

    Notes
    -----

        #. mags, ruptures, distances and sites can be any numeric type castable
        to :class:`numpy.ndarray`.

        #. Only one type of uncertainty and value of damping are handled at a
        time.

    """
    std_result_type = std_type.replace(' ', '_').upper() + '_STDDEV'

    # determine exactly which context attributes are to be probed
    rup_attr, rup_col = choose_attribute('rup_', ['rake', 'hypo_depth'],
                                         gmpe.REQUIRES_RUPTURE_PARAMETERS)
    site_attr, site_col = choose_attribute('site_', ['vs30', 'z1pt0'],
                                           gmpe.REQUIRES_SITES_PARAMETERS)
    dist_attr, dist_col = choose_attribute('dist_', ['rjb', 'rrup', 'rhypo'],
                                           gmpe.REQUIRES_DISTANCES)
    damping = next(iter([im_type.damping for im_type in im_types
                         if isinstance(im_type, imt.SA)]), 5)

    # allow site and rupture attributes to be passed over
    if (rup_attr is None) and (ruptures is None):
        ruptures = 0
    if (site_attr is None) and (sites is None):
        sites = 0

    # cast inputs into numpy arrays
    sites = np.asarray([sites], dtype='float').reshape((-1,))
    distances = np.asarray([distances], dtype='float').reshape((-1,))

    # set up some reusable contexts
    rctx = gsim.base.RuptureContext()
    sctx = gsim.base.SitesContext()
    dctx = gsim.base.DistancesContext()

    empty = True
    for mag in np.asarray([mags], dtype='float').reshape((-1,)):
        for rupture in np.asarray([ruptures], dtype='float').reshape((-1,)):

            rctx.mag = mag
            if rup_attr is not None:
                setattr(rctx, rup_attr, rupture)
            setattr(dctx, dist_attr, np.tile(distances, sites.size))
            if site_attr is not None:
                setattr(sctx, site_attr, np.repeat(sites, distances.size))

            for i, im_type in enumerate(im_types):

                mean, [stddev] = gmpe.get_mean_and_stddevs(
                    sctx, rctx, dctx, im_type, [std_type])

                if i == 0:
                    df_mean = pd.DataFrame({
                        'gmpe': gmpe.__class__.__name__,
                        'rup_mag': np.full_like(mean, rctx.mag),
                        dist_col: getattr(dctx, dist_attr),
                        'damping': damping,
                        })
                    if site_col is not None:
                        df_mean[site_col] = sctx.vs30
                    if rup_col is not None:
                        df_mean[rup_col] = np.full_like(
                            mean, getattr(rctx, rup_attr))

                    df_stddev = df_mean.copy()
                    df_mean['result_type'] = 'MEAN'
                    df_stddev['result_type'] = std_result_type

                if isinstance(im_type, imt.SA):
                    imt_key = im_type.period
                else:
                    imt_key = str(im_type)
                df_mean[imt_key] = np.exp(mean)
                df_stddev[imt_key] = stddev

            if empty:
                df_means = df_mean
                df_stddevs = df_stddev
                empty = False
            else:
                df_means = pd.concat((df_means, df_mean),
                                     ignore_index=True)
                df_stddevs = pd.concat((df_stddevs, df_stddev),
                                       ignore_index=True)

    df_means = df_massage(df_means)
    df_stddevs = df_massage(df_stddevs)

    return (df_means, df_stddevs)


def print_gmpe_summary(gmpe):
    """
    Summarizes the metadata relating to the GMPE.
    """

    print(type(gmpe).__name__)
    print('Supported tectonic region: %s'
          % gmpe.DEFINED_FOR_TECTONIC_REGION_TYPE)
    print('Supported intensity measure types: %s'
          % ', '.join([item.__name__ for item
                       in gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES]))
    print('Supported component: %s'
          % gmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT)
    print('Supported standard deviations: %s'
          % ', '.join([item for item
                       in gmpe.DEFINED_FOR_STANDARD_DEVIATION_TYPES]))
    print('Required site parameters: %s'
          % ', '.join([item for item in gmpe.REQUIRES_SITES_PARAMETERS]))
    print('Required rupture parameters: %s'
          % ', '.join([item for item in gmpe.REQUIRES_RUPTURE_PARAMETERS]))
    print('Required distance parameters: %s'
          % ', '.join([item for item in gmpe.REQUIRES_DISTANCES]))
    print('')


def df_massage(df_ref):
    """Reorders pandas.DataFrame of GSIM test data in a reasonable way"""

    if len(df_ref) == 0:
        return df_ref

    # determine sensible column ordering
    cols_ref = df_ref.columns
    sa_cols = np.array([tb.is_numeric(item) for item in cols_ref])
    imt_cols = np.array([is_imt(item) for item in cols_ref]) & ~sa_cols
    gmpe_cols = np.array([(isinstance(item, str) and ('gmpe' in item))
                          for item in cols_ref])
    rup_cols = np.array([(isinstance(item, str) and ('rup_' in item))
                         for item in cols_ref])
    dist_cols = np.array([(isinstance(item, str) and ('dist_' in item))
                          for item in cols_ref])
    site_cols = np.array([(isinstance(item, str) and ('site_' in item))
                          for item in cols_ref])
    other_cols = ~gmpe_cols & ~rup_cols & ~dist_cols & \
                 ~site_cols & ~imt_cols & ~sa_cols

    cols_sorted = (sorted(cols_ref[gmpe_cols]) +
                   sorted(cols_ref[rup_cols]) +
                   sorted(cols_ref[dist_cols]) +
                   sorted(cols_ref[site_cols]) +
                   sorted(cols_ref[other_cols]) +
                   sorted(cols_ref[imt_cols]) +
                   sorted(cols_ref[sa_cols], key=float))

    df_ref = df_ref[cols_sorted].copy()

    # cast numeric column headers to float
    df_ref.columns = [float(item) if tb.is_numeric(item) else item
                      for item in df_ref.columns.values]

    # drop missing results
    df_ref.dropna(axis=1, inplace=True, how='all')

    # sort results and reindex
    df_ref.sort_values(sorted(cols_ref[gmpe_cols]) +
                       sorted(cols_ref[rup_cols]) +
                       sorted(cols_ref[dist_cols]) +
                       sorted(cols_ref[site_cols]) +
                       sorted(cols_ref[other_cols]), inplace=True)
    df_ref.index = range(len(df_ref))

    return df_ref


def write_test_data(df_results, gmpe_group, gmpes_short, gmpe_class_names,
                    group_name=None, float_format=None,
                    repo_path='~/src/python/GEM/oq-hazardlib/'):
    # pylint: disable=too-many-arguments
    """
    Write pandas.DataFrame to openquake gsim CSV file format.
    """

    test_path = os.path.join(os.path.expanduser(repo_path),
                             'openquake/hazardlib/tests/gsim/data/',
                             gmpe_group)

    if not os.path.exists(test_path):
        os.mkdir(test_path)

    df_results = df_massage(df_results)

    output_files = []
    for gmpe_name in sorted(list(set(df_results['gmpe']))):
        for result_type in sorted(list(set(df_results['result_type']))):

            gmpe_short = [short
                          for short, name in zip(gmpes_short, gmpe_class_names)
                          if gmpe_name == name][0]

            # construct output file name
            output_file = '%s_%s_%s' % (gmpe_group, gmpe_short, result_type)
            if group_name:
                output_file += '_' + group_name
            output_file = output_file.replace('__', '_')
            output_file += '.csv'
            output_file = os.path.join(test_path, output_file)

            indices = df_results['gmpe'] == gmpe_name
            if len(indices) > 0:
                df_gmpe = df_results[indices].copy()
                df_gmpe.drop(['gmpe'], axis=1, inplace=True)
                df_gmpe.drop_duplicates(inplace=True)
                df_gmpe = df_massage(df_gmpe)
                df_gmpe.to_csv(output_file, index=False,
                               float_format=float_format)
                output_files.append(output_file)

    return output_files


def get_imts(names):
    """
    Return any items in list which are castable to intensity measure types.

    :param list names: List of floats or strings.

    :returns list: List of items castable to
    :class:`openquake.hazardlib.imt._IMT`
    """
    im_names, im_types = [], []
    for item in names:
        if tb.is_numeric(item):
            im_types += [imt.from_string('SA(%g)' % item)]
            im_names += [item]
        else:
            try:
                im_types += [imt.from_string(item)]
                im_names += [item]
            except ValueError:
                pass

    return (im_names, im_types)
