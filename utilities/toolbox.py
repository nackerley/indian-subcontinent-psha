'''
General-purpose helper module.

Module exports:
:func:`stdval`
:func:`logspace`
:func:`remove_chartjunk`
:func:`wrap`
:func:`summarize`
:func:`is_numeric`
:func:`df_compare`
:func:`array2compact`
:class:`Structure`
:class:`ChangedDIr`
'''

import os
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


R_SERIES = {
    3: [1, 2, 5, 10],
    5: [1.00, 1.60, 2.50, 4.00, 6.30, 10.00],
    10: [1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.00],
    20: [1.00, 1.12, 1.25, 1.40, 1.60, 1.80, 2.00, 2.24, 2.50, 2.80,
         3.15, 3.55, 4.00, 4.50, 5.00, 5.60, 6.30, 7.10, 8.00, 9.00, 10.00],
    40: [1.00, 1.06, 1.12, 1.18, 1.25, 1.32, 1.40, 1.50, 1.60, 1.70,
         1.80, 1.90, 2.00, 2.12, 2.24, 2.36, 2.50, 2.65, 2.80, 3.00,
         3.15, 3.35, 3.55, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.30,
         5.60, 6.00, 6.30, 6.70, 7.10, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00],
    80: [1.00, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.22, 1.25, 1.28,
         1.32, 1.36, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75,
         1.80, 1.85, 1.90, 1.95, 2.00, 2.06, 2.12, 2.18, 2.24, 2.30,
         2.36, 2.43, 2.50, 2.58, 2.65, 2.72, 2.80, 2.90, 3.00, 3.07,
         3.15, 3.25, 3.35, 3.45, 3.55, 3.65, 3.75, 3.87, 4.00, 4.12,
         4.25, 4.37, 4.50, 4.62, 4.75, 4.87, 5.00, 5.15, 5.30, 5.45,
         5.60, 5.80, 6.00, 6.15, 6.30, 6.50, 6.70, 6.90, 7.10, 7.30,
         7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50, 9.75, 10.00],
}


E_SERIES = {
    6: [10, 15, 22, 33, 47, 68, 100],
    12: [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100],
    24: [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
         33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91, 100],
    48: [100, 105, 110, 115, 121, 127, 133, 140, 147, 154, 162, 169,
         178, 187, 196, 205, 215, 226, 237, 249, 261, 274, 287, 301,
         316, 332, 348, 365, 383, 402, 422, 442, 464, 487, 511, 536,
         562, 590, 619, 649, 681, 715, 750, 787, 825, 866, 909, 953, 1000],
    96: [100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130,
         133, 137, 140, 143, 147, 150, 154, 158, 162, 165, 169, 174,
         178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232,
         237, 243, 249, 255, 261, 267, 274, 280, 287, 294, 301, 309,
         316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412,
         422, 432, 442, 453, 464, 475, 487, 499, 511, 523, 536, 549,
         562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732,
         750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976, 1000],
    192: [100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 113, 114,
          115, 117, 118, 120, 121, 123, 124, 126, 127, 129, 130, 132,
          133, 135, 137, 138, 140, 142, 143, 145, 147, 149, 150, 152,
          154, 156, 158, 160, 162, 164, 165, 167, 169, 172, 174, 176,
          178, 180, 182, 184, 187, 189, 191, 193, 196, 198, 200, 203,
          205, 208, 210, 213, 215, 218, 221, 223, 226, 229, 232, 234,
          237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 271,
          274, 277, 280, 284, 287, 291, 294, 298, 301, 305, 309, 312,
          316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 357, 361,
          365, 370, 374, 379, 383, 388, 392, 397, 402, 407, 412, 417,
          422, 427, 432, 437, 442, 448, 453, 459, 464, 470, 475, 481,
          487, 493, 499, 505, 511, 517, 523, 530, 536, 542, 549, 556,
          562, 569, 576, 583, 590, 597, 604, 612, 619, 626, 634, 642,
          649, 657, 665, 673, 681, 690, 698, 706, 715, 723, 732, 741,
          750, 759, 768, 777, 787, 796, 806, 816, 825, 835, 845, 856,
          866, 876, 887, 898, 909, 920, 931, 942, 953, 965, 976, 988, 1000]
}


def stdval(value, num=96, bump=0, preferred=None):
    '''
    Computes nearest values in a standard-value series.

    For non-standard E-numbers, Renard numbers are used. Note that the
    standard E-series do not strictly follow the Renard number series,
    which is why lookup tables must be used. An optional variable "bump"
    specifies the number by which the series index is to be adjusted up or
    down, and is useful for ceiling/floor type operations.

    Arguments
    :param value: Values to which the nearest standard values are sought
    :param num: Number of preferred values in standard series,
        e.g. 96 for E96 series
    :type num: list[int]
    :param preferred: Preferred values, e.g. [1, 2, 5, 10]
    :type preferred::mod:numpy:array:
    :param float bump: Amount by which each series index is adjusted before
        choosing value

    Note that num=3 gives same result as preferred=[1, 2, 5, 10], but
    num=24 would not give the correct E24 series if it weren't overridden.

    :returns output: Nearest standard values after rounding
    '''

    # we're going to need to do some elementwise operations
    x_type = type(value)
    value = np.asarray(value, dtype=float)

    # and some operations which depend on input being a column vector
    x_shape = value.shape
    value = np.reshape(value, (value.size, 1))

    # negative values will not be handled
    value[value < 0] = np.nan

    if preferred is None:
        if num in E_SERIES.keys():
            preferred = np.array(E_SERIES[num])
        elif num in R_SERIES.keys():
            preferred = np.array(R_SERIES[num])
        log_series = True
    else:
        # for the purpose of "bumping" it will be assumed that the preferred
        # values are approximately logarithmically-spaced and span a decade
        preferred = np.asarray(preferred, dtype=float)
        preferred = np.reshape(preferred, (1, preferred.size))
        num = preferred.size - 1
        log_series = False

    # bump input up or down as requested to support rounding up and down
    if log_series:
        value = value*10**(np.asarray(bump)/num)

    if preferred is not None:
        # determine how many digits of result to keep
        preferred = np.asarray(preferred, dtype=float)
        preferred = np.reshape(preferred, (1, preferred.size))
        digits = len('%d' % preferred[0][0])

        # compute multiplier for rounding
        multiplier = 10**np.floor(np.log10(value) - digits + 1)

        # shift input to have the right number of digits
        value = value/multiplier

        # find nearest standard value in a logarithmic sense
        pref_mat = np.tile(np.log10(preferred), (value.size, 1)).transpose()
        x_mat = np.tile(np.log10(value), (1, preferred.size)).transpose()

        log_dist = pref_mat - x_mat
        i_closest = np.argmin(np.abs(log_dist), 0)

        if log_series or bump == 0:
            output = preferred[0, i_closest]
        else:
            # for non-logarithmic series, bump just means round up or down
            if bump > 0 and log_dist[i_closest] < 0:
                if i_closest == preferred.size:
                    output = preferred[1]*10
                else:
                    output = preferred[i_closest + 1]

            elif bump < 0 and log_dist[i_closest] > 0:
                if i_closest == 1:
                    output = preferred[-1]/10
                else:
                    output = preferred[i_closest - 1]

        # restore correct number
        output = output[:, None]*multiplier

    else:
        # the nth power of a decade is the base
        base = 10**(1.0/num)

        # determine how many digits of result to keep
        digits = np.max((1, -np.round(np.log10(base - 1) - 1)))

        # compute the sequence number
        exponent = np.round(np.log(value)/np.log(base))

        # compute raw result
        raw = base**exponent

        # compute multiplier for rounding
        multiplier = 10**np.floor(np.log10(raw) - digits + 1)

        # round result to requested number of digits
        output = np.round(raw/multiplier)*multiplier

    output = np.reshape(output, x_shape)

    if (x_type is int) | (x_type is float):
        output = float(output)

    return output


def logspace(start, stop, num=12):
    '''
    Computes logarithmically spaced vector of preferred numbers.

    See stdval.
    '''
    log_start = np.floor(np.log10(start))
    log_stop = np.ceil(np.log10(stop))
    n_total = num*(log_stop-log_start) + 1
    temp = stdval(np.logspace(log_start, log_stop, num=n_total), num=num)
    return temp[np.bitwise_and(temp >= start, temp <= stop)]


def linspace(start, stop, target=30):
    '''
    Computes nicely-aligned bin edges for a target number of bins.

    See stdval.
    '''
    step = stdval((stop - start)/target, num=3)
    start = np.floor(start/step)*step
    stop = np.ceil(stop/step)*step
    return np.arange(start, stop + step, step)


def remove_chartjunk(axis, spines, grid=None, ticklabels=None,
                     show_ticks=False):
    '''
    Removes "chartjunk", such as extra lines of axes and tick marks.

    If grid="output" or "input", will add a white grid at the "output" or
    "input" axes, respectively

    If ticklabels="output" or "input", or ['input', 'output'] will remove
    ticklabels from that axis
    '''

    all_spines = ['top', 'bottom', 'right', 'left', 'polar']
    for spine in spines:
        # The try/except is for polar coordinates, which only have a 'polar'
        # spine and none of the others
        try:
            axis.spines[spine].set_visible(False)
        except KeyError:
            pass

    # For the remaining spines, make their line thinner and a slightly
    # off-black dark grey
    for spine in all_spines:
        if spine not in spines:
            # The try/except is for polar coordinates, which only have a
            # 'polar' spine and none of the others
            try:
                axis.spines[spine].set_linewidth(1)
            except KeyError:
                pass
                # axis.spines[spine].set_color(almost_black)
                # axis.spines[spine].set_tick_params(color=almost_black)
                # Check that the axes are not log-scale. If they are, leave
                # the ticks because otherwise people assume a linear scale.
    x_pos = set(['top', 'bottom'])
    y_pos = set(['left', 'right'])
    xy_pos = [x_pos, y_pos]
    xy_ax_names = ['xaxis', 'yaxis']

    for ax_name, pos in zip(xy_ax_names, xy_pos):
        axis = axis.__dict__[ax_name]
        # axis.set_tick_params(color=almost_black)
        # print 'axis.get_scale()', axis.get_scale()
        if show_ticks or axis.get_scale() == 'log':
            # if this spine is not in the list of spines to remove
            for positions in pos.difference(spines):
                # print 'p', p
                axis.set_tick_params(which='both', direction='out')
                axis.set_ticks_position(positions)
                #                axis.set_tick_params(which='both', p)
        else:
            axis.set_ticks_position('none')

    if grid is not None:
        for grid_string in grid:
            assert grid_string in ('input', 'output')
            axis.grid(axis=grid, color='white', linestyle='-', linewidth=0.5)

    if ticklabels is not None:
        if isinstance(ticklabels, str):
            assert ticklabels in set(('input', 'output'))
            if ticklabels == 'input':
                axis.set_xticklabels([])
            if ticklabels == 'output':
                axis.set_yticklabels([])
        else:
            assert set(ticklabels) | set(('input', 'output')) > 0
            if 'input' in ticklabels:
                axis.set_xticklabels([])
            elif 'output' in ticklabels:
                axis.set_yticklabels([])


def wrap(values, limit=180.0):
    '''
    Wraps to +/- specified limit, defaulting to 180 (i.e. degrees)

    :param values: number or numpy.array to be wrapped
    :param limit: limiting value (+/-)
    '''
    # use of negative modulus ensures -limit is wrapped to +limit
    return (values - limit) % (-2*limit) + limit


def summarize(obj):
    '''Summarizes the attributes of an object.'''
    print('\n'.join("%s: %s" % (item, getattr(obj, item))
                    for item in dir(obj) if ((len(item) == 0) or
                                             (item[0] != '_'))))


def is_numeric(value):
    '''
    Check if value is castable to float.
    '''
    try:
        float(value)
        return True
    except ValueError:
        return False


def df_compare(df_new, df_ref):
    '''
    Compares two pandas.DataFrame objects by computing their difference.
    '''

    numeric = np.array([[is_numeric(item)
                         for item in row[1]]
                        for row in df_ref.iterrows()])
    non_zero = (df_ref != 0) & numeric
    df_dif = df_ref.copy()
    df_dif = df_dif.where(~numeric,
                          df_new.where(numeric) - df_ref.where(numeric))
    df_dif = df_dif.where(~non_zero,
                          df_dif.where(non_zero)/df_ref.where(non_zero))
    df_dif = df_dif.where(numeric,
                          df_new.where(~numeric) != df_ref.where(~numeric))

    return df_dif


_NUMPY_STR_BLOAT = [
    ['\n', ''],
    ['[ ', '['],
    [' ,', ','],
    ['. ', '.'],
]


def array2compact(array, separator=','):
    '''
    Returns a compact string representation of an array as nested lists.

    >>> model = np.array([[123.456, 123.000000001], [2345., 345.6789]])
    >>> print np.array_repr(model).replace('\\n', '')
        array([[  123.456 ,   123.    ],       [ 2345.    ,   345.6789]])
    >>> print np.array2string(model, separator=',').replace('\\n', '')
        [[  123.456 ,  123.    ], [ 2345.    ,  345.6789]]
    >>> print array2compact(model)
        [[123.456, 123.], [2345., 345.6789]]

    '''

    compact = np.array2string(array, separator=separator)
    compact = ' '.join(compact.split())
    for item, replacement in _NUMPY_STR_BLOAT:
        compact = compact.replace(item, replacement)
    return compact


def limit_precision(array, sig_figs=5):
    '''
    Round to given number of significant figures.
    '''
    shape = array.shape
    fmt = '%.' + str(sig_figs) + 'g'
    values = np.array([float(fmt % x) for x in array.ravel()])
    return values.reshape(shape)


class Structure(object):
    # pylint: disable=R0903
    '''
    Is this a dangerous hack? It certainly is useful ...
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ChangedDir(object):  # pylint: disable-msg=R0903
    '''
    Context manager for changing the current working directory
    '''
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)
        self.saved_path = None

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


LOC_CODE = {
    'upper right': 1,
    'upper left': 2,
    'lower left': 3,
    'lower right': 4,
    'right': 5,
    'center left': 6,
    'center right': 7,
    'lower center': 8,
    'upper center': 9,
    'center': 10,
}


def annotate(text, loc='upper right', ax=None, frameon=False, prop=None):
    '''
    Adds text to current or specified axis using legend location codes
    '''
    if ax is None:
        ax = plt.gca()

    if loc not in LOC_CODE.keys():
        default_loc = 'upper right'
        print("'%s' not in %s: defaulting to '%s'" % (
            loc, LOC_CODE.keys(), default_loc))
        loc = default_loc
    if prop:
        ax.add_artist(AnchoredText(text, loc=LOC_CODE[loc], frameon=frameon,
                                   prop=prop))
    else:
        ax.add_artist(AnchoredText(text, loc=LOC_CODE[loc], frameon=frameon))


def find_files(pattern, path):
    '''
    Find files in a path matching a pattern.
    '''
    result = []
    for root, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def great_circle_distance_m(lat1, lon1, lat2, lon2, radius_m=6371e3):
    '''
    A vectorized Haversine formula.

    Arguments
    ---------
    lat1, lon1, lat2, lon2: float or numpy `~numpy.array`
        pairs of coordinates in degrees

    Returns
    -------
    float
        distance in km
    '''
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    return 2*radius_m*np.arcsin(
        np.sqrt((np.sin((lat2 - lat1)/2)**2 +
                 np.sin((lon2 - lon1)/2)**2*np.cos(lat1)*np.cos(lat2))))


def df_diff(df1, df2, column):
    '''
    Find the differences between two dataframes, returning all rows from
    df1 which don't have the named column in common with df2.
    '''
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    df_merge = pd.merge(df1[[column]], df2[[column]], on=column)
    matches = [index for index, value in df1[column].iteritems()
               if value in df_merge[column].values]
    return df1.drop(matches)
