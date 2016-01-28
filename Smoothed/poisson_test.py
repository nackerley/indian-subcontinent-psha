# -*- coding: utf-8 -*-
# pylint: disable=superfluous-parens
"""
Utilities for testing poissoneity.


@author: Nick Ackerley
"""

from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def count_events(times, time_range, t_step=1.):
    """
    Customize histogram a bit, geared to catalogs and year-wide bins.
    """
    bin_edges = np.arange(time_range[0], time_range[1] + 0.5*t_step, t_step)
    counts = np.histogram(times, bins=bin_edges)[0]

    return (counts, bin_edges)


def plot_histogram(times, time_range, t_step=1.):
    """
    For visual comparison to Poisson distribution
    """

    counts = count_events(times, time_range, t_step=t_step)[0]
    observed = np.array(stats.itemfreq(counts))
    bins = np.arange(np.ceil(1.5*observed[-1, 0]))
    ideal = len(times)/counts.mean()*stats.poisson.pmf(bins, counts.mean())

    color_cycle = plt.gca()._get_lines.color_cycle
    plt.vlines(bins, 0, ideal,
               label='Poisson', color=next(color_cycle), lw=12, alpha=0.3)
    plt.vlines(observed[:, 0], 0, observed[:, 1],
               label='observed', color=next(color_cycle), lw=4)
    plt.xlim(bins[0], bins[-1])


def dispersion(times, time_range, t_step=1., verbose=False):
    """
    As described in Brown & Zhao (2002) and Luen & Stark (2012). Also known as
    the "conditional chi-square test".

    Works well, but "not as sensitive to overdispersion - apparent fluctuations
    in the rate of seismicity - as some other tests" (see Brown & Zhao, 2002,
    p. 693).
    """
    counts = count_events(times, time_range, t_step=t_step)[0]

    mean = counts.mean()
    statistic = sum(((counts - mean)**2)/mean)  # Brown & Zhao (2002) eq. (7)
    p_value = stats.chi2.sf(statistic, len(counts) - 1)

    if verbose:
        print("X_mean, chi_squared, p-value: %g, %g, %.2g" %
              (mean, statistic, p_value))

    return p_value


def brown_zhao(times, time_range, t_step=1., verbose=False):
    """
    As described in Brown & Zhao (2002), makes use of the variance
    stabilizing transformation of Anscombe (1948).

    FIXME: Seems "over-sensitive"; can't figure out why.
    """
    counts = count_events(times, time_range, t_step=t_step)[0]

    # apply variance-stabilizing transformation
    transformed = np.sqrt(counts + 3/8)  # Brown & Zhao (2002) eq. (4)

    mean = transformed.mean()
    chi_squared = 4*sum((transformed - mean)**2)  # Luen & Stark (2012) eq. (8)
    p_value = stats.norm.sf(chi_squared, len(counts) - 1)

    if verbose:
        print("Y_mean, chi_squared, p-value: %g, %g, %.2g" %
              (mean, chi_squared, p_value))

    return p_value


def kolmogorov_smirnov(times, time_range, dist,
                       mean_wait=None, t_step=1., verbose=False):
    """
    Michael (2011) uses KS to compare wait times to exponential distribution.
    According to Daub (2015) this is "sensitive to short-term clustering in
    the data but not to long-term changes in the rate." (p. 5702)

    Luen & Stark (2012) use KS to compare times to uniform distribution.
    Daub (2015) comments that this test is "more sensitive to long-term
    variations in the rate." (p. 5702)

    FIXME: Lilliefors (1969) points out dangers of using KS to estimate p-value
    when the rate parameters are inferred from the data (rather than known a
    priori). See also Daub (2015). This seems to only apply to exponential
    version of this test. It may be sufficient to use an asymptotic form like
    that of Stephens (1974) for large numbers of events (many degrees of
    freedom). There's a decent informal discussion at
    http://stats.stackexchange.com/questions/110272/a-naive-question-about-the-kolmogorov-smirnov-test

    FIXME: there is actually little overlap in the two sub-methods and so they
    should perhaps be split.
    """

    duration = time_range[1] - time_range[0]

    if mean_wait is None:
        if verbose and dist == 'expon':
            print("WARNING: Using inferred mean wait")
        mean_wait = duration/len(times)

    if dist == 'expon':
        normalized = np.diff(times)/mean_wait
    elif dist == 'uniform':
        normalized = np.array(times - time_range[0])/duration

    statistic, pvalue = stats.kstest(normalized, dist)

    if verbose:
        print("mean wait, statistic, p-value: %g, %g, %.2g" %
              (mean_wait, statistic, pvalue))

    return pvalue


def combine_pvalues(p_values_tuple, method='fisher'):
    """
    https://en.wikipedia.org/wiki/Fisher's_method
    """
    p_values = [stats.combine_pvalues(row, method=method)[1]
                for row in np.vstack(p_values_tuple).T]

    return p_values


def anderson_darling(times, distribution='expon', significance=5,
                     verbose=False):
    """
    http://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test

    FIXME: Not working. I don't understand this method yet.
    """

    assert distribution in ['norm', 'expon', 'logistic', 'gumbel', 'extreme1']

    statistic, critical_values, significance_levels = stats.anderson(
        times, dist=distribution)

    critical_value = critical_values[np.argwhere(
        significance_levels == significance)]
    if verbose:
        print("statistic, significance, critical value: %g, %g%%, %.2g" %
              (statistic, significance, critical_value))

    return statistic


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


def annotate(text, loc='upper right', ax=None, frameon=False):
    """
    Adds text to current or specified axis using legend location codes
    """
    if ax is None:
        ax = plt.gca()

    if loc not in LOC_CODE.keys():
        default_loc = 'upper right'
        print("'%s' not in %s: defaulting to '%s'" % (
            loc, LOC_CODE.keys(), default_loc))
        loc = default_loc
    ax.add_artist(AnchoredText(text, loc=LOC_CODE[loc], frameon=frameon))
