# data from https://github.com/CSSEGISandData/COVID-19

import datetime
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from TimeSeries import TimeSeriesCountries, TimeSeriesStates, TimeSeriesCounties
from California_data import CaliforniaData

matplotlib.interactive(True)

country_populations = pandas.read_csv('country_populations.csv')
state_populations = pandas.read_csv('state_populations.csv')
county_populations = pandas.read_csv('county_populations.csv')

def smoother(n, nsmooth=1):
    for i in range(nsmooth):
        ncopy = np.zeros(len(n) + 2)
        ncopy[1:-1] = n
        n[:] = 0.25*(ncopy[:-2] + ncopy[2:]) + 0.5*ncopy[1:-1]
        n[0] *= 4./3.
        n[-1] *= 4./3.

def _plot_region(ax, region, cases, dates, kw):
    scale_population = kw.get('scale_population', False)
    population_df = kw.get('population_df', None)
    logderivative = kw.get('logderivative', False)
    derivative = kw.get('derivative', False)
    day_zero_value = kw.get('day_zero_value', None)
    start_date = kw.get('start_date', None)
    nsmooth = kw.get('nsmooth', 5)
    doubling_days_max = kw.get('doubling_days_max', 40.)
    line_color = kw.get('line_color', None)

    if scale_population:
        population = int(population_df[population_df['Name'] == region]['Population'])
        cases /= population

    if logderivative:
        if cases.max() == 0:
            cases[...] = 1
        casesmin = (cases[cases>0]).min()
        log10cases = np.log10(cases.clip(casesmin).astype(float))
        log2cases = log10cases/np.log10(2.)
        cases = (log2cases[2:] - log2cases[:-2])/2.
        if nsmooth is not None:
            smoother(cases, nsmooth)
        cases = 1./cases.clip(1./doubling_days_max)
        dates = dates[1:-1]

    if derivative:
        if cases.max() == 0:
            cases[...] = 1
        cases = cases[1:] - cases[:-1]
        dates = dates[1:]

    if nsmooth is not None:
        smoother(cases, nsmooth)

    if day_zero_value is not None:
        ii_included = np.nonzero(cases >= day_zero_value)[0]
        if len(ii_included) <= 2:
            #print(f'Not enough data for {region} over day_zero_value')
            return
        # --- ii is the last value below day_zero_value
        ii = ii_included[0] - 1
        if ii == -1:
            # --- All values are > day_zero_value.
            # --- In this case, extrapolation will be done
            ii = 0
        if cases[ii] == 0.:
            ii += 1
        denom = np.log10(cases[ii+1]) - np.log10(cases[ii])
        if denom == 0.:
            ww = 0.
        else:
            ww = (np.log10(day_zero_value) - np.log10(cases[ii]))/denom
        cases = cases[ii_included]
        dates = np.arange(len(cases)) + (1. - ww)
    elif start_date is not None:
        ii = np.nonzero(np.greater(dates, start_date))[0]
        cases = cases[ii]
        dates = np.take(dates, ii)

    if line_color is None:
        ax.plot(dates, cases, label=region)
    else:
        ax.plot(dates, cases, line_color, label=region)


def _plot_regions(ax, dataframe, region_list, which, kw):
    scale_population = kw.get('scale_population', False)
    logderivative = kw.get('logderivative', False)
    derivative = kw.get('derivative', False)
    day_zero_value = kw.get('day_zero_value', None)
    do_legend = kw.get('do_legend', False)
    ylabel = kw.get('ylabel', None)

    for region in region_list:
        cases, dates = dataframe.data(region, which)
        _plot_region(ax, region, cases, dates, kw)

    if day_zero_value is None:
        # set so ~10 dates are shown on the x axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_xlabel('Date')
    else:
        ax.set_xlabel('Days since start')

    if ylabel is None:
        ylabel = 'cumulative '+which
        if derivative:
            ylabel = f'new {which} per day'
        if scale_population:
            ylabel += ' per capita'
        if logderivative:
            ylabel = f'{which} doubling days'
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False, which='both')

    ax.set_ylim(0.)

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plotcountries(ax, country_list, countries_data, which, **kw):
    kw.setdefault('population_df', country_populations)
    kw.setdefault('start_date', datetime.date(2020, 3, 1))
    _plot_regions(ax, countries_data[which], country_list, which, kw)

def plotstates(ax, state_list, states_data, which, **kw):
    kw.setdefault('population_df', state_populations)
    kw.setdefault('start_date', datetime.date(2020, 3, 1))
    _plot_regions(ax, states_data[which], state_list, which, kw)

def plotcounties(ax, county_list, counties_data=None, which=None, **kw):
    dataframe = kw.get('dataframe', None)
    if dataframe is None:
        dataframe = counties_data[which]
    kw.setdefault('population_df', county_populations)
    kw.setdefault('start_date', datetime.date(2020, 3, 20))
    _plot_regions(ax, dataframe, county_list, which, kw)


delay = 0
def _plot_delayed_death_rates(ax, region, confirmed, deaths, start_date=None, dates=None):

    if len(confirmed) < delay:
        return

    if start_date is not None:
        ii = np.nonzero(np.greater(dates, start_date))[0]
        confirmed = confirmed[ii]
        deaths = deaths[ii]

    if delay > 0:
        delayed_confirmed = confirmed[:-delay]
    else:
        delayed_confirmed = confirmed

    delayed_ratio = deaths[delay:]/delayed_confirmed.clip(1.)
    ax.plot(delayed_confirmed, delayed_ratio, label=region)


def _plot_regions_delayed_death_rates(ax, dataframe_dict, region_list, do_legend=False, start_date=None,
                                      ymax=0.2, xmin = None):

    for region in region_list:
        confirmed, dates = dataframe_dict['confirmed'].data(region)
        deaths, dates = dataframe_dict['deaths'].data(region)
        _plot_delayed_death_rates(ax, region, confirmed, deaths, start_date, dates)

    ax.set_xlabel('cumulative confirmed cases')
    ax.set_ylim(0., ymax)
    if xmin is not None:
        ax.set_xlim(xmin)
    ax.set_xscale('log')

    ylabel = 'deaths/confirmed'
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False, which='both')

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plotcountries_delayed_death_rates(ax, country_list, countries_data, do_legend=False, start_date=None, ymax=0.2, xmin=100.):
    _plot_regions_delayed_death_rates(ax, countries_data, country_list, do_legend, start_date, ymax, xmin)

def plotstates_delayed_death_rates(ax, state_list, states_data, do_legend=False, start_date=datetime.date(2020, 3, 1), ymax=0.2, xmin=100.):
    _plot_regions_delayed_death_rates(ax, states_data, state_list, do_legend, start_date, ymax, xmin)

def plotcounties_delayed_death_rates(ax, county_list, counties_data, do_legend=False, start_date=datetime.date(2020, 3, 20), ymax=0.2, xmin=100.):
    _plot_regions_delayed_death_rates(ax, counties_data, county_list, do_legend, start_date, ymax, xmin)


trajectory_days = 7
def _plot_region_trajectory(ax, region, cases, kw):
    scale_population = kw.get('scale_population', False)
    population_df = kw.get('population_df', None)
    nsmooth = kw.get('nsmooth', None)

    if scale_population:
        population = int(population_df[population_df['Name'] == region]['Population'])
        cases /= population

    ax.plot(cases[trajectory_days:], cases[trajectory_days:] - cases[:-trajectory_days], label=region)


def _plot_regions_trajectory(ax, dataframe, region_list, which, kw):
    scale_population = kw.get('scale_population', False)
    do_legend = kw.get('do_legend', False)
    xymin = kw.get('xymin', None)

    for region in region_list:
        cases, dates = dataframe.data(region, which)
        _plot_region_trajectory(ax, region, cases, kw)

    ax.set_xscale('log')
    ax.set_yscale('log')

    if xymin is not None:
        ax.set_xlim(xymin)
        ax.set_ylim(xymin)

    xlabel = f'Total {which}'
    ylabel = f'{which.capitalize()} last {trajectory_days} days'

    if scale_population:
        xlabel += ' per capita'
        ylabel += ' per capita'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.tick_params(right=True, labelright=False, which='both')

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plotcountries_trajectory(ax, country_list, countries_data, which, **kw):
    kw.setdefault('population_df', country_populations)
    _plot_regions_trajectory(ax, countries_data[which], country_list, which, kw)

def plotstates_trajectory(ax, state_list, states_data, which, **kw):
    kw.setdefault('population_df', state_populations)
    _plot_regions_trajectory(ax, states_data[which], state_list, which, kw)

def plotcounties_trajectory(ax, county_list, counties_data=None, which=None, **kw):
    dataframe = kw.get('dataframe', None)
    if dataframe is None:
        dataframe = counties_data[which]
    kw.setdefault('population_df', county_populations)
    _plot_regions_trajectory(ax, dataframe, county_list, which, kw)

