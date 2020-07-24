# data from https://github.com/CSSEGISandData/COVID-19

import datetime
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.interactive(True)

country_populations = pandas.read_csv('country_populations.csv')
state_populations = pandas.read_csv('state_populations.csv')
county_populations = pandas.read_csv('county_populations.csv')


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

