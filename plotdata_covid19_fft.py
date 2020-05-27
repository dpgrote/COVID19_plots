# data from https://github.com/CSSEGISandData/COVID-19

import datetime
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from TimeSeries import TimeSeriesCountries, TimeSeriesStates, TimeSeriesCounties
from California_data import CaliforniaData

matplotlib.interactive(True)

#which = 'deaths'
#which = 'confirmed'

countries_data = {'confirmed':TimeSeriesCountries(which='confirmed'),
                  'deaths':TimeSeriesCountries(which='deaths')}
states_data    = {'confirmed':TimeSeriesStates(which='confirmed'),
                  'deaths':TimeSeriesStates(which='deaths')}
counties_data  = {'confirmed':TimeSeriesCounties(which='confirmed'),
                  'deaths':TimeSeriesCounties(which='deaths')}

def _plot_region(ax, region, cases, dates, kw):
    derivative = kw.get('derivative', True)
    start_date = kw.get('start_date', None)
    nsmooth = kw.get('nsmooth', None)
    line_color = kw.get('line_color', None)

    if derivative:
        if cases.max() == 0:
            cases[...] = 1
        cases = cases[1:] - cases[:-1]
        if nsmooth is not None:
            smoother(cases, nsmooth)
        dates = dates[1:]

    ii = np.nonzero(np.greater(dates, start_date))[0]
    cases = cases[ii]
    dates = np.take(dates, ii)

    cases_hat = np.abs(np.fft.rfft(cases))
    freq = np.fft.rfftfreq(len(cases))

    if line_color is None:
        ax.plot(1./freq[1:], cases_hat[1:], label=region)
    else:
        ax.plot(1./freq[1:], cases_hat[1:], line_color, label=region)


def _plot_regions(ax, dataframe, region_list, which, kw):
    derivative = kw.get('derivative', False)
    do_legend = kw.get('do_legend', False)
    ylabel = kw.get('ylabel', None)

    for region in region_list:
        cases, dates = dataframe.data(region, which)
        _plot_region(ax, region, cases, dates, kw)

    if ylabel is None:
        ylabel = 'cumulative '+which
        if derivative:
            ylabel = f'new {which} per day'
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False, which='both')

    ax.set_ylim(0.)
    ax.set_xlim(1., 100.)
    ax.set_xscale('log')

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plotcountries(ax, country_list, which, **kw):
    kw.setdefault('start_date', datetime.date(2020, 3, 1))
    _plot_regions(ax, countries_data[which], country_list, which, kw)

def plotstates(ax, state_list, which, **kw):
    kw.setdefault('start_date', datetime.date(2020, 3, 1))
    _plot_regions(ax, states_data[which], state_list, which, kw)

def plotcounties(ax, county_list, which, **kw):
    dataframe = kw.get('dataframe', None)
    if dataframe is None:
        dataframe = counties_data[which]
    kw.setdefault('start_date', datetime.date(2020, 3, 20))
    _plot_regions(ax, dataframe, county_list, which, kw)


####################### Countries
country_list_confirmed = ['US']
country_list_deaths = ['US']

fig, ax = plt.subplots(2, 1, figsize=(12,8))

plotcountries(ax[0], country_list_confirmed, which='confirmed', do_legend=True)
plotcountries(ax[1], country_list_deaths, which='deaths')

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.savefig('../../Dropbox/Public/COVID19/country_cases_fft.png')
fig.show()


####################### States
state_list = ['California']

fig, ax = plt.subplots(2, 1, figsize=(12,8))

plotstates(ax[0], state_list, which='confirmed', do_legend=True)
plotstates(ax[1], state_list, which='deaths')

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.savefig('../../Dropbox/Public/COVID19/state_cases_fft.png')
fig.show()

####################### Counties
county_list = ['Contra Costa']

fig, ax = plt.subplots(2, 1, figsize=(12,8))

plotcounties(ax[0], county_list, which='confirmed', do_legend=True)
plotcounties(ax[1], county_list, which='deaths')

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.savefig('../../Dropbox/Public/COVID19/county_cases_fft.png')
fig.show()

