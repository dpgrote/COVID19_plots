# data from https://github.com/CSSEGISandData/COVID-19

import datetime
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from TimeSeries import TimeSeriesCountries, TimeSeriesStates, TimeSeriesCounties

matplotlib.interactive(True)

#which = 'deaths'
#which = 'confirmed'
#which = 'recovered'

country_list = ['France',
                'Italy',
                'US',
                'Spain',
                'Germany',
                'China',
                'Korea, South',
                'India',
                'Russia',
                'United Kingdom',
                'Japan',
                'Australia']

state_list = ['California',
              'Washington',
              'Oregon',
              'New York',
              'New Jersey',
              'Massachusetts',
              'Michigan',
              'Florida',
              'Virginia',
              'Maryland']

county_list = ['Contra Costa',
               'Alameda',
               'San Francisco',
               'Marin',
               'San Mateo',
               'Santa Clara',
               'Solano'
               ]

countries_data = {'confirmed':TimeSeriesCountries(which='confirmed'),
                  'deaths':TimeSeriesCountries(which='deaths')}
states_data    = {'confirmed':TimeSeriesStates(which='confirmed'),
                  'deaths':TimeSeriesStates(which='deaths')}
counties_data  = {'confirmed':TimeSeriesCounties(which='confirmed'),
                  'deaths':TimeSeriesCounties(which='deaths')}

country_populations = pandas.read_csv('country_populations.csv')
state_populations = pandas.read_csv('state_populations.csv')
county_populations = pandas.read_csv('county_populations.csv')

#import pdb
#pdb.set_trace()

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
    doubling_days_max = kw.get('doubling_days_max', 10.)

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
        if nsmooth is not None:
            smoother(cases, nsmooth)
        dates = dates[1:]

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

    ax.plot(dates, cases, label=region)


def _plot_regions(ax, dataframe_dict, region_list, which, kw):
    scale_population = kw.get('scale_population', False)
    logderivative = kw.get('logderivative', False)
    derivative = kw.get('derivative', False)
    day_zero_value = kw.get('day_zero_value', None)
    do_legend = kw.get('do_legend', False)
    ymin = kw.get('ymin', None)

    for region in region_list:
        cases, dates = dataframe_dict[which].data(region)
        _plot_region(ax, region, cases, dates, kw)

    if day_zero_value is None:
        # set so ~10 dates are shown on the x axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_xlabel('Date')
    else:
        ax.set_xlabel('Days since start')

    if logderivative or derivative:
        ax.set_ylim(0.)
    else:
        ax.set_yscale('log')

    if ymin is not None:
        ax.set_ylim(ymin)

    ylabel = 'cumulative '+which
    if derivative:
        ylabel = f'new {which} per day'
    if scale_population:
        ylabel += ' per capita'
    if logderivative:
        ylabel = f'{which} doubling days'
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False, which='both')

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plotcountries(ax, country_list, which, **kw):
    kw.setdefault('population_df', country_populations)
    kw.setdefault('start_date', datetime.date(2020, 3, 1))
    _plot_regions(ax, countries_data, country_list, which, kw)

def plotstates(ax, state_list, which, **kw):
    kw.setdefault('population_df', state_populations)
    kw.setdefault('start_date', datetime.date(2020, 3, 1))
    _plot_regions(ax, states_data, state_list, which, kw)

def plotcounties(ax, county_list, which, **kw):
    kw.setdefault('population_df', county_populations)
    kw.setdefault('start_date', datetime.date(2020, 3, 20))
    _plot_regions(ax, counties_data, county_list, which, kw)


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


def plotcountries_delayed_death_rates(ax, country_list, do_legend=False, start_date=None, ymax=0.2, xmin=100.):
    _plot_regions_delayed_death_rates(ax, countries_data, country_list, do_legend, start_date, ymax, xmin)

def plotstates_delayed_death_rates(ax, state_list, do_legend=False, start_date=datetime.date(2020, 3, 1), ymax=0.2, xmin=100.):
    _plot_regions_delayed_death_rates(ax, states_data, state_list, do_legend, start_date, ymax, xmin)

def plotcounties_delayed_death_rates(ax, county_list, do_legend=False, start_date=datetime.date(2020, 3, 20), ymax=0.2, xmin=100.):
    _plot_regions_delayed_death_rates(ax, counties_data, county_list, do_legend, start_date, ymax, xmin)


####################### Countries
country_list_confirmed = countries_data['confirmed'].find_maxes(population_df=country_populations, mincases=5000)
for country in ['US', 'China', 'Korea, South']:
    if country not in country_list_confirmed:
        country_list_confirmed.append(country)

country_list_deaths = countries_data['deaths'].find_maxes(population_df=country_populations, mincases=500)
for country in ['US', 'China', 'Korea, South']:
    if country not in country_list_deaths:
        country_list_deaths.append(country)

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcountries(ax[0,0], country_list_confirmed, which='confirmed', scale_population=False)
plotcountries(ax[1,0], country_list_deaths, which='deaths', scale_population=False)
plotcountries(ax[0,1], country_list_confirmed, which='confirmed', scale_population=True, do_legend=True, ymin=1.e-6)
plotcountries(ax[1,1], country_list_deaths, which='deaths', scale_population=True, do_legend=True, ymin=1.e-6)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.58, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.text(0.87, 0.12, 'Top 10 per capita\nwith deaths > 500,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_cases.png')
fig.show()

"""
fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcountries(ax[0,0], country_list_confirmed, which='confirmed', scale_population=False, day_zero_value=100)
plotcountries(ax[1,0], country_list_deaths, which='deaths', scale_population=False, day_zero_value=100)
plotcountries(ax[0,1], country_list_confirmed, which='confirmed', scale_population=True, do_legend=True, day_zero_value=1.e-6)
plotcountries(ax[1,1], country_list_deaths, which='deaths', scale_population=True, do_legend=True, day_zero_value=1.e-6)

fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.58, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.text(0.87, 0.12, 'Top 10 per capita\nwith deaths > 500,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_cases_shifted.png')
fig.show()
"""

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcountries(ax[0,0], country_list_confirmed, which='confirmed', scale_population=False, derivative=True, do_legend=False)
plotcountries(ax[1,0], country_list_deaths, which='deaths', scale_population=False, derivative=True, do_legend=False)
plotcountries(ax[0,1], country_list_confirmed, which='confirmed', scale_population=True, derivative=True, do_legend=True)
plotcountries(ax[1,1], country_list_deaths, which='deaths', scale_population=True, derivative=True, do_legend=True)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.58, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.text(0.87, 0.12, 'Top 10 per capita\nwith deaths > 500,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_cases_per_day.png')
fig.show()

fig, ax = plt.subplots(2, figsize=(7,8))

plotcountries(ax[0], country_list_confirmed, which='confirmed', scale_population=False, logderivative=True, doubling_days_max=20., do_legend=True)
plotcountries(ax[1], country_list_deaths, which='deaths', scale_population=False, logderivative=True, doubling_days_max=20., do_legend=True)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.77, 0.58, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.text(0.77, 0.12, 'Top 10 per capita\nwith deaths > 500,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_doubling_rates.png')
fig.show()


####################### States
state_list = states_data['confirmed'].find_maxes(population_df=state_populations, mincases=100)
if 'California' not in state_list:
    state_list.append('California')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

if 'California' not in state_list:
    state_list.append('California')
plotstates(ax[0,0], state_list, which='confirmed', scale_population=False)
plotstates(ax[1,0], state_list, which='deaths', scale_population=False)
plotstates(ax[0,1], state_list, which='confirmed', scale_population=True, do_legend=True, ymin=1.e-6)
plotstates(ax[1,1], state_list, which='deaths', scale_population=True, ymin=1.e-6)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.82, 0.58, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_cases.png')
fig.show()

"""
fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotstates(ax[0,0], state_list, which='confirmed', scale_population=False, day_zero_value=20)
plotstates(ax[1,0], state_list, which='deaths', scale_population=False, day_zero_value=20)
plotstates(ax[0,1], state_list, which='confirmed', scale_population=True, do_legend=True, day_zero_value=5.e-6)
plotstates(ax[1,1], state_list, which='deaths', scale_population=True, day_zero_value=5.e-6)

fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.82, 0.58, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_cases_shifted.png')
fig.show()
"""

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotstates(ax[0,0], state_list, which='confirmed', scale_population=False, derivative=True, do_legend=False)
plotstates(ax[1,0], state_list, which='deaths', scale_population=False, derivative=True, do_legend=False)
plotstates(ax[0,1], state_list, which='confirmed', scale_population=True, derivative=True, do_legend=True)
plotstates(ax[1,1], state_list, which='deaths', scale_population=True, derivative=True, do_legend=False)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.82, 0.58, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_cases_per_day.png')
fig.show()

fig, ax = plt.subplots(2, figsize=(7,8))

plotstates(ax[0], state_list, which='confirmed', scale_population=False, logderivative=True, do_legend=True)
plotstates(ax[1], state_list, which='deaths', scale_population=False, logderivative=True, do_legend=False)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.75, 0.58, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_doubling_rates.png')
fig.show()


####################### Counties

county_list = counties_data['confirmed'].find_maxes(population_df=county_populations, mincases=100)
if 'Contra Costa' not in county_list:
    county_list.append('Contra Costa')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcounties(ax[0,0], county_list, which='confirmed', scale_population=False)
plotcounties(ax[1,0], county_list, which='deaths', scale_population=False)
plotcounties(ax[0,1], county_list, which='confirmed', scale_population=True, do_legend=True, ymin=1.e-5)
plotcounties(ax[1,1], county_list, which='deaths', scale_population=True, ymin=1.e-6)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.58, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_cases.png')
fig.show()

"""
fig, ax = plt.subplots(2, figsize=(7,8))

plotcounties(ax[0], county_list, which='confirmed', do_legend=True, day_zero_value=10)
plotcounties(ax[1], county_list, which='deaths', do_legend=False, day_zero_value=10)

fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.85, 0.58, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_cases_shifted.png')
fig.show()
"""

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcounties(ax[0,0], county_list, which='confirmed', scale_population=False, derivative=True, do_legend=False)
plotcounties(ax[1,0], county_list, which='deaths', scale_population=False, derivative=True, do_legend=False)
plotcounties(ax[0,1], county_list, which='confirmed', scale_population=True, derivative=True, do_legend=True)
plotcounties(ax[1,1], county_list, which='deaths', scale_population=True, derivative=True, do_legend=False)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.58, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_cases_per_day.png')
fig.show()

fig, ax = plt.subplots(2, figsize=(7,8))

plotcounties(ax[0], county_list, which='confirmed', logderivative=True, do_legend=True)
plotcounties(ax[1], county_list, which='deaths', logderivative=True, do_legend=False)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.78, 0.58, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_doubling_rates.png')
fig.show()


####################### Death rates

fig, ax = plt.subplots(3, figsize=(12,8))

plotcountries_delayed_death_rates(ax[0], country_list_deaths, do_legend=True)
plotstates_delayed_death_rates(ax[1], state_list, do_legend=True, ymax=0.1)
plotcounties_delayed_death_rates(ax[2], county_list, do_legend=True, ymax=0.1)

fig.tight_layout()
fig.subplots_adjust(bottom=.125)

if delay > 0:
    fig.suptitle(f'Confirmed cases lagged by {delay} days', y=0.96)
fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.tight_layout()
fig.savefig('../../Dropbox/Public/COVID19/delayed_death_rates.png')
fig.show()
