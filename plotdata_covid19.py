# data from https://github.com/CSSEGISandData/COVID-19

import datetime
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from DailyReports import DailyReports

matplotlib.interactive(True)

#which = 'Deaths'
#which = 'Confirmed'
#which = 'Recovered'

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

dailyreports = DailyReports()

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

def _plot_region(ax, region, cases, dates, scale_population, population_df=None,
                 logderivative=False, day_zero_value=None, start_date=None, nsmooth=5,
                 doubling_days_max=10.):
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

    if day_zero_value is not None:
        ii_included = np.nonzero(cases >= day_zero_value)[0]
        if len(ii_included) == 0:
            #print(f'No data for {region} over day_zero_value')
            return
        # --- ii is the last value below day_zero_value
        ii = ii_included[0] - 1
        if ii == -1:
            # --- All values are > day_zero_value.
            # --- In this case, extrapolation will be done
            ii = 0
        ww = (np.log10(day_zero_value) - np.log10(cases[ii]))/(np.log10(cases[ii+1]) - np.log10(cases[ii]))
        cases = cases[ii_included]
        dates = np.arange(len(cases)) + (1. - ww)
    elif start_date is not None:
        ii = np.nonzero(np.greater(dates, start_date))[0]
        cases = cases[ii]
        dates = np.take(dates, ii)

    ax.plot(dates, cases, label=region)


def plotcountry(ax, country='France', which='Confirmed', scale_population=False,
                logderivative=False, day_zero_value=None, start_date=None, nsmooth=5,
                doubling_days_max=10.):
    cases, dates = dailyreports.country_data(country, which)
    _plot_region(ax, country, cases, dates, scale_population,
                 population_df=country_populations, logderivative=logderivative,
                 day_zero_value=day_zero_value, start_date=start_date, nsmooth=nsmooth,
                 doubling_days_max=doubling_days_max)

def plotstate(ax, state='California', which='Confirmed', scale_population=False,
              logderivative=False, day_zero_value=None, start_date=None, nsmooth=5,
              doubling_days_max=10.):
    cases, dates = dailyreports.state_data(state, which)
    _plot_region(ax, state, cases, dates, scale_population,
                 population_df=state_populations, logderivative=logderivative,
                 day_zero_value=day_zero_value, start_date=start_date, nsmooth=nsmooth,
                 doubling_days_max=doubling_days_max)

def plotcounty(ax, county='Contra Costa', which='Confirmed', scale_population=False,
               logderivative=False, day_zero_value=None, start_date=None, nsmooth=3,
               doubling_days_max=10.):
    cases, dates = dailyreports.county_data(county, which)
    _plot_region(ax, county, cases, dates, scale_population,
                 population_df=county_populations, logderivative=logderivative,
                 day_zero_value=day_zero_value, start_date=start_date, nsmooth=nsmooth,
                 doubling_days_max=doubling_days_max)

def _plot_regions(ax, plotfunc, region_list, which='Confirmed', scale_population=False,
                  do_legend=False, logderivative=False, start_date=None, day_zero_value=None,
                  doubling_days_max=10.):

    for region in region_list:
        plotfunc(ax, region, which, scale_population, logderivative, day_zero_value, start_date,
                 doubling_days_max=doubling_days_max)

    if day_zero_value is None:
        # set so ~10 dates are shown on the x axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_xlabel('Date')
    else:
        ax.set_xlabel('Days since start')

    if logderivative:
        ax.set_ylim(0.)
    else:
        ax.set_yscale('log')

    ylabel = '# '+which
    if scale_population:
        ylabel += '/pop'
    if logderivative:
        ylabel = 'doubling days ' + which
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False, which='both')

    #if start_date is not None and day_zero_value is None:
        #ax.set_xlim(start_date)

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plotcountries(ax, country_list, which='Confirmed', scale_population=False,
                  do_legend=False, logderivative=False, start_date=None, day_zero_value=None,
                  doubling_days_max=10.):
    _plot_regions(ax, plotcountry, country_list, which, scale_population,
                  do_legend, logderivative, start_date, day_zero_value,doubling_days_max)

def plotstates(ax, state_list, which='Confirmed', scale_population=False,
               do_legend=False, logderivative=False, start_date=datetime.date(2020, 3, 1), day_zero_value=None,
               doubling_days_max=10.):
    _plot_regions(ax, plotstate, state_list, which, scale_population,
                  do_legend, logderivative, start_date, day_zero_value,doubling_days_max)

def plotcounties(ax, county_list, which='Confirmed', scale_population=False,
                 do_legend=False, logderivative=False, start_date=datetime.date(2020, 3, 20), day_zero_value=None,
                 doubling_days_max=10):
    _plot_regions(ax, plotcounty, county_list, which, scale_population,
                  do_legend, logderivative, start_date, day_zero_value,doubling_days_max)


delay = 7
def _plot_delayed_death_rates(ax, region, confirmed, deaths, start_date=None, dates=None):

    if len(confirmed) < delay:
        return

    if start_date is not None:
        ii = np.nonzero(np.greater(dates, start_date))[0]
        confirmed = confirmed[ii]
        deaths = deaths[ii]

    delayed_ratio = deaths[delay:]/confirmed[:-delay].clip(1.)
    ax.plot(confirmed[:-delay], delayed_ratio, label=region)

def plotcountry_delayed_death_rates(ax, country='France', start_date=None):
    confirmed, dates = dailyreports.country_data(country, 'Confirmed')
    deaths, dates = dailyreports.country_data(country, 'Deaths')
    _plot_delayed_death_rates(ax, country, confirmed, deaths, start_date, dates)

def plotstate_delayed_death_rates(ax, state='California', start_date=None):
    confirmed, dates = dailyreports.state_data(state, 'Confirmed')
    deaths, dates = dailyreports.state_data(state, 'Deaths')
    _plot_delayed_death_rates(ax, state, confirmed, deaths, start_date, dates)

def plotcounty_delayed_death_rates(ax, county='Contra Costa', start_date=None):
    confirmed, dates = dailyreports.county_data(county, 'Confirmed')
    deaths, dates = dailyreports.county_data(county, 'Deaths')
    _plot_delayed_death_rates(ax, county, confirmed, deaths, start_date, dates)

def _plot_regions_delayed_death_rates(ax, plotfunc, region_list, do_legend=False, start_date=None):

    for region in region_list:
        plotfunc(ax, region, start_date)

    ax.set_xlabel('Confirmed cases')
    ax.set_ylim(0., 0.2)
    ax.set_xscale('log')

    ylabel = 'deaths/confirmed'
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False, which='both')

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plotcountries_delayed_death_rates(ax, country_list, do_legend=False, start_date=None):
    _plot_regions_delayed_death_rates(ax, plotcountry_delayed_death_rates, country_list, do_legend, start_date)

def plotstates_delayed_death_rates(ax, state_list, do_legend=False, start_date=datetime.date(2020, 3, 1)):
    _plot_regions_delayed_death_rates(ax, plotstate_delayed_death_rates, state_list, do_legend, start_date)

def plotcounties_delayed_death_rates(ax, county_list, do_legend=False, start_date=datetime.date(2020, 3, 20)):
    _plot_regions_delayed_death_rates(ax, plotcounty_delayed_death_rates, county_list, do_legend, start_date)



####################### Countries
country_list_confirmed = dailyreports.find_max_countries('Confirmed', population_df=country_populations, mincases=5000)
for country in ['US', 'China', 'Korea, South']:
    if country not in country_list_confirmed:
        country_list_confirmed.append(country)

country_list_deaths = dailyreports.find_max_countries('Deaths', population_df=country_populations, mincases=500)
for country in ['US', 'China', 'Korea, South']:
    if country not in country_list_deaths:
        country_list_deaths.append(country)

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcountries(ax[0,0], country_list_confirmed, which='Confirmed', scale_population=False)
plotcountries(ax[1,0], country_list_deaths, which='Deaths', scale_population=False)
plotcountries(ax[0,1], country_list_confirmed, which='Confirmed', scale_population=True, do_legend=True)
plotcountries(ax[1,1], country_list_deaths, which='Deaths', scale_population=True, do_legend=True)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.60, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.text(0.87, 0.10, 'Top 10 per capita\nwith deaths > 500,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_cases.png')
fig.show()

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcountries(ax[0,0], country_list_confirmed, which='Confirmed', scale_population=False, day_zero_value=100)
plotcountries(ax[1,0], country_list_deaths, which='Deaths', scale_population=False, day_zero_value=100)
plotcountries(ax[0,1], country_list_confirmed, which='Confirmed', scale_population=True, do_legend=True, day_zero_value=1.e-6)
plotcountries(ax[1,1], country_list_deaths, which='Deaths', scale_population=True, do_legend=True, day_zero_value=1.e-6)

fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.60, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.text(0.87, 0.10, 'Top 10 per capita\nwith deaths > 500,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_cases_shifted.png')
fig.show()

fig, ax = plt.subplots(2, figsize=(7,8))

plotcountries(ax[0], country_list_confirmed, which='Confirmed', scale_population=False, logderivative=True, doubling_days_max=20., do_legend=True)
plotcountries(ax[1], country_list_deaths, which='Deaths', scale_population=False, logderivative=True, doubling_days_max=20., do_legend=True)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.80, 0.60, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.text(0.87, 0.10, 'Top 10 per capita\nwith deaths > 500,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_doubling_rates.png')
fig.show()


####################### States
state_list = dailyreports.find_max_states('Confirmed', population_df=state_populations, mincases=100)
if 'California' not in state_list:
    state_list.append('California')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

if 'California' not in state_list:
    state_list.append('California')
plotstates(ax[0,0], state_list, which='Confirmed', scale_population=False)
plotstates(ax[1,0], state_list, which='Deaths', scale_population=False)
plotstates(ax[0,1], state_list, which='Confirmed', scale_population=True, do_legend=True)
plotstates(ax[1,1], state_list, which='Deaths', scale_population=True)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_cases.png')
fig.show()

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotstates(ax[0,0], state_list, which='Confirmed', scale_population=False, day_zero_value=20)
plotstates(ax[1,0], state_list, which='Deaths', scale_population=False, day_zero_value=20)
plotstates(ax[0,1], state_list, which='Confirmed', scale_population=True, do_legend=True, day_zero_value=5.e-6)
plotstates(ax[1,1], state_list, which='Deaths', scale_population=True, day_zero_value=5.e-6)

fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.85, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_cases_shifted.png')
fig.show()

fig, ax = plt.subplots(2, figsize=(7,8))

plotstates(ax[0], state_list, which='Confirmed', scale_population=False, logderivative=True, do_legend=True)
plotstates(ax[1], state_list, which='Deaths', scale_population=False, logderivative=True, do_legend=False)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.75, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_doubling_rates.png')
fig.show()


####################### Counties

county_list = dailyreports.find_max_counties('Confirmed', population_df=county_populations, mincases=100)
if 'Contra Costa' not in county_list:
    county_list.append('Contra Costa')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcounties(ax[0,0], county_list, which='Confirmed', scale_population=False)
plotcounties(ax[1,0], county_list, which='Deaths', scale_population=False)
plotcounties(ax[0,1], county_list, which='Confirmed', scale_population=True, do_legend=True)
plotcounties(ax[1,1], county_list, which='Deaths', scale_population=True)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_cases.png')
fig.show()

"""
fig, ax = plt.subplots(2, figsize=(7,8))

plotcounties(ax[0], county_list, which='Confirmed', do_legend=True, day_zero_value=10)
plotcounties(ax[1], county_list, which='Deaths', do_legend=False, day_zero_value=10)

fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.85, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_cases_shifted.png')
fig.show()
"""

fig, ax = plt.subplots(2, figsize=(7,8))

plotcounties(ax[0], county_list, which='Confirmed', logderivative=True, do_legend=True)
plotcounties(ax[1], county_list, which='Deaths', logderivative=True, do_legend=False)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.80, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_doubling_rates.png')
fig.show()


####################### Death rates

fig, ax = plt.subplots(3, figsize=(12,8))

plotcountries_delayed_death_rates(ax[0], country_list_deaths, do_legend=True)
plotstates_delayed_death_rates(ax[1], state_list, do_legend=True)
plotcounties_delayed_death_rates(ax[2], county_list, do_legend=True)

fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle(f'Confirmed cases lagged by {delay} days', y=0.96)
fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.tight_layout()
fig.savefig('../../Dropbox/Public/COVID19/delayed_death_rates.png')
fig.show()
