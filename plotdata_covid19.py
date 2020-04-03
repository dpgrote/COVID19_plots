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
                 logderivative=False, day_zero_value=None):
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
        smoother(cases, nsmooth=5)
        cases = 1./cases.clip(0.1)
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

    ax.plot(dates, cases, label=region)


def plotcountry(ax, country='France', which='Confirmed', scale_population=False,
                logderivative=False, day_zero_value=None):
    cases, dates = dailyreports.country_data(country, which)
    _plot_region(ax, country, cases, dates, scale_population,
                 population_df=country_populations, logderivative=logderivative,
                 day_zero_value=day_zero_value)

def plotstate(ax, state='California', which='Confirmed', scale_population=False,
              logderivative=False, day_zero_value=None):
    cases, dates = dailyreports.state_data(state, which)
    _plot_region(ax, state, cases, dates, scale_population,
                 population_df=state_populations, logderivative=logderivative,
                 day_zero_value=day_zero_value)

def plotcounty(ax, county='Contra Costa', which='Confirmed', scale_population=False,
               logderivative=False, day_zero_value=None):
    cases, dates = dailyreports.county_data(county, which)
    _plot_region(ax, county, cases, dates, scale_population,
                 population_df=county_populations, logderivative=logderivative,
                 day_zero_value=day_zero_value)

def _plot_regions(fig, ax, plotfunc, region_list, which='Confirmed', scale_population=False,
                  do_legend=False, logderivative=False, start_date=None, day_zero_value=None):

    for region in region_list:
        plotfunc(ax, region, which, scale_population, logderivative, day_zero_value)

    if day_zero_value is None:
        # set nice formatting and centering
        fig.autofmt_xdate()

        # set so ~10 dates are shown on the x axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_xlabel('Date')
    else:
        ax.set_xlabel('Days since start')

    if logderivative:
        ax.set_ylim(0.)
    else:
        ax.set_yscale('log')

    fig.tight_layout()
    fig.subplots_adjust(bottom=.125)
    ylabel = '# '+which
    if scale_population:
        ylabel += '/pop'
    if logderivative:
        ylabel = 'doubling days ' + which
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False, which='both')

    if start_date is not None and day_zero_value is None:
        ax.set_xlim(start_date)

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plotcountries(fig, ax, country_list, which='Confirmed', scale_population=False,
                  do_legend=False, logderivative=False, start_date=None, day_zero_value=None):
    _plot_regions(fig, ax, plotcountry, country_list, which, scale_population,
                  do_legend, logderivative, start_date, day_zero_value)

def plotstates(fig, ax, state_list, which='Confirmed', scale_population=False,
               do_legend=False, logderivative=False, start_date=datetime.date(2020, 3, 1), day_zero_value=None):
    _plot_regions(fig, ax, plotstate, state_list, which, scale_population,
                  do_legend, logderivative, start_date, day_zero_value)

def plotcounties(fig, ax, county_list, which='Confirmed', scale_population=False,
                 do_legend=False, logderivative=False, start_date=datetime.date(2020, 3, 20), day_zero_value=None):
    _plot_regions(fig, ax, plotcounty, county_list, which, scale_population,
                  do_legend, logderivative, start_date, day_zero_value)


####################### Countries
country_list = dailyreports.find_max_countries('Confirmed', population_df=country_populations, mincases=5000)
for country in ['US', 'China', 'Korea, South']:
    if country not in country_list:
        country_list.append(country)

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcountries(fig, ax[0,0], country_list, which='Confirmed', scale_population=False)
plotcountries(fig, ax[1,0], country_list, which='Deaths', scale_population=False)
plotcountries(fig, ax[0,1], country_list, which='Confirmed', scale_population=True, do_legend=True)
plotcountries(fig, ax[1,1], country_list, which='Deaths', scale_population=True)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.55, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_cases.png')
fig.show()

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcountries(fig, ax[0,0], country_list, which='Confirmed', scale_population=False, day_zero_value=100)
plotcountries(fig, ax[1,0], country_list, which='Deaths', scale_population=False, day_zero_value=100)
plotcountries(fig, ax[0,1], country_list, which='Confirmed', scale_population=True, do_legend=True, day_zero_value=1.e-6)
plotcountries(fig, ax[1,1], country_list, which='Deaths', scale_population=True, day_zero_value=1.e-6)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.55, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_cases_shifted.png')
fig.show()

fig, ax = plt.subplots(2, figsize=(7,8))

plotcountries(fig, ax[0], country_list, which='Confirmed', scale_population=False, logderivative=True, do_legend=True)
plotcountries(fig, ax[1], country_list, which='Deaths', scale_population=False, logderivative=True, do_legend=False)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.80, 0.55, 'Top 10 per capita\nwith cases > 5000,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/country_doubling_rates.png')
fig.show()


####################### States
state_list = dailyreports.find_max_states('Confirmed', population_df=state_populations, mincases=100)
if 'California' not in state_list:
    state_list.append('California')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

if 'California' not in state_list:
    state_list.append('California')
plotstates(fig, ax[0,0], state_list, which='Confirmed', scale_population=False)
plotstates(fig, ax[1,0], state_list, which='Deaths', scale_population=False)
plotstates(fig, ax[0,1], state_list, which='Confirmed', scale_population=True, do_legend=True)
plotstates(fig, ax[1,1], state_list, which='Deaths', scale_population=True)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_cases.png')
fig.show()

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotstates(fig, ax[0,0], state_list, which='Confirmed', scale_population=False, day_zero_value=20)
plotstates(fig, ax[1,0], state_list, which='Deaths', scale_population=False, day_zero_value=20)
plotstates(fig, ax[0,1], state_list, which='Confirmed', scale_population=True, do_legend=True, day_zero_value=5.e-6)
plotstates(fig, ax[1,1], state_list, which='Deaths', scale_population=True, day_zero_value=5.e-6)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.85, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_cases_shifted.png')
fig.show()

fig, ax = plt.subplots(2, figsize=(7,8))

plotstates(fig, ax[0], state_list, which='Confirmed', scale_population=False, logderivative=True, do_legend=True)
plotstates(fig, ax[1], state_list, which='Deaths', scale_population=False, logderivative=True, do_legend=False)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.75, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/state_doubling_rates.png')
fig.show()


####################### Counties

county_list = dailyreports.find_max_counties('Confirmed', population_df=county_populations, mincases=100)
if 'Contra Costa' not in county_list:
    county_list.append('Contra Costa')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcounties(fig, ax[0,0], county_list, which='Confirmed', scale_population=False)
plotcounties(fig, ax[1,0], county_list, which='Deaths', scale_population=False)
plotcounties(fig, ax[0,1], county_list, which='Confirmed', scale_population=True, do_legend=True)
plotcounties(fig, ax[1,1], county_list, which='Deaths', scale_population=True)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.87, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_cases.png')
fig.show()

"""
fig, ax = plt.subplots(2, figsize=(7,8))

plotcounties(fig, ax[0], county_list, which='Confirmed', do_legend=True, day_zero_value=10)
plotcounties(fig, ax[1], county_list, which='Deaths', do_legend=False, day_zero_value=10)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.85, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_cases_shifted.png')
fig.show()
"""

fig, ax = plt.subplots(2, figsize=(7,8))

plotcounties(fig, ax[0], county_list, which='Confirmed', logderivative=True, do_legend=True)
plotcounties(fig, ax[1], county_list, which='Deaths', logderivative=True, do_legend=False)

fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
fig.text(0.80, 0.55, 'Top 10 per capita\nwith cases > 100,\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_doubling_rates.png')
fig.show()

