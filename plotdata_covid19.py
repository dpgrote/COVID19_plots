# data from https://github.com/CSSEGISandData/COVID-19

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

matplotlib.interactive(True)

#which = 'deaths'
#which = 'confirmed'
#which = 'recovered'

countrylist = ['France',
               'Italy',
               'US',
               'Spain',
               'Germany',
               'China',
               'Korea, South',
               'India',
               'Russia',
               'United Kingdom',
               'Japan']

countieslist = ['Contra Costa County, CA',
                'San Francisco County, CA',
                'Alameda County, CA',
                'Marin, CA',
                'Santa Clara County, CA',
                'San Mateo, CA',
                'Santa Cruz, CA',
                'Sacramento County, CA']

df_dict = {'confirmed': pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'),
           'deaths': pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'),
           'recovered': pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')}

dates_dict = {}
for k, v in df_dict.items():
    dates_dict[k] = v.keys()[4:].to_numpy()

country_populations = pd.read_csv('../country_populations.csv')
state_populations = pd.read_csv('../state_populations.csv')

#import pdb
#pdb.set_trace()

def smoother(n, nsmooth=1):
    for i in range(nsmooth):
        ncopy = np.zeros(len(n) + 2)
        ncopy[1:-1] = n
        n[:] = 0.25*(ncopy[:-2] + ncopy[2:]) + 0.5*ncopy[1:-1]
        n[0] *= 4./3.
        n[-1] *= 4./3.

def plotcountry(ax, country='France', which='confirmed', scale_population=False, scale_landarea=False, logderivative=False):

    df = df_dict[which]
    dates = dates_dict[which]

    cases = np.zeros(len(dates))
    for index, row_data in df[df['Country/Region'] == country].iterrows():
        cases += row_data[4:].to_numpy(dtype=float)

    if scale_population:
        population = int(country_populations[country_populations['Name'] == country]['Population'])
        cases /= population

    if scale_landarea:
        landarea = int(country_populations[country_populations['Name'] == country]['Land_Area'])
        cases *= landarea

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

    ax.plot(dates, cases, label=country)


def makeplot(fig, ax, which='confirmed', scale_population=False, scale_landarea=False, do_legend=False, logderivative=False):

    for country in countrylist:
        plotcountry(ax, country, which, scale_population, scale_landarea, logderivative)

    fig.autofmt_xdate()

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    if logderivative:
        ax.set_ylim(0.)
    else:
        ax.set_yscale('log')

    fig.tight_layout()
    ax.set_xlabel('Date')
    ylabel = '# '+which
    if scale_population:
        ylabel += '/pop'
    if scale_landarea:
        ylabel += '*area'
    if logderivative:
        ylabel = 'doubling days ' + which
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False)

    if do_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


fig, ax = plt.subplots(2, 2, figsize=(12,8))

makeplot(fig, ax[0,0], which='confirmed', scale_population=False, scale_landarea=False)
makeplot(fig, ax[1,0], which='deaths', scale_population=False, scale_landarea=False)
makeplot(fig, ax[0,1], which='confirmed', scale_population=True, scale_landarea=False, do_legend=True)
makeplot(fig, ax[1,1], which='deaths', scale_population=True, scale_landarea=False)

fig.savefig('/Users/davidgrote/Dropbox/Public/COVID19/country_cases.png')
fig.show()


def plotmaxstates(fig, ax, which='confirmed', nstates=5, scale_population=False, logderivative=False):

    df = df_dict[which]
    dates = dates_dict[which]

    if logderivative:
        dates = dates[1:-1]

    top_n_states = []
    top_n_cases = []
    for index, row_data in df[df['Country/Region'] == 'US'].iterrows():
        state = row_data[0]
        if isinstance(state, str) and state.find(',') > 0:
            # skip county data sets
            # These could be added to the state cases, but that would be messy since it needs the two letter state codes.
            # The counties are not being updated anyway.
            continue
        most_recent_cases = float(row_data[-1])
        if scale_population:
            population = state_populations[state_populations['State'] == state]['Population estimate July 1 2019']
            if len(population) == 0:
                # State wasn't found in state_populations
                continue
            most_recent_cases /= float(population)
        if len(top_n_states) <= nstates:
            top_n_states.append(state)
            top_n_cases.append(most_recent_cases)
        else:
            if most_recent_cases > np.min(top_n_cases):
                ii = np.argmin(top_n_cases)
                top_n_states[ii] = state
                top_n_cases[ii] = most_recent_cases

    for state in top_n_states:
        state_data = df[df['Province/State'] == state]

        cases = state_data.to_numpy()[0,4:]

        if scale_population:
            population = float(state_populations[state_populations['State'] == state]['Population estimate July 1 2019'])
            cases /= population

        ii = (cases > 0)

        if logderivative:
            if cases.max() == 0:
                cases[...] = 1
            casesmin = (cases[cases>0]).min()
            log10cases = np.log10(cases.clip(casesmin).astype(float))
            log2cases = log10cases/np.log10(2.)
            cases = (log2cases[2:] - log2cases[:-2])/2.
            cases = 1./cases.clip(0.1)
            smoother(cases, nsmooth=5)
            ii = ii[1:-1]

        ax.plot(dates[ii], cases[ii], label=state)

    fig.autofmt_xdate()

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    if logderivative:
        ax.set_ylim(0.)
    else:
        ax.set_yscale('log')

    fig.tight_layout()
    ax.set_xlabel('Date')
    ylabel = '# '+which
    if scale_population:
        ylabel += '/pop'
    if logderivative:
        ylabel = 'doubling days ' + which
    ax.set_ylabel(ylabel)
    ax.tick_params(right=True, labelright=False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotmaxstates(fig, ax[0,0], which='confirmed', nstates=10)
plotmaxstates(fig, ax[1,0], which='deaths', nstates=10)
plotmaxstates(fig, ax[0,1], which='confirmed', nstates=10, scale_population=True)
plotmaxstates(fig, ax[1,1], which='deaths', nstates=10, scale_population=True)

fig.savefig('/Users/davidgrote/Dropbox/Public/COVID19/state_cases.png')
fig.show()

fig, ax = plt.subplots(2, 2, figsize=(12,8))

makeplot(fig, ax[0,0], which='confirmed', scale_population=False, scale_landarea=False, logderivative=True, do_legend=True)
makeplot(fig, ax[1,0], which='deaths', scale_population=False, scale_landarea=False, logderivative=True, do_legend=False)
plotmaxstates(fig, ax[0,1], which='confirmed', nstates=10, logderivative=True)
plotmaxstates(fig, ax[1,1], which='deaths', nstates=10, logderivative=True)

fig.savefig('/Users/davidgrote/Dropbox/Public/COVID19/doubling_rates.png')
fig.show()


def plotcounty(ax, county='Contra Costa', which='confirmed'):

    df = df_dict[which]
    dates = dates_dict[which]

    county_data = df[df['Province/State'] == county]

    cases = county_data.to_numpy()[0,4:]
    ii = (cases > 0)
    if any(ii):
        ax.plot(dates[ii], cases[ii], label=county)

def countiesplot(fig, ax, which):

    for county in countieslist:
        plotcounty(ax, county, which)

    fig.autofmt_xdate()

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_yscale('log')

    #fig.tight_layout()
    ax.set_xlabel('Date')
    ylabel = '# '+which
    ax.set_ylabel(ylabel)

    ax.legend()

"""
fig, ax = plt.subplots(1, 2, figsize=(12,8))

countiesplot(fig, ax[0], which='confirmed')
countiesplot(fig, ax[1], which='deaths')

fig.show()
"""
