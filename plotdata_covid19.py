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

df_dict = {'confirmed': pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'),
           'deaths': pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'),
           'recovered': pd.read_csv('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')}

dates_dict = {}
for k, v in df_dict.items():
    dates_dict[k] = v.keys()[4:].to_numpy()

country_populations = pd.read_csv('../country_populations.csv')

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

fig, ax = plt.subplots(2, figsize=(7,8))

makeplot(fig, ax[0], which='confirmed', scale_population=False, scale_landarea=False, logderivative=True, do_legend=True)
makeplot(fig, ax[1], which='deaths', scale_population=False, scale_landarea=False, logderivative=True, do_legend=False)

fig.savefig('/Users/davidgrote/Dropbox/Public/COVID19/doubling_rates.png')
fig.show()
