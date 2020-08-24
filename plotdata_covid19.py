# data from https://github.com/CSSEGISandData/COVID-19

import datetime
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from plotdata_covid19_functions import *

from TimeSeries import TimeSeriesCountries, TimeSeriesStates, TimeSeriesCounties
from California_data import CaliforniaData

matplotlib.interactive(True)

#which = 'deaths'
#which = 'confirmed'
#which = 'recovered'

plot_trajectories = False
plot_death_rates = False
plot_doubling_rates = False
plot_scatter_plots = False
plot_time_shifted = False

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

california_data = CaliforniaData()

country_populations = pandas.read_csv('country_populations.csv')
state_populations = pandas.read_csv('state_populations.csv')
county_populations = pandas.read_csv('county_populations.csv')

#import pdb
#pdb.set_trace()

def plot_finish(fig, png_name):
    fig.tight_layout()
    fig.subplots_adjust(bottom=.125)
    fig.suptitle('data from https://github.com/CSSEGISandData/COVID-19', y=0.02)
    fig.savefig(f'../../Dropbox/Public/COVID19/{png_name}')
    fig.show()


####################### Countries
mincases = 60000
mindeaths = 1500
country_list_confirmed = countries_data['confirmed'].find_maxes(scale_population=True, mincases=mincases, derivative=True)
for country in ['US', 'China', 'Korea, South', 'Sweden']:
    if country not in country_list_confirmed:
        country_list_confirmed.append(country)

country_list_deaths = countries_data['deaths'].find_maxes(scale_population=True, mincases=mindeaths, derivative=True)
for country in ['US', 'China', 'Korea, South', 'Sweden']:
    if country not in country_list_deaths:
        country_list_deaths.append(country)

fig, ax = plt.subplots(2, 2, figsize=(12,8))

countries_data['confirmed'].plot_regions(ax[0,0], country_list_confirmed, scale_population=False)
ax[0,0].set_ylim(1.e4)
ax[0,0].set_yscale('log')
countries_data['deaths'].plot_regions(ax[1,0], country_list_deaths, scale_population=False)
ax[1,0].set_ylim(1.e2)
ax[1,0].set_yscale('log')
countries_data['confirmed'].plot_regions(ax[0,1], country_list_confirmed, scale_population=True, do_legend=False)
countries_data['confirmed'].plot_regions(ax[0,1], ['World'], scale_population=True, do_legend=True, line_color='k--')
countries_data['deaths'].plot_regions(ax[1,1], country_list_deaths, scale_population=True, do_legend=False)
countries_data['deaths'].plot_regions(ax[1,1], ['World'], scale_population=True, do_legend=True, line_color='k--')

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.text(0.87, 0.55, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
fig.text(0.87, 0.09, f'Top 10 per capita\nwith deaths > {mindeaths},\nplus others')
plot_finish(fig, 'country_cases.png')

if plot_trajectories:
    fig, ax = plt.subplots(2, 2, figsize=(12,8))

    countries_data['confirmed'].plot_regions_trajectory(ax[0,0], country_list_confirmed, scale_population=False, xymin=100.)
    countries_data['deaths'].plot_regions_trajectory(ax[1,0], country_list_deaths, scale_population=False, xymin=100.)
    countries_data['confirmed'].plot_regions_trajectory(ax[0,1], country_list_confirmed, scale_population=True, xymin=1.e-6, do_legend=True)
    countries_data['deaths'].plot_regions_trajectory(ax[1,1], country_list_deaths, scale_population=True, xymin=1.e-6, do_legend=True)

    fig.text(0.87, 0.55, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    fig.text(0.87, 0.09, f'Top 10 per capita\nwith deaths > {mindeaths},\nplus others')
    plot_finish(fig, 'country_trajectories.png')

if plot_time_shifted:
    fig, ax = plt.subplots(2, 2, figsize=(12,8))

    countries_data['confirmed'].plot_regions(ax[0,0], country_list_confirmed, scale_population=False, day_zero_value=100)
    countries_data['deaths'].plot_regions(ax[1,0], country_list_deaths, scale_population=False, day_zero_value=100)
    countries_data['confirmed'].plot_regions(ax[0,1], country_list_confirmed, scale_population=True, do_legend=True, day_zero_value=1.e-6)
    countries_data['deaths'].plot_regions(ax[1,1], country_list_deaths, scale_population=True, do_legend=True, day_zero_value=1.e-6)

    fig.text(0.87, 0.55, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    fig.text(0.87, 0.09, f'Top 10 per capita\nwith deaths > {mindeaths},\nplus others')
    plot_finish(fig, 'country_cases_shifted.png')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

countries_data['confirmed'].plot_regions(ax[0,0], country_list_confirmed, scale_population=False, derivative=True, do_legend=False)
ax[0,0].set_ylim(1.)
ax[0,0].set_yscale('log')
countries_data['deaths'].plot_regions(ax[1,0], country_list_deaths, scale_population=False, derivative=True, do_legend=False)
ax[1,0].set_ylim(1.)
ax[1,0].set_yscale('log')
countries_data['confirmed'].plot_regions(ax[0,1], country_list_confirmed, scale_population=True, derivative=True, do_legend=False)
countries_data['confirmed'].plot_regions(ax[0,1], ['World'], scale_population=True, derivative=True, do_legend=True, line_color='k--')
ax[0,1].set_ylim(0., 0.0003)
countries_data['deaths'].plot_regions(ax[1,1], country_list_deaths, scale_population=True, derivative=True, do_legend=False)
countries_data['deaths'].plot_regions(ax[1,1], ['World'], scale_population=True, derivative=True, do_legend=True, line_color='k--')
ax[1,1].set_ylim(None, 1.e-5)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.text(0.87, 0.55, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
fig.text(0.87, 0.09, f'Top 10 per capita\nwith deaths > {mindeaths},\nplus others')
plot_finish(fig, 'country_cases_per_day.png')

if plot_doubling_rates:
    fig, ax = plt.subplots(2, figsize=(7,8))

    countries_data['confirmed'].plot_regions(ax[0], country_list_confirmed, scale_population=False, logderivative=True, do_legend=True)
    countries_data['deaths'].plot_regions(ax[1], country_list_deaths, scale_population=False, logderivative=True, do_legend=True)

    # set nice formatting and centering for dates
    fig.autofmt_xdate()
    fig.text(0.77, 0.55, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    fig.text(0.77, 0.09, f'Top 10 per capita\nwith deaths > {mindeaths},\nplus others')
    plot_finish(fig, 'country_doubling_rates.png')

# Plot change of rate
country_list = countries_data['confirmed'].find_maxes(scale_population=True, derivative=False, ncases=30)
fig, ax = plt.subplots(1, figsize=(12,8))
countries_data['confirmed'].plot_regions_rate_change(ax, country_list, scale_population=True)
plot_finish(fig, 'country_change_of_rate_confirmed_plot.png')
fig, ax = plt.subplots(1, figsize=(12,8))
countries_data['deaths'].plot_regions_rate_change(ax, country_list, scale_population=True)
plot_finish(fig, 'country_change_of_rate_deaths_plot.png')


####################### States
mincases = 1000
state_list = states_data['confirmed'].find_maxes(scale_population=True, mincases=mincases, derivative=True)
if 'California' not in state_list:
    state_list.append('California')
if 'Georgia' not in state_list:
    state_list.append('Georgia')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

states_data['confirmed'].plot_regions(ax[0,0], state_list, scale_population=False)
ax[0,0].set_ylim(1.e3)
ax[0,0].set_yscale('log')
states_data['deaths'].plot_regions(ax[1,0], state_list, scale_population=False)
ax[1,0].set_ylim(1.e2)
ax[1,0].set_yscale('log')
states_data['confirmed'].plot_regions(ax[0,1], state_list, scale_population=True, do_legend=False)
countries_data['confirmed'].plot_regions(ax[0,1], ['US'], scale_population=True, do_legend=True, line_color='k--')
#ax[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
states_data['deaths'].plot_regions(ax[1,1], state_list, scale_population=True)
countries_data['deaths'].plot_regions(ax[1,1], ['US'], scale_population=True, line_color='k--')

# set nice formatting and centering for dates
fig.autofmt_xdate()

fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
plot_finish(fig, 'state_cases.png')

if plot_trajectories:
    fig, ax = plt.subplots(2, 2, figsize=(12,8))

    states_data['confirmed'].plot_regions_trajectory(ax[0,0], state_list, scale_population=False, xymin=100)
    states_data['deaths'].plot_regions_trajectory(ax[1,0], state_list, scale_population=False, xymin=100)
    states_data['confirmed'].plot_regions_trajectory(ax[0,1], state_list, scale_population=True, do_legend=True, xymin=1.e-6)
    states_data['deaths'].plot_regions_trajectory(ax[1,1], state_list, scale_population=True, xymin=1.e-6)


    fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    plot_finish(fig, 'state_trajectories.png')

if plot_time_shifted:
    fig, ax = plt.subplots(2, 2, figsize=(12,8))

    states_data['confirmed'].plot_regions(ax[0,0], state_list, scale_population=False, day_zero_value=20)
    states_data['deaths'].plot_regions(ax[1,0], state_list, scale_population=False, day_zero_value=20)
    states_data['confirmed'].plot_regions(ax[0,1], state_list, scale_population=True, do_legend=True, day_zero_value=5.e-6)
    states_data['deaths'].plot_regions(ax[1,1], state_list, scale_population=True, day_zero_value=5.e-6)


    fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    plot_finish(fig, 'state_cases_shifted.png')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

states_data['confirmed'].plot_regions(ax[0,0], state_list, scale_population=False, derivative=True, do_legend=False)
ax[0,0].set_ylim(1.e1)
ax[0,0].set_yscale('log')
states_data['deaths'].plot_regions(ax[1,0], state_list, scale_population=False, derivative=True, do_legend=False)
ax[1,0].set_ylim(1.)
ax[1,0].set_yscale('log')
states_data['confirmed'].plot_regions(ax[0,1], state_list, scale_population=True, derivative=True, do_legend=False)
countries_data['confirmed'].plot_regions(ax[0,1], ['US'], scale_population=True, derivative=True, do_legend=True, line_color='k--')
states_data['deaths'].plot_regions(ax[1,1], state_list, scale_population=True, derivative=True, do_legend=False)
countries_data['deaths'].plot_regions(ax[1,1], ['US'], scale_population=True, derivative=True, do_legend=False, line_color='k--')

# set nice formatting and centering for dates
fig.autofmt_xdate()

fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
plot_finish(fig, 'state_cases_per_day.png')

if plot_doubling_rates:
    fig, ax = plt.subplots(2, figsize=(7,8))

    states_data['confirmed'].plot_regions(ax[0], state_list, scale_population=False, logderivative=True, do_legend=True)
    states_data['deaths'].plot_regions(ax[1], state_list, scale_population=False, logderivative=True, do_legend=False)

    # set nice formatting and centering for dates
    fig.autofmt_xdate()

    fig.text(0.77, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    plot_finish(fig, 'state_doubling_rates.png')


if plot_scatter_plots:
    state_list = states_data['confirmed'].find_maxes(scale_population=True, derivative=False, ncases=30)
    # Scatter plot
    fig, ax = plt.subplots(2, figsize=(7,8))
    states_data['confirmed'].scatter_plot(ax[0], state_list)
    states_data['deaths'].scatter_plot(ax[1], state_list)
    plot_finish(fig, 'state_scatter_plot.png')

    fig, ax = plt.subplots(2, figsize=(7,8))
    states_data['confirmed'].scatter_plot(ax[0], state_list, since_days=14)
    states_data['deaths'].scatter_plot(ax[1], state_list, since_days=14)
    fig.suptitle('In the last 14 days', y=0.04)
    plot_finish(fig, 'state_scatter_plot_14days.png')

# Plot change of rate
state_list = states_data['confirmed'].find_maxes(scale_population=True, derivative=False, ncases=30)
fig, ax = plt.subplots(1, figsize=(12,8))
states_data['confirmed'].plot_regions_rate_change(ax, state_list, scale_population=True)
plot_finish(fig, 'state_change_of_rate_confirmed_plot.png')
fig, ax = plt.subplots(1, figsize=(12,8))
states_data['deaths'].plot_regions_rate_change(ax, state_list, scale_population=True)
plot_finish(fig, 'state_change_of_rate_deaths_plot.png')


####################### Counties
mincases = 100
#county_list = counties_data['confirmed'].find_maxes(scale_population=True, mincases=mincases, derivative=True)
county_list = counties_data['confirmed'].find_maxes(mincases=mincases, derivative=True)
if 'Contra Costa' not in county_list:
    county_list.append('Contra Costa')
if 'Alameda' not in county_list:
    county_list.append('Alameda')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

counties_data['confirmed'].plot_regions(ax[0,0], county_list, scale_population=False)
ax[0,0].set_ylim(1.e2)
ax[0,0].set_yscale('log')
counties_data['deaths'].plot_regions(ax[1,0], county_list, scale_population=False)
ax[1,0].set_ylim(1.)
ax[1,0].set_yscale('log')
counties_data['confirmed'].plot_regions(ax[0,1], county_list, scale_population=True, do_legend=False)
states_data['confirmed'].plot_regions(ax[0,1], ['California'], scale_population=True, do_legend=True, line_color='k--', start_date=datetime.date(2020, 3, 20))
counties_data['deaths'].plot_regions(ax[1,1], county_list, scale_population=True)
states_data['deaths'].plot_regions(ax[1,1], ['California'], scale_population=True, line_color='k--',start_date=datetime.date(2020, 3, 20))

# set nice formatting and centering for dates
fig.autofmt_xdate()

fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
plot_finish(fig, 'county_cases.png')

if plot_trajectories:
    fig, ax = plt.subplots(2, 2, figsize=(12,8))

    counties_data['confirmed'].plot_regions_trajectory(ax[0,0], county_list, scale_population=False, xymin=1)
    counties_data['deaths'].plot_regions_trajectory(ax[1,0], county_list, scale_population=False, xymin=1)
    counties_data['confirmed'].plot_regions_trajectory(ax[0,1], county_list, scale_population=True, do_legend=True, xymin=1.e-6)
    counties_data['deaths'].plot_regions_trajectory(ax[1,1], county_list, scale_population=True, xymin=1.e-6)


    fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    plot_finish(fig, 'county_trajectories.png')

if plot_time_shifted:
    fig, ax = plt.subplots(2, figsize=(7,8))

    counties_data['confirmed'].plot_regions(ax[0], county_list, do_legend=True, day_zero_value=10)
    counties_data['deaths'].plot_regions(ax[1], county_list, do_legend=False, day_zero_value=10)


    fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    plot_finish(fig, 'county_cases_shifted.png')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

counties_data['confirmed'].plot_regions(ax[0,0], county_list, scale_population=False, derivative=True, do_legend=False)
ax[0,0].set_ylim(1.)
ax[0,0].set_yscale('log')
counties_data['deaths'].plot_regions(ax[1,0], county_list, scale_population=False, derivative=True, do_legend=False)
ax[1,0].set_ylim(1.)
ax[1,0].set_yscale('log')
counties_data['confirmed'].plot_regions(ax[0,1], county_list, scale_population=True, derivative=True, do_legend=False)
states_data['confirmed'].plot_regions(ax[0,1], ['California'], scale_population=True, derivative=True, do_legend=True, line_color='k--', start_date=datetime.date(2020, 3, 20))
#ax[0,1].set_ylim(None, 0.0004)
counties_data['deaths'].plot_regions(ax[1,1], county_list, scale_population=True, derivative=True, do_legend=False)
states_data['deaths'].plot_regions(ax[1,1], ['California'], scale_population=True, derivative=True, do_legend=False, line_color='k--', start_date=datetime.date(2020, 3, 20))
ax[1,1].set_ylim(None, 8.e-6)

# set nice formatting and centering for dates
fig.autofmt_xdate()

fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
plot_finish(fig, 'county_cases_per_day.png')

if plot_doubling_rates:
    fig, ax = plt.subplots(2, figsize=(7,8))

    counties_data['confirmed'].plot_regions(ax[0], county_list, logderivative=True, do_legend=True)
    counties_data['deaths'].plot_regions(ax[1], county_list, logderivative=True, do_legend=False)

    # set nice formatting and centering for dates
    fig.autofmt_xdate()

    fig.text(0.77, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
    plot_finish(fig, 'county_doubling_rates.png')


# Plot change of rate
county_list = counties_data['confirmed'].find_maxes(scale_population=True, derivative=False, ncases=50)
#print(county_list)
fig, ax = plt.subplots(1, figsize=(12,8))
counties_data['confirmed'].plot_regions_rate_change(ax, county_list, scale_population=True)
plot_finish(fig, 'county_change_of_rate_confirmed_plot.png')
fig, ax = plt.subplots(1, figsize=(12,8))
counties_data['deaths'].plot_regions_rate_change(ax, county_list, scale_population=True)
plot_finish(fig, 'county_change_of_rate_deaths_plot.png')

"""
# The format for the dates keeps changing
# -------- with hospitalization counts
county_list = california_data.find_maxes(which='COVID-19 Positive Patients', scale_population=True, mincases=50)
if 'Contra Costa' not in county_list:
    county_list.append('Contra Costa')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

california_data.plot_regions(ax[0,0], county_list, which='COVID-19 Positive Patients', scale_population=True, ylabel='New confirmed patients per capita')
california_data.plot_regions(ax[1,0], county_list, which='ICU COVID-19 Positive Patients', scale_population=True, ylabel='New confirmed ICU patients per capita')
california_data.plot_regions(ax[0,1], county_list, which='Suspected COVID-19 Positive Patients', scale_population=True, do_legend=True, ylabel='New suspected patients per capita')
california_data.plot_regions(ax[1,1], county_list, which='ICU COVID-19 Suspected Patients', scale_population=True, ylabel='New suspected ICU patients per capita')

# set nice formatting and centering for dates
fig.autofmt_xdate()

fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
plot_finish(fig, 'county_hospitalization.png')
"""

####################### Death rates

if plot_death_rates:
    fig, ax = plt.subplots(3, figsize=(12,8))

    plotcountries_delayed_death_rates(ax[0], country_list_deaths, countries_data, do_legend=True)
    plotstates_delayed_death_rates(ax[1], state_list, states_data, do_legend=True, ymax=0.1)
    plotcounties_delayed_death_rates(ax[2], county_list, counties_data, do_legend=True, ymax=0.1)


    if delay > 0:
        fig.suptitle(f'Confirmed cases lagged by {delay} days', y=0.96)
    plot_finish(fig, 'delayed_death_rates.png')

