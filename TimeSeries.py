# data from https://github.com/CSSEGISandData/COVID-19

import os
import pandas
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import state_codes

matplotlib.interactive(True)

def smoother(n, nsmooth=1):
    for i in range(nsmooth):
        ncopy = np.zeros(len(n) + 2)
        ncopy[1:-1] = n
        n[:] = 0.25*(ncopy[:-2] + ncopy[2:]) + 0.5*ncopy[1:-1]
        n[0] *= 4./3.
        n[-1] *= 4./3.

class _TimeSeriesPlotter(object):
    def plot_region(self, ax, region, kw):
        scale_population = kw.get('scale_population', False)
        logderivative = kw.get('logderivative', False)
        derivative = kw.get('derivative', False)
        day_zero_value = kw.get('day_zero_value', None)
        start_date = kw.get('start_date', None)
        number_of_days = kw.get('number_of_days', None)
        nsmooth = kw.get('nsmooth', 15)
        doubling_days_max = kw.get('doubling_days_max', 40.)
        line_color = kw.get('line_color', None)

        cases, dates = self.data(region)

        if scale_population:
            population = int(self.populations[self.populations['Name'] == region]['Population'])
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
        elif number_of_days is not None:
            cases = cases[-number_of_days:]
            dates = dates[-number_of_days:]

        if line_color is None:
            ax.plot(dates, cases, label=region)
        else:
            ax.plot(dates, cases, line_color, label=region)

    def plot_regions(self, ax, region_list, **kw):
        scale_population = kw.get('scale_population', False)
        logderivative = kw.get('logderivative', False)
        derivative = kw.get('derivative', False)
        day_zero_value = kw.get('day_zero_value', None)
        do_legend = kw.get('do_legend', False)
        ylabel = kw.get('ylabel', None)

        for region in region_list:
            self.plot_region(ax, region, kw)

        if day_zero_value is None:
            # set so ~10 dates are shown on the x axis
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            ax.set_xlabel('Date')
        else:
            ax.set_xlabel('Days since start')

        if ylabel is None:
            ylabel = 'cumulative '+self.which
            if derivative:
                ylabel = f'new {self.which} per day'
            if scale_population:
                ylabel += ' per capita'
            if logderivative:
                ylabel = f'{self.which} doubling days'
        ax.set_ylabel(ylabel)
        ax.tick_params(right=True, labelright=False, which='both')

        ax.set_ylim(0.)

        if do_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    def plot_region_trajectory(self, ax, region, trajectory_days, kw):
        scale_population = kw.get('scale_population', False)
        nsmooth = kw.get('nsmooth', None)

        cases, dates = self.data(region)

        if scale_population:
            population = int(self.populations[self.populations['Name'] == region]['Population'])
            cases /= population

        ax.plot(cases[trajectory_days:], cases[trajectory_days:] - cases[:-trajectory_days], label=region)


    def plot_regions_trajectory(self, ax, region_list, **kw):
        scale_population = kw.get('scale_population', False)
        do_legend = kw.get('do_legend', False)
        xymin = kw.get('xymin', None)
        trajectory_days = kw.get('trajectory_days', 7)

        for region in region_list:
            self.plot_region_trajectory(ax, region, trajectory_days, kw)

        ax.set_xscale('log')
        ax.set_yscale('log')

        if xymin is not None:
            ax.set_xlim(xymin)
            ax.set_ylim(xymin)

        xlabel = f'Total {self.which}'
        ylabel = f'{self.which.capitalize()} last {trajectory_days} days'

        if scale_population:
            xlabel += ' per capita'
            ylabel += ' per capita'

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.tick_params(right=True, labelright=False, which='both')

        if do_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    def map_region_name(self, region):
        return region

    def scatter_plot(self, ax, region_list, since_days=None):
        current_cases_min = 10000000
        current_cases_max = 0
        current_cases_scaled_min = 0
        current_cases_scaled_max = 0
        for region in region_list:
            cases, dates = self.data(region)
            if since_days is None:
                current_cases = cases[-1]
            else:
                current_cases = cases[-1] - cases[-since_days-1]
            population = int(self.populations[self.populations['Name'] == region]['Population'])
            current_cases_scaled = current_cases/population
            name = self.map_region_name(region)
            if name is None:
                continue
            ax.text(current_cases, current_cases_scaled, name,
                    horizontalalignment = 'center',
                    verticalalignment = 'center')
            current_cases_min = min(current_cases_min, current_cases)
            current_cases_max = max(current_cases_max, current_cases)
            current_cases_scaled_max = max(current_cases_scaled_max, current_cases_scaled)
        current_cases_min = 10.**(int(np.log10(current_cases_min)))
        current_cases_max = 10.**(int(np.log10(current_cases_max))+1)
        ax.set_xlim(current_cases_min, current_cases_max)
        ax.set_ylim(current_cases_scaled_min, current_cases_scaled_max*1.1)
        ax.set_xlabel(f'{self.which.capitalize()}')
        ax.set_ylabel(f'{self.which.capitalize()} per capita')
        ax.set_xscale('log')

        ax.tick_params(right=True, labelright=False, which='both')

    def plot_regions_rate_change(self, ax, region_list, **kw):
        scale_population = kw.get('scale_population', True)
        trajectory_days = kw.get('trajectory_days', 7)

        cases_min = 1.e10
        cases_max = 0.

        for region in region_list:
            name = self.map_region_name(region)
            if name is None:
                continue
    
            cases, dates = self.data(region)
            last_week_cases = (cases[-1] - cases[-trajectory_days-1])/trajectory_days
            previous_week_cases = (cases[-trajectory_days-1] - cases[-2*trajectory_days-1])/trajectory_days

            if scale_population:
                population = int(self.populations[self.populations['Name'] == region]['Population'])
                last_week_cases /= population
                previous_week_cases /= population

            cases_min = min(last_week_cases, cases_min)
            cases_min = min(previous_week_cases, cases_min)
            cases_max = max(last_week_cases, cases_max)
            cases_max = max(previous_week_cases, cases_max)

            color = 'k'
            if region in ['US', 'California', 'Contra Costa', 'Alameda']:
                color = 'r'
            ax.text(last_week_cases, previous_week_cases, name,
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    color = color)

            if region in ['US', 'California', 'Contra Costa', 'Alameda', 'Spain']:
                t = trajectory_days
                last_week_cases = (cases[-t-1:] - cases[-2*t-1:-t])/t
                previous_week_cases = (cases[-2*t-1:-t] - cases[-3*t-1:-2*t])/t

                if scale_population:
                    last_week_cases /= population
                    previous_week_cases /= population

                ax.plot(last_week_cases, previous_week_cases, color)

        if not scale_population:
            cases_min = max(1., cases_min)
            cases_min = 10.**(int(np.log10(cases_min))  )
            cases_max = 10.**(int(np.log10(cases_max))+1)
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            base = 10.**(np.floor(np.log10(cases_max)))
            cases_max = base*(np.floor(5*cases_max/base) + 1)/5
            cases_min = 0.

        ax.set_xlim(cases_min, cases_max)
        ax.set_ylim(cases_min, cases_max)
        ax.plot([cases_min, cases_max], [cases_min, cases_max], 'r')

        ax.text(cases_max/2., cases_max*0.97, 'Improving',
                horizontalalignment = 'center',
                verticalalignment = 'center',
                color = 'r')
        ax.text(cases_max/2., cases_max*0.03, 'Worsening',
                horizontalalignment = 'center',
                verticalalignment = 'center',
                color = 'r')

        xlabel = f'Most recent week {self.which} per day'
        ylabel = f'Two weeks ago {self.which} per day'

        if scale_population:
            xlabel += ' per capita'
            ylabel += ' per capita'

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.tick_params(right=True, labelright=False, which='both')

class _TimeSeriesBase(_TimeSeriesPlotter):
    """Reads csv files from csse_covid_19_data/csse_covid_19_time_series
    - which: either confirmed or deaths
    - region: global or US
    - first_keys: number of keys before the dates
    """
    def __init__(self, which, region, first_keys):
        self.which = which
        self.region = region
        self.first_keys = first_keys
        self.root_dir = '../COVID-19/csse_covid_19_data/csse_covid_19_time_series'

        self.region_column = None

        self.filename = f'time_series_covid19_{which}_{region}.csv'
        self.dataframe = pandas.read_csv(os.path.join(self.root_dir, self.filename))
        self.dates = [datetime.strptime(d, "%m/%d/%y").date() for d in self.dataframe.keys().array[first_keys:]]

    def select_area(self):
        "Allows inheritors to down select based on other criteria"
        return self.dataframe

    def data(self, region):
        dataframe = self.select_area()
        df = dataframe[dataframe[self.region_column] == region]
        alldata = df.to_numpy()[:,self.first_keys:].astype(float)
        data = alldata.sum(axis=0)
        return data, self.dates

    def find_maxes(self, ncases=10, scale_population=False,  mincases=0, derivative=False, min_population=0):
        maxregions = []
        maxcases = []
        dataframe = self.select_area()
        for region in dataframe[self.region_column].unique():
            if region == 'Qatar':
                continue
            # Get cases on most recent day
            allcases, dates = self.data(region)
            if derivative:
                cases = allcases[-1] - allcases[-2]
            else:
                cases = allcases[-1]
            if allcases[-1] < mincases:
                continue
            population = self.populations[self.populations['Name'] == region]['Population']
            if len(population) == 0:
                continue
            population = float(population)
            if population < min_population:
                continue
            if scale_population:
                cases = cases/population
            if len(maxcases) < ncases:
                maxregions.append(region)
                maxcases.append(cases)
            elif cases > min(maxcases):
                ii = np.argmin(maxcases)
                maxregions[ii] = region
                maxcases[ii] = cases
        # sort in descending order
        ii = np.argsort(maxcases)[::-1]
        result = []
        for i in ii:
            result.append(maxregions[i])
        return result


class TimeSeriesCountries(_TimeSeriesBase):
    """Handles time series data for countries
    - which: either confirmed or deaths
    """
    def __init__(self, which):
        _TimeSeriesBase.__init__(self, which, region='global', first_keys=4)
        self.region_column = 'Country/Region'
        self.populations = pandas.read_csv('country_populations.csv')

    def data(self, region):
        if region == 'World':
            dataframe = self.select_area()
            alldata = dataframe.to_numpy()[:,self.first_keys:].astype(float)
            data = alldata.sum(axis=0)
            return data, self.dates
        else:
            return _TimeSeriesBase.data(self, region)


class TimeSeriesStates(_TimeSeriesBase):
    """Handles time series data for states
    - which: either confirmed or deaths
    """
    def __init__(self, which):
        first_key = 11
        if which == 'deaths':
            # This data has an extra column, the population
            first_key = 12
        _TimeSeriesBase.__init__(self, which, region='US', first_keys=first_key)
        self.region_column = 'Province_State'
        self.populations = pandas.read_csv('state_populations.csv')

    def map_region_name(self, region):
        try:
            result = state_codes.state_codes[region]
        except KeyError:
            result = None
        return result

class TimeSeriesCounties(_TimeSeriesBase):
    """Handles time series data for counties
    - which: either confirmed or deaths
    - state='California': which state to select counties from
    """
    def __init__(self, which, state='California'):
        first_key = 11
        if which == 'deaths':
            # This data has an extra column, the population
            first_key = 12
        _TimeSeriesBase.__init__(self, which, region='US', first_keys=first_key)
        self.region_column = 'Admin2'
        self.state = state
        self.populations = pandas.read_csv('county_populations.csv')

    def select_area(self):
        return self.dataframe[self.dataframe['Province_State'] == self.state]
