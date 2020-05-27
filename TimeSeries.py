# data from https://github.com/CSSEGISandData/COVID-19

import os
import pandas
import numpy as np
from datetime import datetime

class _TimeSeriesBase(object):
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

    def data(self, region, which=None):
        dataframe = self.select_area()
        df = dataframe[dataframe[self.region_column] == region]
        alldata = df.to_numpy()[:,self.first_keys:].astype(float)
        data = alldata.sum(axis=0)
        return data, self.dates

    def find_maxes(self, ncases=10, population_df=None, mincases=0):
        maxregions = []
        maxcases = []
        dataframe = self.select_area()
        for region in dataframe[self.region_column].unique():
            if region == 'Qatar':
                continue
            # Get cases on most recent day
            allcases, dates = self.data(region)
            cases = allcases[-1]
            if cases < mincases:
                continue
            if population_df is not None:
                population = population_df[population_df['Name'] == region]['Population']
                if len(population) == 0:
                    continue
                population = float(population)
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

    def data(self, region, which=None):
        if region == 'World':
            dataframe = self.select_area()
            alldata = dataframe.to_numpy()[:,self.first_keys:].astype(float)
            data = alldata.sum(axis=0)
            return data, self.dates
        else:
            return _TimeSeriesBase.data(self, region, which)


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

    def select_area(self):
        return self.dataframe[self.dataframe['Province_State'] == self.state]
