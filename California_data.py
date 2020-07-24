from urllib import request
import pandas
from datetime import datetime
import numpy as np

from TimeSeries import _TimeSeriesPlotter

data_url = 'https://data.chhs.ca.gov/api/3/action/datastore_search?resource_id=6cd8d424-dfaa-4bdd-9410-a3d656e1176e&limit=100000'

class CaliforniaData(_TimeSeriesPlotter):
    def __init__(self):
        # Fetch data from web site
        fileobj = request.urlopen(data_url)

        # Extract the results as a pandas dataframe
        full_page = pandas.read_json(fileobj.read())
        data_dict = full_page['result']['records']
        self.dataframe = pandas.DataFrame(data_dict)

        self.populations = pandas.read_csv('county_populations.csv')

    def data(self, county):
        # which: one of:
        #   Total Count Confirmed
        #   Total Count Deaths
        #   COVID-19 Positive Patients
        #   Suspected COVID-19 Positive Patients
        #   ICU COVID-19 Positive Patients
        #   ICU COVID-19 Suspected Patients
        df = self.dataframe[self.dataframe['County Name'] == county]
        dates = [datetime.strptime(d, "%Y-%m-%dT00:00:00").date() for d in df['Most Recent Date'].array]
        results = df[self.which].to_numpy(dtype=float)
        return results, dates

    def plot_regions(ax, region_list, which, **kw):
        # Save "which" so self.data has access to it. Kind of kludgy.
        self.which = which
        _TimeSeriesPlotter.plot_regions(self, ax, region_list, **kw)

    def find_maxes(self, which, ncases=10, scale_population=False, mincases=0):
        maxcounties = []
        maxcases = []
        for county in self.dataframe['County Name'].unique():
            # Get cases on most recent day
            allcases, dates = self.data(county, which)
            cases = allcases[-1]
            if cases < mincases:
                continue
            if scale_population:
                population = self.populations[self.populations['Name'] == county]['Population']
                if len(population) == 0:
                    continue
                population = float(population)
                cases = cases/population
            if len(maxcases) < ncases:
                maxcounties.append(county)
                maxcases.append(cases)
            elif cases > min(maxcases):
                ii = np.argmin(maxcases)
                maxcounties[ii] = county
                maxcases[ii] = cases
        # sort in descending order
        ii = np.argsort(maxcases)[::-1]
        result = []
        for i in ii:
            result.append(maxcounties[i])
        return result


if __name__ == '__main__':
    cdata = CaliforniaData()
