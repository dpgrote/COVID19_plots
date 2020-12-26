# data from https://github.com/CSSEGISandData/COVID-19

import datetime
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from plotdata_covid19_functions import *

from TimeSeries import TimeSeriesCountries, TimeSeriesStates, TimeSeriesCounties

#matplotlib.interactive(True)

countries_data = {'confirmed':TimeSeriesCountries(which='confirmed'),
                  'deaths':TimeSeriesCountries(which='deaths')}
states_data    = {'confirmed':TimeSeriesStates(which='confirmed'),
                  'deaths':TimeSeriesStates(which='deaths')}
counties_data  = {'confirmed':TimeSeriesCounties(which='confirmed'),
                  'deaths':TimeSeriesCounties(which='deaths')}

#import pdb
#pdb.set_trace()

####################### Countries

country_list = countries_data['confirmed'].find_maxes(scale_population=True, derivative=False, ncases=30, min_population=1.e7)

countries_data['confirmed'].plot_regions_rate_change_animate(country_list, scale_population=True,
        cases_min = 0.,
        cases_max = 0.00026)


####################### States

state_list = states_data['confirmed'].find_maxes(scale_population=True, derivative=False, ncases=30)

states_data['confirmed'].plot_regions_rate_change_animate(state_list, scale_population=True,
        cases_min = 0.,
        cases_max = 0.0005)


####################### Counties

county_list = counties_data['confirmed'].find_maxes(scale_population=True, derivative=False, ncases=50)

counties_data['confirmed'].plot_regions_rate_change_animate(county_list, scale_population=True,
        cases_min = 0.,
        cases_max = 0.0004)

