# data from https://github.com/CSSEGISandData/COVID-19

from plotdata_covid19_functions import *

from California_data import CaliforniaData

california_data = CaliforniaData()

# The format for the dates keeps changing
# -------- with hospitalization counts
mincases = 100
county_list = california_data.find_maxes(which='COVID-19 Positive Patients', scale_population=True, mincases=50)
if 'Contra Costa' not in county_list:
    county_list.append('Contra Costa')

fig, ax = plt.subplots(2, 2, figsize=(12,8))

plotcounties(ax[0,0], county_list, dataframe=california_data, which='COVID-19 Positive Patients', scale_population=True, ylabel='New confirmed patients per capita', nsmooth=4)
plotcounties(ax[1,0], county_list, dataframe=california_data, which='ICU COVID-19 Positive Patients', scale_population=True, ylabel='New confirmed ICU patients per capita', nsmooth=4)
plotcounties(ax[0,1], county_list, dataframe=california_data, which='Suspected COVID-19 Positive Patients', scale_population=True, do_legend=True, ylabel='New suspected patients per capita', nsmooth=4)
plotcounties(ax[1,1], county_list, dataframe=california_data, which='ICU COVID-19 Suspected Patients', scale_population=True, ylabel='New suspected ICU patients per capita', nsmooth=4)

# set nice formatting and centering for dates
fig.autofmt_xdate()
fig.tight_layout()
fig.subplots_adjust(bottom=.125)

fig.suptitle('data from https://data.chhs.ca.gov', y=0.02)
fig.text(0.87, 0.58, f'Top 10 per capita\nwith cases > {mincases},\nplus others')
fig.savefig('../../Dropbox/Public/COVID19/county_hospitalization.png')
fig.show()
