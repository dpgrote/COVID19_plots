# data from https://github.com/CSSEGISandData/COVID-19

import os
import pandas
import numpy as np
from datetime import datetime

import state_codes

class DailyReports(object):
    def __init__(self, reports_directory='../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports'):
        self.reports_directory = reports_directory

        files = os.listdir(self.reports_directory)
        files.sort()

        self.reports = []
        self.dates = []
        for f in files:
            if f.endswith('.csv'):
                report = pandas.read_csv(os.path.join(self.reports_directory, f))
                self.reports.append(report)
                # Rename columns from the older format
                if 'Province/State' in report.columns:
                    self.fix_old_column_names(report)
                #self.fix_dates(report)
                self.fix_country_names(report)
                self.dates.append(f[0:2] + '/' + f[3:5] + '/' + f[8:10])
        self.dates = [datetime.strptime(d, "%m/%d/%y") for d in self.dates]

    def fix_old_column_names(self, report):
        report.rename(columns = {'Country/Region':'Country_Region',
                                 'Last Update':'Last_Update'},
                      inplace = True)
        # split up the Province/State
        counties = []
        states = []
        for ps in report['Province/State']:
            county = np.nan
            state = np.nan
            if isinstance(ps, str):
                ss = ps.split(', ')
                if len(ss) == 2:
                    # For names like 'Contra Costa County, CA'
                    county = ss[0]
                    if county.endswith(' County'):
                        county = county[:-7]
                    state = state_codes.state_codes_reverse.get(ss[1], ss[1])
                elif len(ss) == 1:
                    state = ss[0]
            counties.append(county)
            states.append(state)
        report['Province_State'] = states
        report['Admin2'] = counties

    def fix_dates(self, report):
        # Fixes up the dates, using the format MM/DD/YY
        fixed_dates = []
        for date in report['Last_Update']:
            if date.find('/') > -1:
                # old format dates, split off the time
                sdate = date.split()
                fixed_dates.append(sdate[0])
            else:
                fdate = date[5:7].lstrip('0') + '/' + date[8:10] + '/' + date[2:4]
                fixed_dates.append(fdate)
        report['Last_Update'] = fixed_dates

    def fix_country_names(self, report):
        countries = []
        for country in report['Country_Region']:
            if country == 'Mainland China':
                country = 'China'
            if country == 'South Korea' or country == 'Republic of Korea':
                country = 'Korea, South'
            if country == 'UK':
                country = 'United Kingdom'
            countries.append(country)
        report['Country_Region'] = countries

    def county_data(self, county, which='Confirmed'):
        cases = []
        dates = []
        for date, report in zip(self.dates, self.reports):
            if report['Admin2'].isnull().all():
                # All NaNs, no data for this date
                continue
            df = report[report['Admin2'] == county]
            if len(df) > 0:
                case = 0
                for index in df.index:
                    value = report[which][index]
                    if not np.isnan(value):
                        case += int(value)
                cases.append(case)
                dates.append(date)
        return np.array(cases, dtype=float), dates
        
    def state_data(self, state, which='Confirmed'):
        cases = []
        dates = []
        for date, report in zip(self.dates, self.reports):
            df = report[report['Province_State'] == state]
            if len(df) > 0:
                case = 0
                for index in df.index:
                    value = report[which][index]
                    if not np.isnan(value):
                        case += int(value)
                cases.append(case)
                dates.append(date)
        return np.array(cases, dtype=float), dates

    def country_data(self, country, which='Confirmed'):
        cases = []
        dates = []
        for date, report in zip(self.dates, self.reports):
            df = report[report['Country_Region'] == country]
            if len(df) > 0:
                case = 0
                for index in df.index:
                    value = report[which][index]
                    if not np.isnan(value):
                        case += int(value)
                cases.append(case)
                dates.append(date)
        return np.array(cases, dtype=float), dates

