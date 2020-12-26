import os
import pickle
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#matplotlib.interactive(True)

class CA_counties(object):
    def __init__(self, csvfile='/Users/davidgrote/COVID19/COVID19_plots/California_Counties.csv'):
        self.csvfile = csvfile
        self.dataframe = pandas.read_csv(csvfile)
        self.extract_boundaries()

    def extract_boundaries(self):
        self.boundaries_dict = {}
        for county_name in self.dataframe['Name']:
            df = self.dataframe[self.dataframe['Name']==county_name]  # dataframe
            ss = df['the_geom']  # series
            vv = ss[ss.index[0]]  # string
            self.boundaries_dict[county_name] = self.boundary_from_string(vv)

    def boundary_from_string(self, datastring):
        ii = datastring.find('(')
        datastring = datastring[ii+1:-1]
        block_list = []
        while len(datastring) > 4:
            ii = datastring.find(')')
            blockstring = datastring[2:ii]
            datastring = datastring[ii+4:]
            coordslist = blockstring.split(',')
            coords = []
            for coordstring in coordslist:
                c = coordstring.split()
                coords.append([float(c[0]), float(c[1])])
            block_list.append(np.array(coords))
        return block_list

    def plot_county(self, county_name):
        fig, ax = plt.subplots(figsize=(6,6))
        for coords in self.boundaries_dict[county_name]:
            ax.plot(coords[:,0], coords[:,1])
        fig.show()



counties = CA_counties()

"""
counties.plot_county('Solano')

fig, ax = plt.subplots(figsize=(8,8))
for county_name in counties.boundaries_dict.keys():
    for coords in counties.boundaries_dict[county_name]:
        ax.plot(coords[:,0], coords[:,1])

fig.show()

"""
