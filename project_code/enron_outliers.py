#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../project_code/final_project_dataset.pkl", "r") )
'''uncomment the below line to remove outlier'''
#data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### Plot data with outlier

salary = data[:, 0]
bonus = data[:, 1]
matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


            





