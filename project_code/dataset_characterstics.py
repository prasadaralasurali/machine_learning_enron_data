#!/usr/bin/python

import pickle
import sys
sys.path.append("../tools/")

'''load 'my_dataset_cleaned.pkl' (contains enron data with outliers removed and
new features added)'''
data_dict = pickle.load(open("my_dataset_cleaned.pkl", "r") )

print "Number of datapoints: ", len(data_dict)

#number of features
for key in data_dict:
    print "Number of features (including new features and POI label): ", len(data_dict[key])
    print "List of features"
    print "----------------"
    for item in data_dict[key]:
        print item
    break
#number of POIs
poi = 0
for key in data_dict:
    if data_dict[key]['poi'] == True:
        poi += 1
print "\nNumber of POIs: ", poi

#A function to summarise features (number of missing values, mean, min and max) 
def feature_summary(feature):
    NaN = 0
    feature_data = []
    for item in data_dict:
        if data_dict[item][feature] == 'NaN':
            NaN += 1
        else:
            feature_data.append(data_dict[item][feature])
    print "Number of missing values: ", NaN
    try:
        mean = round(sum(feature_data) / float(len(feature_data)), 2)
        print "Mean: ", mean
        print "Max: ", max(feature_data)
        print "Min: ", min(feature_data)
    except:
        pass
    print "##############################\n"

#create the list of features
for key in data_dict:
    feature_list = data_dict[key].keys()
    break
#print summary of each features
print "\nCharacteristics of each feature after removing outliers"
print "-------------------------------------------------------"
for feature in feature_list:
    print feature
    print "-" * len(feature)
        
    feature_summary(feature)
            



