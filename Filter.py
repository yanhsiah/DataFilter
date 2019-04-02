#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import math
import sys
from statistics import mean


# In[107]:


# constants
sample_size = 31
sample_radius = math.floor(sample_size / 2)
cols = ['ch0', 'ch1', 'ch2', 'ch3']
num_files = 17


# In[108]:


# find stable high and low
def find_high_low(ds):
    freq_count = ds.value_counts(normalize=True)

    indexes = []
    index = 0
    ratio = 0
    mix_avg = 0
    for voltage, freq in freq_count.iteritems():
        ratio = ratio + freq
        mix_avg += voltage
        indexes.append(index)
        if ratio > 0.7:
            break
        index += 1
    mix_avg /= (index + 1)
    # print(mix_avg)

    high = 0
    high_portion = 0
    low = 0
    low_portion = 0
    for voltage, freq in freq_count.iteritems():
        # print(voltage, freq)
        if index < 0:
            break
        if voltage > mix_avg:
            high += voltage * freq
            high_portion += freq
        else:
            low += voltage * freq
            low_portion += freq
        index -= 1

    return high / high_portion, low / low_portion


# In[115]:


# crop the pick up and drop down segments
def crop(input_file_path, output_file_path):
    # step 0: read input file
    
    df = pd.read_csv(input_file_path, usecols=cols)
    # get middle from high, low
    total_rows = df.shape[0]
    high, low = find_high_low(df.round(6)[cols[0]])
    # high, low = 4.8059039E-05, 2.3650005E-05
    middle = (high + low) / 2.0

    # step 1:
    picked_prev = False
    changing_points = []
    for i in range(total_rows):
        picked_curr = df.loc[i][0] < middle
        if bool(picked_prev) ^ bool(picked_curr):
            changing_points.append(i)
            picked_prev = picked_curr
    # print(changing_points)
    
    # step 2:
    indexes = []
    fluctuation = []
    changing_points.append(sys.float_info.max)
    for i in range(1, len(changing_points)):
        changing_point = changing_points[i - 1]
        fluctuation.append(changing_point)
        if changing_points[i] - changing_point > sample_radius: 
            changing_point = math.floor(mean(fluctuation))
            fluctuation = []
            # indexes.append(changing_point)
            indexes.append([changing_point - sample_radius, changing_point + sample_radius])
    # print(indexes)

    # step 3:
    parsed = pd.DataFrame()
    for index in indexes:
        parsed = parsed.append(pd.DataFrame(df.loc[index[0]:index[1]]))
    parsed.to_csv(output_file_path)
    print('p&d = ', len(indexes), 'output to', output_file_path)
    
    return high, low


# In[118]:


def process_folder(dir_name):
    metadata = pd.DataFrame(columns=['high', 'low'])
    for i in range(num_files):
        input_file_path = dir_name + '/' + str(i) + '.csv'
        output_file_path = 'filtered_' + input_file_path
        high, low = crop(input_file_path, output_file_path)
        metadata = metadata.append(pd.DataFrame([[high, low]], columns=['high', 'low']), ignore_index=True)

    metadata.to_csv('filtered_' + dir_name + '/metadata.csv')
    print(metadata)


# In[121]:


for folder in ['chocolate', 'pasta', 'peach']:
    process_folder(folder)


# In[ ]:




