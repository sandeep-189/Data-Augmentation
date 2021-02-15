import pandas as pd
import os
import numpy as np
from torch.utils.data import random_split
import torch
import dataset

DEFAULT_RWHAR_FILEPATH = ["./data/RWHAR/"]

DEFAULT_PAMAP2_FILEPATHS = ["./data/PAMAP2/subject101.dat","./data/PAMAP2/subject102.dat","./data/PAMAP2/subject103.dat","./data/PAMAP2/subject104.dat","./data/PAMAP2/subject105.dat","./data/PAMAP2/subject106.dat","./data/PAMAP2/subject107.dat","./data/PAMAP2/subject108.dat","./data/PAMAP2/subject109.dat"]



def clean_all_files(filepaths, clean_func): 
    # filespaths is a list of text files with panda tables stored in them.
    # the clean_func is a reference to the function to be applied to clean the dataset
    
    total = pd.DataFrame()
    
    for file in filepaths:
        table = clean_func(file) # table has cleaned dataset from one file
        total = total.append(table,ignore_index = True) # joins all data into one dataframe
    return total

def load_table(filepaths = DEFAULT_PAMAP2_FILEPATHS, clean_func = dataset.clean_PAMAP2_dataset, save_file = "clean_PAMAP2.pkl", force_reload = False):
    # function to load a panda table from filepaths, clean each table using given clean_func, append them into one table and return it.
    # the table is saved and reloaded when needed or when forced to reload it.
    
    if ((not os.path.isfile(save_file)) or force_reload): 
        print("Cleaning and loading subject files")
        table = clean_all_files(filepaths, clean_func)
        print("Compressing to file")
        table.to_pickle(save_file, compression="gzip")
    else:
        print("File exists. Loading")
        table = pd.read_pickle(save_file, compression="gzip")
        
    return table

def load_PAMAP2(force_reload = False):
    # helper function for one line loading with optimizations 
    
    table = load_table(force_reload = force_reload)
    
    print("Windowing")
    data = windowing(table)
    print("Done!")
    return data

def load_PAMAP2_activity(activity_num = 0, force_reload = False):
    # helper function to load only one activty from PAMAP2 table
    
    table = load_table(force_reload = force_reload)
    
    print("Keep only activity number ", activity_num)
    table = table.where(table.iloc[:,1] == activity_num) # every activity except activity_num becomes NA
    table = table.dropna()
    
    print("Windowing")
    data = windowing(table)
    print("Done!")
    return data

def load_PAMAP2_acceleration(force_reload = False):
    # helper function to load only the 1st IMU 3d acceleration from PAMAP2 table. This was done to get a simple dataset for quick testing
    
    table = load_table(force_reload = force_reload)
        
    print("Keep only acceleration")
    table = table.iloc[:,:5]
    
    print("Windowing")
    
    data = windowing(table, data_columns = range(2,5))
    print("Done!")
    return data

def load_PAMAP2_acc_activity(activity_num = 0, force_reload = False):
    # helper function to load only one activty and the 1st IMU 3d acceleration from PAMAP2 table. This was done to get a simple dataset for quick testing
    
    table = load_table(force_reload = force_reload)
    
    print("Keep only acceleration")
    table = table.iloc[:,:5] # do mean of acceleration
    
    print("Keep only activity number ", activity_num)
    table = table.where(table.iloc[:,1] == activity_num) # every activity except activity_num becomes NA
    table = table.dropna()
    
    print("Windowing")
    data = windowing(table, data_columns = range(2,5))
    print("Done!")
    return data

def load_RWHAR(force_reload = False):
    # helper function for one line loading with optimizations 
    
    sel_location = "forearm"
    
    table = load_table(filepaths = DEFAULT_RWHAR_FILEPATH,
                       clean_func = dataset.clean_RWHAR, 
                       subject_column = False, 
                       save_file = "clean_RWHAR.pkl", 
                       force_reload = force_reload
                      )
    print("Selecting location : ",sel_location)
    table = table[table.location == dataset.RWHAR_BAND_LOCATION[sel_location]] # selecting location
    table = table.drop(columns = "location") # drop the now unnecessary location column
    table = table.dropna() # drop all entries which do not have complete information
    
    print("Windowing")
    # Since each subject has different timestamps and the windowing function does not check the timestamps, we have to window each subject 
    # separately and append them together
    
    data =[]
    for sub in table.subject.unique():
        tmp = windowing(table[table.subject==sub], sampling_freq = 50, group_col_num = 7, data_columns = range(1,7))
        data.extend(tmp)
    print("Done!")
    return data

def windowing(df, time_window = 1, sampling_freq = 100, group_col_num = 1, data_columns = range(2,29)):
    # df is the cleaned dataframe from dataset. This function will break the different activities to "window"
    # seconds of grouped data using group_col_num as the activities column with sampling frequency determining number of  
    # samples in each second. If the window cannot be filled by same activity data from subject, it will be dropped.
    # Note: we assume that the timestamps are close by since the activities are done for a continuous period of time. 
    #       when the activity changes, only then is there a jump in timestamp. The table should be fed subject by subject
    
    event_window = time_window * sampling_freq # number of events in one window
    output_shape = (len(data_columns), event_window)
    
    data = []
    ptr = 0
    while(ptr+event_window < df.shape[0]):
        activity = df.iloc[ptr,group_col_num] # which activity id to group by
        
        # check if all activities in event window have same id and can fill one full data value
        if (df.iloc[ptr:ptr+event_window, group_col_num] == activity).all() :
            d = df.iloc[ptr:ptr+event_window,data_columns].values
            d = np.reshape(d, output_shape)
            data.append((d, activity)) # append as a tuple of data and activity number
            
        else: # move pointer to nearest different activity number and then continue
            ptr += (df.iloc[ptr:ptr+event_window, group_col_num] != activity).values.argmax()
            continue
            
        ptr += event_window
        
    return data

def split(dataset, val_pc):
    # dataset is the dataset to be split. val_pc is the percentage in float split given to val dataset.
    
    valnum = int(np.floor(val_pc * len(dataset)))
    trainnum = len(dataset) - valnum
    
    return random_split(dataset, [trainnum, valnum])

def dist(dataset, num_classes):
    # function is used to get a weight distribution of each label of a classified dataset to pass to Cross entropy loss object. 
    # it helps to train faster and better for imbalanced datasets
    # dataset is expected to be a torch dataset. This will return a vector of weights of each class
    
    weight = np.zeros((num_classes,1))
    for data in dataset:
        weight[data["label"].item()] += 1
    weight /= len(dataset)
    return weight

def remove_prefix_from_dict(prefix, dicti):
    # pytorch lightning saves models with each models variable name. However we need only the state dict for
    # restoring a saved model so we need to strip the prefixed name from each dictionary key.
    # prefix is the variable name (must include the dot '.')
    # dicti is the state dict taken from the saved pytorch lightning model (not needed if you do not use PL)
    
    keys = list(dicti.keys())
    for key in keys:
        if key.startswith(prefix):
            new_key = key[len(prefix):] # new key is everthing after the prefix name
            dicti[new_key] = dicti.pop(key) # the value of old key is assigned to stripped key (old key is removed from dictionary)
            
    return dicti

def conv1d_ele_size(input_size, kernel, padding, stride, dilation):
    # Convinience function to get the output size of a 1d convolution 
    
    return int((input_size + (2*padding) - dilation*(kernel - 1) - 1)/stride + 1)

def generate_pe(channel, length, period = 100, channel_cosine = True):
    # This function generates a positional embedding needed for transformer to gain awareness of position of input
    # suited for time series. If channel_cosine is false, instead of sin and cos functions across the row and column  
    # respectively, we use sin of wavelength period for each channel. we return a addable pe of size channel, length.
    # The batch dimension will be broadcasted automatically.
    
    if not channel_cosine: # pe is a sin function of length repeated across channel
        pos = torch.arange(0, length, dtype = torch.float32)
        pe = torch.sin(pos * 2 * np.pi / period)
        pe = pe.repeat((channel,1))
        
    else: # pe is a sum of sin and cos of the position of variable
        pe = torch.zeros((channel,length), dtype = torch.float32)
        for i in torch.arange(0, length, dtype = torch.long):
            for j in torch.arange(0, channel, dtype = torch.long):
                pe[j,i] = torch.sin(i * 2 * np.pi / period) + torch.cos(j * 2 * np.pi / period)
    
    return pe