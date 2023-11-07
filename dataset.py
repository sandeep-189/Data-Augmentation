## Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
## Description : Python file containing dataset class for timeseries data and the functions to clean PAMAP2 and RWHAR datasets

import pandas as pd
import os
import numpy as np
from torch.utils.data import random_split
import torch
import zipfile
from io import BytesIO

class TimeSeriesDataset(torch.utils.data.Dataset):
    # Reads a list of tuple of timeseries data and target data into a Pytorch dataloader compatible format
    def __init__(self, List):
        super(TimeSeriesDataset,self).__init__()
        self.data = List
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data,tgt = self.data[idx]
        sample = {"data":torch.tensor(data, dtype = torch.float),"label":torch.tensor(tgt-1, dtype = torch.long)}
        return sample

PAMAP2_DROP_COLUMNS = [2,3,7,8,9,16,17,18,19,20,24,25,26,33,34,35,36,37,41,42,43,50,51,52,53] # check indicies in link in function    

def clean_PAMAP2_dataset(filepath, skip_col = PAMAP2_DROP_COLUMNS):
    # There are 54 columns in the dataset with 9 subjects who perform 24 different activities. We will select the pertinent data using
    # skip_col to remove columns and remove absent data rows and non activity data. The cleaned set by default will have 29 columns
    # See https://archive.ics.uci.edu/ml/machine-learning-databases/00231/readme.pdf for data info

    columns = [x for x in range(54) if x not in skip_col]
    # read from file reading only column numbers in columns variable
    sub1 = pd.read_table(filepath, sep=" ", header = None, usecols = columns,)
    
     # as per the source, activity id 0 is used when the sensors are recording but no selected activity is being done by the subject
    # as recommended, it is marked with NaN and later removed by dropna.
    sub1 = sub1.where((sub1.iloc[:,1] > 0) & (sub1.iloc[:,1] < 8))

    # remove any rows that have NaN 
    sub1 = sub1.dropna()
    
    sub1 = sub1.astype("float64")
    return sub1

RWHAR_ACTIVITY_NUM = {
    "climbingdown" :1,
    "climbingup"   :2,
    "jumping"      :3,
    "lying"        :4,
    "running"      :5,
    "sitting"      :6,
    "standing"     :7,
    "walking"      :8,
}

RWHAR_BAND_LOCATION = {
    "chest"   : 1,
    "forearm" : 2,
    "head"    : 3,
    "shin"    : 4,
    "thigh"   : 5,
    "upperarm": 6,
    "waist"   : 7,
}

def check_RWHAR_zip(path):
    # verify that the path is to the zip containing csv and not another zip of csv
    
    if any(".zip" in filename for filename in zipfile.ZipFile(path,"r").namelist()):
        # There are multiple zips in some cases
        with zipfile.ZipFile(path,"r") as temp:
            path = BytesIO(temp.read(max(temp.namelist()))) # max chosen so the exact same acc and gyr files are selected each time (repeatability)
    return path

def RWHAR_load_csv(path):
    # Loads up the csv at given path, returns a dictionary of data at each location
    
    path = check_RWHAR_zip(path)
    tables_dict = {}
    with zipfile.ZipFile(path,"r") as Zip:
        zip_files = Zip.namelist()
        
        for csv in zip_files:
            if "csv" in csv:
                location = RWHAR_BAND_LOCATION[csv[csv.rfind("_") + 1:csv.rfind(".")]] # location is between last _ and .csv extension
                sensor = csv[:3]
                prefix = sensor.lower()+"_"
                table = pd.read_csv(Zip.open(csv))
                table.rename(columns = {"attr_x" : prefix+"x",
                                        "attr_y" : prefix+"y",
                                        "attr_z" : prefix+"z",
                                        "attr_time" : "timestamp",
                                       }, inplace = True)
                table.drop(columns = "id", inplace = True)
                tables_dict[location] = table
        
    return tables_dict

def find_first_close_match(df1, df2):
    found = False
    mask = df1.iloc[:50].apply(np.isclose, b = df2.iloc[:50], atol = 2)
    for i in range(mask.shape[0]):
        if mask.iat[i].any():
            # The row number in acc and gyr where this index is found
            acc_start = i
            gyr_start = mask.iat[i].argmax()
            found = True
            break
    if not found:
        mask = df1.apply(np.isclose, b = df2, atol = 2)
        for i in range(mask.shape[0]):
            if mask.iat[i].any():
                # The row number in acc and gyr where this index is found
                acc_start = i
                gyr_start = mask.iat[i].argmax()
                found = True
                break
    return acc_start, gyr_start

def RWHAR_load_table_activity(path_acc, path_gyr):
    # Logic for loading each activity zip file for acc and gyr and then merging the tables at each location
    
    acc_tables = RWHAR_load_csv(path_acc)
    gyr_tables = RWHAR_load_csv(path_gyr)
    data = pd.DataFrame()
    
    for loc in acc_tables.keys():
        acc_tab = acc_tables[loc]
        gyr_tab = gyr_tables[loc]

        # find first timestamp which aligns the 2 tables
        acc_start, gyr_start = find_first_close_match(acc_tab['timestamp'],gyr_tab['timestamp'])
        
        # remove all data that comes before the start
        acc_tab = acc_tab.iloc[acc_start:]
        gyr_tab = gyr_tab.iloc[gyr_start:]
        
        # merge the two table based on timestamp being within +- tolerance milliseconds 
        # (all which do not conform will be dropped later)
        acc_tab = pd.merge_asof(left = gyr_tab, right = acc_tab, on = "timestamp", 
                                direction = "nearest", tolerance = 20)
        
        acc_tab["location"] = loc
        
        data = data.append(acc_tab)
    
    return data

def clean_RWHAR(filepath, sel_location = None):
    # the function reads the files in RWHAR dataset and each subject and each activity labelled in a panda table 
    # filepath is the parent folder containing all the RWHAR dataset.
    # Note: all entries are loaded but their timestamps are not syncronised. So a single location must be selected and 
    # all entries with NA must be dropped.
    
    subject_dir = os.listdir(filepath)
    dataset = pd.DataFrame()
    
    for sub in subject_dir:
        if "proband" not in sub:
            continue
#         files = os.listdir(filepath+sub)
#         files = [file for file in files if (("acc" in file or "gyr" in file) and "csv" in file)]
        subject_num = int(sub[7:]) # proband is 7 letters long so subject num is number following that
        sub_pd = pd.DataFrame()
        
        for activity in RWHAR_ACTIVITY_NUM.keys(): # pair the acc and gyr zips of the same activity
            activity_name = "_"+activity+"_csv.zip"
            
            path_acc = filepath+sub+"/acc"+activity_name # concat the path to acc file for given activity and subject
            path_gyr = filepath+sub+"/gyr"+activity_name # concat the path to gyr file for given activity and subject

            table = RWHAR_load_table_activity(path_acc,path_gyr)
            table["activity"] = RWHAR_ACTIVITY_NUM[activity] # add a activity column and fill it with activity num
            sub_pd = sub_pd.append(table)
            
        sub_pd["subject"] = subject_num # add subject id to all entries
        dataset = dataset.append(sub_pd)
    
    if sel_location is not None:
        print("Selecting location : ",sel_location)
        dataset = dataset[dataset.location == RWHAR_BAND_LOCATION[sel_location]]
        dataset = dataset.drop(columns = "location")
    
    dataset = dataset.drop(columns = "timestamp")
    dataset = dataset.dropna()
    print(dataset['activity'].value_counts())
    return dataset

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def generate_pe(channel, length, period = 100, channel_cosine = True):
    # This function generates a positional embedding needed for transformer to gain awareness of position of input
    # suited for time series. If channel_cosine is false, instead of sin and cos functions across the row and column  
    # respectively, we use sin of wavelength period for each channel. we return a addable pe of size channel, length.
    # The batch dimension will be broadcasted automatically.
    
    if not channel_cosine: # pe is an alternating sin and cos function of length across channel
        angle_rads = get_angles(np.arange(length)[:, np.newaxis],
                              np.arange(channel)[np.newaxis, :],
                              channel)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pe = torch.from_numpy(angle_rads.reshape(channel,length).astype(np.float32))
        
    else: # pe is a sum of sin and cos of the position of variable
        pe = torch.zeros((channel,length), dtype = torch.float32)
        for i in torch.arange(0, length, dtype = torch.long):
            for j in torch.arange(0, channel, dtype = torch.long):
                pe[j,i] = torch.sin(i * 2 * np.pi / period) + torch.cos(j * 2 * np.pi / period)
    
    return pe
