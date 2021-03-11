import pandas as pd
import os
import numpy as np
from torch.utils.data import random_split
import torch
import zipfile
from io import BytesIO

class TimeSeriesDataset(torch.utils.data.Dataset):
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

def RWHAR_load_table_activity(path_acc, path_gyr):
    # Logic for loading each activity zip file for acc and gyr and then merging the tables at each location
    
    acc_tables = RWHAR_load_csv(path_acc)
    gyr_tables = RWHAR_load_csv(path_gyr)
    data = pd.DataFrame()
    
    for loc in acc_tables.keys():
        acc_tab = acc_tables[loc].set_index("timestamp")
        gyr_tab = gyr_tables[loc].set_index("timestamp")
        
        acc_tab = acc_tab.join(gyr_tab)
        acc_tab["location"] = loc
        
        data = data.append(acc_tab)
                
    return data

def regularize_RWHAR(table, freq = 50):
    # function 
    pass

def clean_RWHAR(filepath):
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
    
    
    return dataset