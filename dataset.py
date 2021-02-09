import torch

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