from torch.utils.data import Dataset, DataLoader
class ProteinDataset(Dataset):
    def __init__(self, data):
        self.md = data["MD"]
        self.seq = data["SEQ"]
        self.esm = data["ESM"]
        self.ref1 = data["REFCONF1"]
        self.ref2 = data["REFCONF2"]
        self.name = data["NAME"]


    def __len__(self):
        return len(self.md)

    def __getitem__(self, idx):
        return {
            "MD": self.md[idx],
            "SEQ": self.seq[idx],
            "ESM": self.esm[idx],
            "REFCONF1": self.ref1[idx],
            "REFCONF2": self.ref2[idx],
            "NAME": self.name[idx]
        }
    
