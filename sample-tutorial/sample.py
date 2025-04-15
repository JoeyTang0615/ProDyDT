from process import pdb_process, optimize_sample, kabsch_algorithm , rmsd, get_esm_encoding
import os  
from diffusions.diffusions import DiffusionModel
from models.models import DiT
import torch
from dataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np

device = 'cuda'

# Load you Protein File
conf_a = pdb_process('sample-tutorial/*_A.pdb')
conf_a = conf_a / 10
conf_b = pdb_process('sample-tutorial/*_B.pdb')
conf_b = conf_b / 10

with open("tutorial/PDB/1e0d_esm.pkl", 'rb') as f:
    seq = pickle.load(f)

seq = get_esm_encoding('sample-tutorial/*.fasta','sample-tutorial/*_esm.pkl')

model = DiT(hidden_size=256, depth=6, num_heads=16, mlp_ratio=4.0, esm_feature_size=1280, max_protein_length=600).to(device)
model.load_state_dict(torch.load('model_weights//best_model.pth'))
name  = "**"

#Sample numbers, we suggest L less than 500 for a relative short time
N = 500
conf_t = torch.rand(N,conf_a.size(0),3).float().to(device)

esm = seq.to(device)
ref_1 = conf_a.float().unsqueeze(0).to(device)
ref_2 = conf_b.float().unsqueeze(0).to(device)

dm = DiffusionModel(model,n_steps=1000, device=device)

for t in tqdm(reversed(range(1000)), desc="Sampling Progress", total=1000):
    md_1 = dm.p_sample(conf_t,esm,torch.tensor([t]).to(device),ref_1, ref_2)
    conf_t = md_1.detach()

conf_t = conf_t.to('cpu').detach()
path = f"case_study/{name}"
os.makedirs(path, exist_ok=True)
print(path)
output = optimize_sample(conf_t,ref_1.squeeze(0).float().to('cpu'),ref_2.squeeze(0).float().to('cpu'))
print(output.shape)

#Output the N*L*3 N samples of same length protein with the input 
#You can restructure the protein by restructPDB.py  
with open(f"case_study/{name}/{name}.pkl", 'wb') as f:
    pickle.dump(output,f)
