import pickle
import esm
import torch
from Bio import SeqIO, PDB
import numpy as np

def get_esm_encoding(fasta_file,output_file=None):
    with open(fasta_file, "r") as handle:
        record = next(SeqIO.parse(handle, "fasta"))  
        protein_sequence = str(record.seq)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()

    data = [(0, protein_sequence)]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    
    token_representations = results["representations"][33]
    sequence_rep = token_representations.mean(1)  
    
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump(sequence_rep, f)
    
    return sequence_rep


def pdb_process(pdb_file):

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ca_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())
    
    return torch.tensor(ca_coords)

import torch
import numpy as np
import pickle

def kabsch_algorithm(P, Q):
    """
    P and Q point set, shape should be (N, 3)
    """

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q


    H = P_centered.T @ Q_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T

    t = centroid_Q - R @ centroid_P

    return R, t

def rmsd(P, Q, R, t):

    P_transformed = R @ P.T + t[:, np.newaxis]
    return np.sqrt(np.mean(np.sum((Q.T - P_transformed)**2, axis=0)))

def optimize_sample(GEN_data, ref_1, ref_2):

    GEN_data_mirror = GEN_data * -1 
    r1_list = []
    r2_list = []

    for i in range(len(GEN_data)):
        conf = torch.tensor(GEN_data[i]*10).to(torch.float64)

        target1 = torch.tensor(ref_1*10).to(torch.float64)
        a = np.array(conf)
        b = np.array(target1)
        R, t = kabsch_algorithm(a,b)
        r1_list.append(rmsd(a ,b ,R, t))

        target2 = torch.tensor(ref_2*10).to(torch.float64)
        a = np.array(conf)
        b = np.array(target2)
        R, t = kabsch_algorithm(a,b)
        r2_list.append(rmsd(a ,b ,R, t))

    r1_list = np.array(r1_list)
    r2_list = np.array(r2_list)
    r_list = r1_list + r2_list

    r1_list_mirror = []
    r2_list_mirror = []

    for i in range(len(GEN_data)):
        conf = torch.tensor(GEN_data_mirror[i]*10).to(torch.float64)

        target1 = torch.tensor(ref_1*10).to(torch.float64)
        a = np.array(conf)
        b = np.array(target1)
        R, t = kabsch_algorithm(a,b)
        r1_list_mirror.append(rmsd(a ,b ,R, t))

        target2 = torch.tensor(ref_2*10).to(torch.float64)
        a = np.array(conf)
        b = np.array(target2)
        R, t = kabsch_algorithm(a,b)
        r2_list_mirror.append(rmsd(a ,b ,R, t))

    r1_list_mirror = np.array(r1_list_mirror)
    r2_list_mirror = np.array(r2_list_mirror)
    r_mirror_list = r1_list_mirror + r2_list_mirror

    final_GEN_data = []

    for i in range(len(GEN_data)):
        if r_list[i] < r_mirror_list[i]:
            final_GEN_data.append(GEN_data[i])
        else:
            final_GEN_data.append(GEN_data_mirror[i])

    final_GEN_data = torch.stack(final_GEN_data)

    return final_GEN_data
