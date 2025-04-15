import numpy as np

CHARPROTSET =  {"Q": 0, "W": 1, "E": 2, "R": 3, "T": 4,
                "Y": 5, "I": 6, "P": 7, "A": 8, "S": 9,
                "D": 10, "F": 11, "G": 12, "H": 13, "K": 14,
                "L": 15, "C": 16, "V": 17, "N": 18, "M": 19}

AMINOSET = {"GLN": 0, "TRP": 1, "GLU": 2, "ARG": 3, "THR": 4,
                "TYR": 5, "ILE": 6, "PRO": 7, "ALA": 8, "SER": 9,
                "ASP": 10, "PHE": 11, "GLY": 12, "HIS": 13, "LYS": 14,
                "LEU": 15, "CYS": 16, "VAL": 17, "ASN": 18, "MET": 19}


AMINO_ACID_MAP = {
                "GLN": "Q",   "TRP": "W",   "GLU": "E",   "ARG": "R",  
                "THR": "T",   "TYR": "Y",   "ILE": "I",   "PRO": "P",  
                "ALA": "A",   "SER": "S",   "ASP": "D",   "PHE": "F",  
                "GLY": "G",   "HIS": "H",    "LYS": "K",   "LEU": "L",  
                "CYS": "C",   "VAL": "V",   "ASN": "N",   "MET": "M"   
            }

def parse_fasta_seq(fasta_fp):
    """Gets the sequence in a one-entry FASTA file."""
    seq = ""
    with open(fasta_fp, "r") as i_fh:
        content = i_fh.read()
        entry_count = content.count(">")
        if entry_count > 1:
            raise ValueError("Can only read FASTA files with one entry.")
        elif entry_count == 0:
            raise ValueError("No entry found in the input file.")
        for line in content.split("\n"):
            if line.startswith(">"):
                continue
            seq += line.rstrip()
    return seq


one_to_three = {"Q": "GLN", "W": "TRP", "E": "GLU", "R": "ARG", "T": "THR",
                "Y": "TYR", "I": "ILE", "P": "PRO", "A": "ALA", "S": "SER",
                "D": "ASP", "F": "PHE", "G": "GLY", "H": "HIS", "K": "LYS",
                "L": "LEU", "C": "CYS", "V": "VAL", "N": "ASN", "M": "MET"}




def seq_to_cg_pdb(seq, out_fp=None):
    """
    Gets an amino acid sequence and returns a template
    CG PDB file.
    """
    pdb_lines = []
    for i, aa_i in enumerate(seq):
        res_idx = i + 1
        aa_i = one_to_three[aa_i]
        line_i = "ATOM{:>7} CG   {} A{:>4}       0.000   0.000   0.000  1.00  0.00\n".format(
                   str(res_idx), aa_i, str(res_idx))
        pdb_lines.append(line_i)
    pdb_content = "".join(pdb_lines)
    if out_fp is not None:
        with open(out_fp, "w") as o_fh:
            o_fh.write(pdb_content)
    return pdb_content


def random_sample_trajectory(traj, n_samples):
    """Samples a random subset of a trajectory."""
    random_ids = np.random.choice(traj.shape[0], n_samples,
                                  replace=traj.shape[0] < n_samples)
    return traj[random_ids]