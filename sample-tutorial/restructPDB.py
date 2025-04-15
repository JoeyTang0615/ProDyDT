import os
def single_to_three_letter(sequence):
    aa_dict = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
        'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
        'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
        'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    return [aa_dict[res] for res in sequence]

def create_pdb_file(sequence, coordinates, filepath="resPdblist" ,filename = "output"):
    three_letter_sequence = single_to_three_letter(sequence)  
    pdb_content = ""
    atom_index = 1  

    for res_index, (residue, coord) in enumerate(zip(three_letter_sequence, coordinates), start=1):

        pdb_content += f"ATOM  {atom_index:>5}  CA  {residue:<3} A{res_index:>4}    {coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}  1.00 20.00           C\n"
        atom_index += 1

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    file_name = os.path.join(filepath, filename)
    with open(file_name, "w") as file:
        file.write(pdb_content)
    # print(f"PDB file '{file_name}' has been created with Ca atoms and three-letter amino acids.")

