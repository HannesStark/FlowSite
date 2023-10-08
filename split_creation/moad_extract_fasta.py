import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import prody
import os
import json
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import copy


moad_path = 'data/moad/BindingMOAD_2020'
moad_path = 'data/moad/moad_prepared_redundancy_reduced'
unit_names = os.listdir(moad_path)

seqs = {}
from utils.featurize import atom_features_list
for unit_name in tqdm(unit_names):
    try:
        protein_path = f'{moad_path}/{unit_name}/{unit_name}.pdb'
        prody_struct = prody.parsePDB(protein_path)
        for chain_id, chain in enumerate(prody_struct.iterChains()):
            seq = []
            for residue in chain.iterResidues():
                if residue.getResname() not in atom_features_list['residues']: continue
                alpha_carbon = residue.select('name CA')
                nitrogen = residue.select('name N')
                carbon = residue.select('name C')
                if alpha_carbon is not None and nitrogen is not None and carbon is not None:
                    seq.append(residue.getSequence())
            if not seq == []:
                seqs[f'{unit_name}_chain{chain_id}'] = (''.join([letterlist[0] for letterlist in seq]))
    except Exception as e:
        print(str(e))
fasta_file = 'data/all_moad_sequences_nonJupyter.fasta'
with open(fasta_file, "w") as f:
    for identifier, sequence in tqdm(seqs.items()):
        f.write(f">{identifier}\n")
        f.write(f"{sequence}\n")
