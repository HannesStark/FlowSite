
import shutil
from tqdm import tqdm
import os


moad_path = 'data/moad/BindingMOAD_2020'
unit_names = os.listdir(moad_path)
prepared_path = 'data/moad/moad_prepared'

pdb_ids = list(set([unit_name[:4] for unit_name in unit_names]))

print(len(pdb_ids))
os.makedirs(prepared_path, exist_ok=True)
for unit_name in tqdm(unit_names):
        pdb_id = unit_name[:4]
        new_name = pdb_id + '_unit' + unit_name[8:]
        os.makedirs(os.path.join(prepared_path, new_name), exist_ok=True)
        shutil.copy(os.path.join(moad_path, unit_name), os.path.join(prepared_path, new_name, new_name + '.pdb'))