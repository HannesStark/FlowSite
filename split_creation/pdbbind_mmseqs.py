import copy
import os

import numpy as np
from tqdm import tqdm

fasta_file = 'data/all_pdbbind_sequences.fasta'

pdb_ids = os.listdir('../ligbind/data/PDBBind_processed')


PROTEIN_SEQS = fasta_file
OUT_DIR = "data/"
OUT_PATH = "data/"
SEQ_SIMILARITY = 0.1
DEV_SIZE = 0.03
TEST_SIZE = 0.03
train_out_path = 'index/pdbbind_mmseqs_10sim_train'
val_out_path = 'index/pdbbind_mmseqs_10sim_val'
test_out_path = 'index/pdbbind_mmseqs_10sim_test'

# Compute clusters
res_path = os.path.join(OUT_DIR, "clusterRes")
tmp_path = os.path.join(OUT_DIR, "tmp")
os.system(
    f"mmseqs easy-cluster {PROTEIN_SEQS} {res_path} {tmp_path} --min-seq-id {SEQ_SIMILARITY} --cov-mode 1 --cluster-mode 2"
)

# Load cluster
clusters = {}  # type: ignore
cluster_path = os.path.join(OUT_DIR, "clusterRes_cluster.tsv")
with open(cluster_path, "r") as f:
    for row in f.read().splitlines():
        cluster, item = row.split()
        clusters.setdefault(cluster, set()).add(item)

print("Num clusters: ", len(clusters))

# merge clusters with same pdb_id
merged_clusters = copy.deepcopy(clusters)
for pdb_id in tqdm(pdb_ids):
    try:
        containing_clusters = []
        for cluster_rep, cluster in merged_clusters.items():
            cluster_list = '_'.join(list(cluster))
            if pdb_id in cluster_list:
                containing_clusters.append(cluster_rep)
        merged_cluster = []
        for containing_cluster in containing_clusters:
            merged_cluster.extend(list(merged_clusters[containing_cluster]))
            del merged_clusters[containing_cluster]
        merged_clusters[merged_cluster[0]] = set(merged_cluster)
    except Exception as e:
        print(pdb_id)
        print(str(e))

print('Num merged clusters: ', len(merged_clusters))

# Perform split and save
merged_keys = list(merged_clusters.keys())
indices = np.random.permutation(len(merged_keys))
train_size = 1 - DEV_SIZE - TEST_SIZE
train_idx = indices[: int(train_size * len(merged_keys))]
dev_idx = indices[int(train_size * len(merged_keys)) : int((train_size + DEV_SIZE) * len(merged_keys))]
test_idx = indices[int((train_size + DEV_SIZE) * len(merged_keys)) :]

train_items = [merged_keys[i] for i in train_idx]
dev_items = [merged_keys[i] for i in dev_idx]
test_items = [merged_keys[i] for i in test_idx]

train_proteins = [prot for clust in train_items for prot in merged_clusters[clust]]
dev_proteins = [prot for clust in dev_items for prot in merged_clusters[clust]]
test_proteins = [prot for clust in test_items for prot in merged_clusters[clust]]
print('Per chain sizes: ')
print("Train size: ", len(train_proteins))
print("Dev size: ", len(dev_proteins))
print("Test size: ", len(test_proteins))

train_proteins_merge = list(set([chain[:4] for chain in train_proteins]))
dev_proteins_merge = list(set([chain[:4] for chain in dev_proteins]))
test_proteins_merge = list(set([chain[:4] for chain in test_proteins]))
print('Per PDB file sizes: ')
print("Train size: ", len(train_proteins_merge))
print("Dev size: ", len(dev_proteins_merge))
print("Test size: ", len(test_proteins_merge))

np.savetxt(train_out_path, train_proteins_merge, fmt="%s")
np.savetxt(val_out_path, dev_proteins_merge, fmt="%s")
np.savetxt(test_out_path, test_proteins_merge, fmt="%s")
