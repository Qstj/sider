# ====== Create Network Propagation Profiles ====== #
# ====== Iterative Enrichment - Consensus Genes as final Seeds ====== #
import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import gseapy as gp
from run_NP import main_propagation
from sklearn.preprocessing import StandardScaler
import time

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default="/data/project/minwoo/Side_effect/NetGP/Data")
parser.add_argument('--target_fname', type=str, default='target_onehot_example.tsv')
parser.add_argument('--outdir', type=str, default="/data/project/minwoo/Side_effect/NetGP/Results")

parser.add_argument('--combined_score_cutoff', type=int, default=800)
parser.add_argument('--restart_prob', type=float, default=0.2)
parser.add_argument('--out_fname', type=str, default="drug_target_profile.out")
parser.add_argument('--ppi_fname', type=str, default="9606.protein.links.symbols.v11.5.txt")
args = parser.parse_args()

createFolder(args.outdir)
out_fpath = os.path.join(args.outdir, args.out_fname)
target_fpath = os.path.join(args.datadir, args.target_fname)
target_data = pd.read_csv(target_fpath, sep='\t', header=0, index_col=0)

drug_list = target_data.index.to_list()
#drug2idx = dict(zip(drug_data['drug_name'], drug_data['drug_idx']))

ppi_fpath = os.path.join(args.datadir, args.ppi_fname)

ppi_net = pd.read_csv(ppi_fpath, sep='\t', header=0)
ppi_genes = pd.Series(ppi_net['source'].append(ppi_net['target']).unique())

target_info_df = target_data[target_data.columns[target_data.columns.isin(ppi_genes)]]


# ==================================== #
# ====== Seed & Nwk Preparation ====== #
# ==================================== #

network_dir = os.path.join(args.outdir, 'network')
createFolder(network_dir)

seed_dir = os.path.join(args.outdir, 'seeds')
createFolder(seed_dir)

print("#PPI Edges:", ppi_net.shape[0])
for drug in drug_list:
    
    drug_target_df = target_data.query('drug_name == @drug')
    assert drug_target_df.shape[0] == 1
    target_mask = drug_target_df.iloc[0] == 1
    target_genes = target_data.columns[target_mask].to_list()
    
    # ====== Prepare Seed Files ====== #
    filename = os.path.join(seed_dir, f'{drug}_targets.seed')
    pd.DataFrame(target_genes).to_csv(filename, sep='\t', header=False, index=False)
    
    # ====== Prepare Network Files ====== #
    mask1 = np.logical_or(ppi_net['source'].isin(target_genes), ppi_net['target'].isin(target_genes))
    mask2 = np.logical_or(mask1, ppi_net['combined_score'] >= args.combined_score_cutoff)
    drug_subnet = ppi_net[mask2]
    
    filename = os.path.join(network_dir, f'{drug}_subnet_score_{args.combined_score_cutoff}.nwk')
    drug_subnet.to_csv(filename, sep='\t', header=False, index=False)


# ======================= #
# ====== Run NetGP ====== #
# ======================= #
args.constantWeight = 'True'
args.absoluteWeight = 'False'
args.addBidirectionEdge = 'True'
args.normalize = 'True'

outdir_intermediate = os.path.join(args.outdir, 'intermediate_results')
createFolder(outdir_intermediate)

# === Kegg or Reactome? === #
gene_sets = ['Reactome_2016'] #'Reactome_2013', 'Reactome_2015',  #ontology_terms
#gene_sets = ['KEGG_2013', 'KEGG_2015', 'KEGG_2016', 'KEGG_2019_Human', 'KEGG_2019_Mouse', 'KEGG_2021_Human']
#gene_sets = [term+'.gmt' for term in ontology_terms]

np_result_df = {}
for i, drug in enumerate(drug_list):
    print(f"# ====== Drug: {drug} ====== #")
    top_gene_sets_list = []
    # =============================================================== #
    # ============ Initial NP (Seeds: Drug Target Genes) ============ #
    # =============================================================== #
    args.input_graphs =  os.path.join(network_dir, f'{drug}_subnet_score_{args.combined_score_cutoff}.nwk')
    args.seed = os.path.join(seed_dir, f'{drug}_targets.seed') # seed list file
    np_result = main_propagation(args)
    # === Index Unification === #
    np_result = np_result.set_index(np_result['protein'])
    np_result = np_result.reindex(ppi_genes).fillna(0)
    assert np_result.shape[0] == len(ppi_genes), f'np scores list not length: {len(ppi_genes)} but {np_result.shape[0]}'
    
    # === gene-by-drug np score result === #
    np_top_genes = np_result.sort_values(by='np_score', ascending=False).iloc[:200,:]['protein'].to_list()
    top_gene_sets_list.append(np_top_genes)
    
    # ====== Iteration Starts ====== #
    step = 1
    while True:
        print(f'# === Iteration {step} === #')
        step += 1
        # ============================== #
        # ========= Enrich Step ======== #
        # ============================== #
        try:
            enr = gp.enrichr(gene_list=np_top_genes,
                             gene_sets=gene_sets,
                             organism='human',
                             outdir=None
                            )
        except:
            try:
                time.sleep(10)
                enr = gp.enrichr(gene_list=np_top_genes,
                                 gene_sets=gene_sets,
                                 organism='human',
                                 outdir=None
                                )
            except:
                time.sleep(10)
                enr = gp.enrichr(gene_list=np_top_genes,
                                 gene_sets=gene_sets,
                                 organism='human',
                                 outdir=None
                                 )
        
        sig_enr = enr.results.query('`Adjusted P-value` <= 0.05')

        sig_enr['Gene_list'] = sig_enr['Genes'].str.split(';').apply(lambda x: [g.strip() for g in x])
        sig_enr = sig_enr.explode('Gene_list')

        # === New Seed === #
        sig_path_genes = list(sig_enr['Gene_list'].unique())
        #top_gene_sets_list.append(sig_path_genes)

        seed_fpath = os.path.join(outdir_intermediate, f'{drug}_iterative_seed_genes.txt')
        pd.DataFrame(sig_path_genes).to_csv(seed_fpath, sep='\t', header=False, index=False)


        # ============================== #
        # =========== NP Step ========== #
        # ============================== #
        # === Iterative NP : w/ new seeds === #
        args.input_graphs =  os.path.join(network_dir, f'{drug}_subnet_score_{args.combined_score_cutoff}.nwk')
        args.seed = seed_fpath
        np_result = main_propagation(args)

        np_result = np_result.set_index(np_result['protein'])
        np_result = np_result.reindex(ppi_genes).fillna(0)
        assert np_result.shape[0] == len(ppi_genes), f'np scores list not length: {len(ppi_genes)} but {np_result.shape[0]}'

        np_top_genes = np_result.sort_values(by='np_score', ascending=False).iloc[:200,:]['protein'].to_list()
        
        # === Converged? === #
        if set(np_top_genes) == set(top_gene_sets_list[-1]):
            break
        
        top_gene_sets_list.append(np_top_genes)
    
    # === Final NP : w/ consensus seeds === #
    print('# === Final Iteration === #')
    top_gene_sets_list = [set(g_lst) for g_lst in top_gene_sets_list]
    final_np_top_genes = set.intersection(*top_gene_sets_list)

    final_seed_fpath = os.path.join(outdir_intermediate, f'{drug}_final_seed_genes.txt')
    pd.DataFrame(final_np_top_genes).to_csv(final_seed_fpath, sep='\t', header=False, index=False)

    args.input_graphs =  os.path.join(network_dir, f'{drug}_subnet_score_{args.combined_score_cutoff}.nwk')
    args.seed = final_seed_fpath
    np_result = main_propagation(args)

    np_result = np_result.set_index(np_result['protein'])
    np_result = np_result.reindex(ppi_genes).fillna(0)
    assert np_result.shape[0] == len(ppi_genes), f'np scores list not length: {len(ppi_genes)} but {np_result.shape[0]}'
    
    # === gene-by-drug np score result === #
    np_result_df[drug] = np_result['np_score']
    
np_result_df = pd.DataFrame(np_result_df)
# filename = os.path.join(args.outdir, 'netgp_total_np_profile.out')
# np_result_df.reset_index().to_csv(filename, sep='\t', header=True, index=False)


# ====== Keep Target Genes ====== #
#np_result_df = pd.read_csv(filename, sep='\t', header=0, index_col=0)
# total_target_genes = target_info_df.columns
# mask = np_result_df.index.isin(total_target_genes)
# np_result_df = np_result_df[mask]

# === Standard Scaling === #
scaler = StandardScaler()
np_result_transform = scaler.fit_transform(np_result_df)
np_result_transform = pd.DataFrame(np_result_transform, columns=np_result_df.columns, index=np_result_df.index)
np_result_transform = np_result_transform.transpose().reset_index().rename(columns={'index': 'drug_name'})

np_result_transform.to_csv(out_fpath, sep='\t', header=True, index=False)