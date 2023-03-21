import pandas as pd


def read_data():
    # instant read in
    df = pd.read_csv('data/clinical_TumorCompendium_v11_PolyA_2020-04-09.tsv', sep='\t', header=0)

    # takes around 6 minutes
    df2 = pd.read_csv("data/TumorCompendium_v11_PolyA_hugo_log2tpm_33466genes_reduced.tsv", sep="\t", header=0)

    return df, df2