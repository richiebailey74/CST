# standard
import pandas as pd

from business_logic.experiments.hypertuning.visualize import generate_plots
from business_logic.experiments.hypertuning.helpers import runexperiments_full_2_CST, getAllNonCSTValues


def run_hypertun_experiments_carcinoma_v_adenocarcinoma():
    # read in data
    df_pos5 = pd.read_csv('data/carcinomas_v_adenocarcinomas_reduced_pos_samples.tsv', sep='\t',
                          header=0)
    df_pos5 = df_pos5.set_index(df_pos5.iloc[:, 0])
    df_pos5 = df_pos5.iloc[:, 1:]

    df_neg5 = pd.read_csv('data/carcinomas_v_adenocarcinomas_reduced_neg_samples.tsv', sep='\t',
                          header=0)
    df_neg5 = df_neg5.set_index(df_neg5.iloc[:, 0])
    df_neg5 = df_neg5.iloc[:, 1:]

    pos5 = df_pos5.sample(200, replace=False, random_state=1062601)
    neg5 = df_neg5.sample(200, replace=False, random_state=1062601)

    nonCST_vals = getAllNonCSTValues(pos5, neg5)

    pos5 = df_pos5.sample(200, replace=False, random_state=59)
    neg5 = df_neg5.sample(200, replace=False, random_state=59)
    CST_vals = runexperiments_full_2_CST(pos5, neg5)

    # concatentate experiment values
    for i in range(2, 203, 10):
        nonCST_vals[i].append(CST_vals)

    bac_vals = nonCST_vals

    # plot graphs
    generate_plots(bac_vals)

    return

