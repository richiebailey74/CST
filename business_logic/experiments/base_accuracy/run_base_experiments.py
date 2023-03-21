import pandas as pd
from business_logic.experiments.base_accuracy.helpers import runexperiments_full_2
from business_logic.experiments.base_accuracy.visualize import graph_BAC_results


def run_reduced_male_v_female_carcinoma():
    df_pos1 = pd.read_csv('data/male_v_female_carcinomas_reduced_pos_samples.tsv', sep='\t',
                          header=0)
    df_pos1 = df_pos1.set_index(df_pos1.iloc[:, 0])
    df_pos1 = df_pos1.iloc[:, 1:]

    df_neg1 = pd.read_csv('data/male_v_female_carcinomas_reduced_neg_samples.tsv', sep='\t',
                          header=0)
    df_neg1 = df_neg1.set_index(df_neg1.iloc[:, 0])
    df_neg1 = df_neg1.iloc[:, 1:]

    pos1 = df_pos1.sample(200, replace=False)
    neg1 = df_neg1.sample(200, replace=False)

    exp1_cst, exp1_pca, exp1_mi, exp1_umap, exp1_kpca = runexperiments_full_2(pos1, neg1)
    experiment1 = "Classifying Between Male/Female\nwithin 6 Types of Carcinomas"
    graph_BAC_results(exp1_cst, exp1_pca, exp1_mi, exp1_umap, exp1_kpca, experiment1)


def run_reduced_male_v_female_adenocarcinoma():
    df_pos2 = pd.read_csv('data/male_v_female_adenocarcinomas_reduced_pos_samples.tsv', sep='\t',
                          header=0)
    df_pos2 = df_pos2.set_index(df_pos2.iloc[:, 0])
    df_pos2 = df_pos2.iloc[:, 1:]

    df_neg2 = pd.read_csv('data/male_v_female_adenocarcinomas_reduced_neg_samples.tsv', sep='\t',
                          header=0)
    df_neg2 = df_neg2.set_index(df_neg2.iloc[:, 0])
    df_neg2 = df_neg2.iloc[:, 1:]

    pos2 = df_pos2.sample(200, replace=False)
    neg2 = df_neg2.sample(200, replace=False)

    exp2_cst, exp2_pca, exp2_mi, exp2_umap, exp2_kpca = runexperiments_full_2(pos2, neg2)
    experiment2 = "Classifying Between Male/Female\nwithin 6 Types of Adenocarcinomas"
    graph_BAC_results(exp2_cst, exp2_pca, exp2_mi, exp2_umap, exp2_kpca, experiment2)


def run_reduced_ped_v_nonped_carcinomas():
    df_pos3 = pd.read_csv('data/ped_v_nonped_carcinomas_reduced_pos_samples.tsv', sep='\t',
                          header=0)
    df_pos3 = df_pos3.set_index(df_pos3.iloc[:, 0])
    df_pos3 = df_pos3.iloc[:, 1:]

    df_neg3 = pd.read_csv('data/ped_v_nonped_carcinomas_reduced_neg_samples.tsv', sep='\t',
                          header=0)
    df_neg3 = df_neg3.set_index(df_neg3.iloc[:, 0])
    df_neg3 = df_neg3.iloc[:, 1:]

    pos3 = df_pos3.sample(83, replace=False)
    neg3 = df_neg3.sample(83, replace=False)

    exp3_cst, exp3_pca, exp3_mi, exp3_umap, exp3_kpca = runexperiments_full_2(pos3, neg3)
    experiment3 = "Classifying Between Ped/Non-ped\nwithin 6 Types of Carcinomas"
    graph_BAC_results(exp3_cst, exp3_pca, exp3_mi, exp3_umap, exp3_kpca, experiment3)


def run_reduced_carcinoma_v_adenocarcinoma():
    df_pos5 = pd.read_csv('data/carcinomas_v_adenocarcinomas_reduced_pos_samples.tsv', sep='\t',
                          header=0)
    df_pos5 = df_pos5.set_index(df_pos5.iloc[:, 0])
    df_pos5 = df_pos5.iloc[:, 1:]

    df_neg5 = pd.read_csv('data/carcinomas_v_adenocarcinomas_reduced_neg_samples.tsv', sep='\t',
                          header=0)
    df_neg5 = df_neg5.set_index(df_neg5.iloc[:, 0])
    df_neg5 = df_neg5.iloc[:, 1:]

    pos5 = df_pos5.sample(200, replace=False)
    neg5 = df_neg5.sample(200, replace=False)

    exp5_cst, exp5_pca, exp5_mi, exp5_umap, exp5_kpca = runexperiments_full_2(pos5, neg5)
    experiment5 = "Classifying Between 6 Types of Carcinomas\n and 4 Types of Adenocarcinomas"
    graph_BAC_results(exp5_cst, exp5_pca, exp5_mi, exp5_umap, exp5_kpca, experiment5)


