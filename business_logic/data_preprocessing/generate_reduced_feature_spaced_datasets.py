from .feature_select_most_variable import write_n_most_variable
from .read_whole_data import read_data
from pandas import pd


def male_v_female_carcinomas():
    df, df2 = read_data()

    df_samples = df[df['disease'] == 'lung squamous cell carcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples11 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples11 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'kidney clear cell carcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples12 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples12 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'thyroid carcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples13 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples13 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'head & neck squamous cell carcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples14 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples14 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'kidney papillary cell carcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples15 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples15 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'bladder urothelial carcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples16 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples16 = df2.loc[:, list(samples)]

    pos_samples3 = pd.concat(
        [pos_samples11.T, pos_samples12.T, pos_samples13.T, pos_samples14.T, pos_samples15.T, pos_samples16.T])
    neg_samples3 = pd.concat(
        [neg_samples11.T, neg_samples12.T, neg_samples13.T, neg_samples14.T, neg_samples15.T, neg_samples16.T])

    write_n_most_variable("male_v_female_carcinomas_reduced", pos_samples3, neg_samples3, 1200)
    return

def male_v_female_adenocarcinomas():
    df, df2 = read_data()

    df_samples = df[df['disease'] == 'lung adenocarcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples17 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples17 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'stomach adenocarcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples18 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples18 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'prostate adenocarcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples19 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples19 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'colon adenocarcinoma']
    samples = df_samples[df_samples['gender'] == 'male']['th_sampleid']
    pos_samples20 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['gender'] == 'female']['th_sampleid']
    neg_samples20 = df2.loc[:, list(samples)]

    pos_samples4 = pd.concat([pos_samples17.T, pos_samples18.T, pos_samples19.T, pos_samples20.T])
    neg_samples4 = pd.concat([neg_samples17.T, neg_samples18.T, neg_samples19.T, neg_samples20.T])

    write_n_most_variable("male_v_female_adenocarcinomas_reduced", pos_samples4, neg_samples4, 1200)
    return

def ped_v_nonped_carcinomas():
    df, df2 = read_data()

    df_samples = df[df['disease'] == 'lung squamous cell carcinoma']
    samples = df_samples[df_samples['pedaya'] == 'Yes, age < 30 years']['th_sampleid']
    pos_samples21 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['pedaya'] == 'No']['th_sampleid']
    neg_samples21 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'kidney clear cell carcinoma']
    samples = df_samples[df_samples['pedaya'] == 'Yes, age < 30 years']['th_sampleid']
    pos_samples22 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['pedaya'] == 'No']['th_sampleid']
    neg_samples22 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'thyroid carcinoma']
    samples = df_samples[df_samples['pedaya'] == 'Yes, age < 30 years']['th_sampleid']
    pos_samples23 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['pedaya'] == 'No']['th_sampleid']
    neg_samples23 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'head & neck squamous cell carcinoma']
    samples = df_samples[df_samples['pedaya'] == 'Yes, age < 30 years']['th_sampleid']
    pos_samples24 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['pedaya'] == 'No']['th_sampleid']
    neg_samples24 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'kidney papillary cell carcinoma']
    samples = df_samples[df_samples['pedaya'] == 'Yes, age < 30 years']['th_sampleid']
    pos_samples25 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['pedaya'] == 'No']['th_sampleid']
    neg_samples25 = df2.loc[:, list(samples)]

    df_samples = df[df['disease'] == 'bladder urothelial carcinoma']
    samples = df_samples[df_samples['pedaya'] == 'Yes, age < 30 years']['th_sampleid']
    pos_samples26 = df2.loc[:, list(samples)]
    samples = df_samples[df_samples['pedaya'] == 'No']['th_sampleid']
    neg_samples26 = df2.loc[:, list(samples)]

    pos_samples5 = pd.concat(
        [pos_samples21.T, pos_samples22.T, pos_samples23.T, pos_samples24.T, pos_samples25.T, pos_samples26.T])
    neg_samples5 = pd.concat(
        [neg_samples21.T, neg_samples22.T, neg_samples23.T, neg_samples24.T, neg_samples25.T, neg_samples26.T])

    write_n_most_variable("ped_v_nonped_carcinomas_reduced", pos_samples5, neg_samples5, 1200)
    return

def carcinomas_v_adenocarcinomas():
    df, df2 = read_data()

    samples1 = df[df['disease'] == 'lung squamous cell carcinoma']['th_sampleid']
    g1_samples = df2.loc[:, list(samples1)]

    samples2 = df[df['disease'] == 'kidney clear cell carcinoma']['th_sampleid']
    g2_samples = df2.loc[:, list(samples2)]

    samples3 = df[df['disease'] == 'thyroid carcinoma']['th_sampleid']
    g3_samples = df2.loc[:, list(samples3)]

    samples4 = df[df['disease'] == 'head & neck squamous cell carcinoma']['th_sampleid']
    g4_samples = df2.loc[:, list(samples4)]

    samples5 = df[df['disease'] == 'kidney papillary cell carcinoma']['th_sampleid']
    g5_samples = df2.loc[:, list(samples5)]

    samples6 = df[df['disease'] == 'bladder urothelial carcinoma']['th_sampleid']
    g6_samples = df2.loc[:, list(samples6)]

    samples7 = df[df['disease'] == 'lung adenocarcinoma']['th_sampleid']
    g7_samples = df2.loc[:, list(samples7)]

    samples8 = df[df['disease'] == 'stomach adenocarcinoma']['th_sampleid']
    g8_samples = df2.loc[:, list(samples8)]

    samples9 = df[df['disease'] == 'prostate adenocarcinoma']['th_sampleid']
    g9_samples = df2.loc[:, list(samples9)]

    samples10 = df[df['disease'] == 'colon adenocarcinoma']['th_sampleid']
    g10_samples = df2.loc[:, list(samples10)]

    pos_samples7 = pd.concat([g1_samples.T, g2_samples.T, g3_samples.T, g4_samples.T, g5_samples.T, g6_samples.T])
    neg_samples7 = pd.concat([g7_samples.T, g8_samples.T, g9_samples.T, g10_samples.T])

    write_n_most_variable("carcinomas_v_adenocarcinomas_reduced", pos_samples7, neg_samples7, 8000)
    return