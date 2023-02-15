
import os

import pandas as pd

from eval_dataset import eval_monoclass_metrics_dataset, eval_multiclass_metrics_dataset

from define_variables import *




def read_dataset_monoclass_eval(dataset_name):

    data_dir = os.path.join(data_path, dataset_name)

    print(data_dir)

    res_eval_path = os.path.join(data_dir, "derivatives", "evaluation_results")

    try:
        os.makedirs(res_eval_path)
    except OSError:
        print("res_eval_path {} already exists".format(res_eval_path))


    csv_name = "monoclass_" + dataset_name + "_eval_res.csv"

    csv_file = os.path.join(res_eval_path, csv_name)

    if os.path.exists(csv_file):

        df_dataset = pd.read_csv(csv_file)

        df_dataset["Dataset"] = [dataset_name]*len(df_dataset.index)

    else:
        print("Error, dataset {} was not computed, {} do not exists".format(dataset_name, csv_file))
        exit(-1)

    stats_df_dataset  = df_dataset.groupby("Evaluation").mean()

    print(stats_df_dataset)

    stats_df_dataset.to_csv(os.path.join(data_path,"Stats_all_monoclass_evals_{}.csv".format(dataset_name)), columns= ["VP", "FP", "VN", "FN", "Kappa", "Dice", "Jaccard", "LCE", "GCE"])

    return df_dataset

def read_all_dataset_monoclass_evals():

    df_dataset_monoclass_evals = []

    for dataset in dataset_dirs:

        df_dataset = read_dataset_monoclass_eval(dataset)
        print(df_dataset)


        df_dataset_monoclass_evals.append(df_dataset)

    return pd.concat(df_dataset_monoclass_evals)

################################ df_dataset_multiclass_evals #######################################################


def read_dataset_multiclass_eval(dataset_name):

    data_dir = os.path.join(data_path, dataset_name)

    print(data_dir)

    res_eval_path = os.path.join(data_dir, "derivatives", "evaluation_results")

    try:
        os.makedirs(res_eval_path)
    except OSError:
        print("res_eval_path {} already exists".format(res_eval_path))

    csv_name = "multiclass_" + dataset_name + "_eval_res.csv"

    csv_file = os.path.join(res_eval_path, csv_name)

    if os.path.exists(csv_file):

        print ("reading {}".format(csv_name))

        df_dataset = pd.read_csv(csv_file)

        df_dataset["Dataset"] = [dataset_name]*len(df_dataset.index)

    else:

        print("Error, dataset {} was not computed, {} do not exists".format(dataset_name, csv_file))
        exit(-1)

    stats_df_dataset  = df_dataset.groupby("Evaluation").mean()

    print(stats_df_dataset)

    stats_df_dataset.to_csv(os.path.join(res_eval_path,"Stats_all_multiclass_evals_{}.csv".format(dataset_name)), columns= ["ICC", "VP", "FP", "VN", "FN", "Kappa", "Dice", "Jaccard"])

    return df_dataset


def read_all_dataset_multiclass_evals():

    df_dataset_multiclass_evals = []

    for dataset in dataset_dirs:

        df_dataset = read_dataset_multiclass_eval(dataset)
        print(df_dataset)

        df_dataset_multiclass_evals.append(df_dataset)

    return pd.concat(df_dataset_multiclass_evals)


if __name__ == '__main__':

    all_df = read_all_dataset_monoclass_evals()

    #stats_all_df  = all_df.groupby("Evaluation").mean()

    #print(stats_all_df)

    #stats_all_df.to_csv(os.path.join(data_path,"Stats_all_monoclass_evals.csv"), columns= ["VP", "FP", "VN", "FN", "Kappa", "Dice", "Jaccard", "LCE", "GCE"])

    all_df = read_all_dataset_multiclass_evals()

    #stats_all_df  = all_df.groupby("Evaluation").mean()

    #print(stats_all_df)

    #stats_all_df.to_csv(os.path.join(data_path,"Stats_all_multiclass_evals.csv"), columns= ["ICC", "VP", "FP", "VN", "FN", "Kappa", "Dice", "Jaccard"])

