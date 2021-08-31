
import os

import pandas as pd

from bids.layout import BIDSLayout

from eval_dataset import eval_monoclass_metrics_dataset

from define_variables import *




def read_dataset_monoclass_eval(dataset_name):

    data_dir = os.path.join(data_path, dataset_name)

    print(data_dir)

    if os.path.exists(data_dir):

        res_eval_path = os.path.join(data_dir, "derivatives", "evaluation_results")

        csv_name = "monoclass_" + dataset_name + "_eval_res.csv"

        df_dataset = pd.read_csv(os.path.join(res_eval_path, csv_name))

        df_dataset["Dataset"] = [dataset_name]*len(df_dataset.index)

    else:

        df_dataset = eval_monoclass_metrics_dataset(dataset_name)

    return df_dataset

def read_all_dataset_monoclass_evals():

    df_dataset_monoclass_evals = []

    for dataset in dataset_dirs:

        df_dataset = read_dataset_monoclass_eval(dataset)
        print(df_dataset)

        df_dataset_monoclass_evals.append(df_dataset)

    all_df = pd.concat(df_dataset_monoclass_evals)
    print(all_df)

    return all_df


if __name__ == '__main__':

    all_df = read_all_dataset_monoclass_evals ()

    stats_all_df  = all_df.groupby("Evaluation").mean()

    print(stats_all_df)

    stats_all_df.to_csv(os.path.join(data_path,"Stats_all_monoclass_evals.csv"), columns= ["VP", "FP", "VN", "FN", "Kappa", "Dice", "Jaccard", "LCE", "GCE"])

