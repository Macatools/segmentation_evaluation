
import os

import pandas as pd

from bids.layout import BIDSLayout

from eval_monoclass_seg import compute_all_monoclass_metrics

from eval_multiclass_seg import compute_all_multiclass_metrics

from define_variables import *




def eval_monoclass_metrics_dataset(dataset_name, man_analysis_name = "manual_segmentation", suffix = "space-orig_desc-brain_mask"):

    data_dir = os.path.join(data_path, dataset_name)

    layout = BIDSLayout(data_dir)

    # Verbose
    print("BIDS layout:", layout)
    subjects = layout.get_subjects()
    sessions = layout.get_sessions()

    print(subjects)
    print(sessions)

    res_eval_path = os.path.join(data_dir, "derivatives", "evaluation_results")

    try:
        os.makedirs(res_eval_path)
    except OSError:
        print("res_eval_path {} already exists".format(res_eval_path))


    results = []
    for sub in subjects:
        #if sub in ["032139", "032140", "032141", "032142"]:
        #    continue
        for ses in sessions:
            man_mask_file = os.path.join(data_dir, "derivatives", man_analysis_name, "sub-{}".format(sub), "ses-{}".format(ses), "anat", "sub-{}_ses-{}_{}.nii.gz".format(sub, ses, suffix))

            for auto_analysis_name in auto_analysis_names:
                auto_mask_file = os.path.join(data_dir, "derivatives", auto_analysis_name, "sub-{}".format(sub), "ses-{}".format(ses), "anat", "sub-{}_ses-{}_{}.nii.gz".format(sub, ses, suffix))

                eval_name = "manual-{}".format(auto_analysis_name)

                print("Comparing {} and {}".format(man_mask_file, auto_mask_file))

                list_res = compute_all_monoclass_metrics(
                    man_mask_file, auto_mask_file,
                    pref = os.path.join(res_eval_path, sub + "_" + ses + "_" + eval_name + "_"))

                list_res.insert(0, eval_name)
                list_res.insert(0, ses)
                list_res.insert(0, sub)

                results.append(list_res)


    #sub, ses, eval_name, VP, FP, VN, FN, kappa, dice, JC, LCE, GCE
    df = pd.DataFrame(results, columns = ["Subject", "Session", "Evaluation", "VP", "FP", "VN", "FN", "Kappa", "Dice", "Jaccard", "LCE", "GCE"])

    csv_name = "monoclass_" + dataset_name + "_eval_res.csv"

    df.to_csv(os.path.join(res_eval_path, csv_name))

    return df



def eval_multiclass_metrics_dataset(dataset_name, man_analysis_name = "manual_segmentation", suffix = "space-orig_desc-brain_dseg"):

    data_dir = os.path.join(data_path, dataset_name)

    layout = BIDSLayout(data_dir)

    # Verbose
    print("BIDS layout:", layout)
    subjects = layout.get_subjects()
    sessions = layout.get_sessions()

    print(subjects)
    print(sessions)

    res_eval_path = os.path.join(data_dir, "derivatives", "evaluation_results")

    try:
        os.makedirs(res_eval_path)
    except OSError:
        print("res_eval_path {} already exists".format(res_eval_path))


    results = []
    for sub in subjects:
        #if sub in ["032139", "032140", "032141", "032142"]:
        #    continue
        for ses in sessions:
            man_mask_file = os.path.join(data_dir, "derivatives", man_analysis_name, "sub-{}".format(sub), "ses-{}".format(ses), "anat", "sub-{}_ses-{}_{}.nii.gz".format(sub, ses, suffix))

            for auto_analysis_name in auto_analysis_names:
                auto_mask_file = os.path.join(data_dir, "derivatives", auto_analysis_name, "sub-{}".format(sub), "ses-{}".format(ses), "anat", "sub-{}_ses-{}_{}.nii.gz".format(sub, ses, suffix))

                eval_name = "manual-{}".format(auto_analysis_name)

                print("Comparing {} and {}".format(man_mask_file, auto_mask_file))

                list_res = compute_all_multiclass_metrics(
                    man_mask_file, auto_mask_file)

                list_res.insert(0, eval_name)
                list_res.insert(0, ses)
                list_res.insert(0, sub)

                results.append(list_res)


    #sub, ses, eval_name, VP, FP, VN, FN, kappa, dice, JC
    df = pd.DataFrame(results, columns = ["Subject", "Session", "Evaluation", "VP", "FP", "VN", "FN", "Kappa", "Dice", "Jaccard"])

    csv_name = "multiclass_" + dataset_name + "_eval_res.csv"

    df.to_csv(os.path.join(res_eval_path, csv_name))

    return df


if __name__ == '__main__':

    for dataset in dataset_dirs:

        df_dataset = eval_monoclass_metrics_dataset(dataset)
        df_dataset = eval_multiclass_metrics_dataset(dataset)

