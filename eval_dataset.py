
import os

import pandas as pd

from bids.layout import BIDSLayout

from eval_multiclass_seg import compute_all_multiclass_metrics

from define_variables import bids_path, data_path, dataset_dirs, auto_analysis_names, man_analysis_name, subjects, sessions, tab_auto_suffix, man_suffix

def new_eval_multiclass_metrics_dataset(dataset_name, man_analysis_name = "manual_segmentation", tab_auto_suffix = {}, man_suffix = "space-native_desc-brain_dseg_padded", output_dir = "new_evaluation_results"):


    data_dir = os.path.join(data_path, dataset_name)

    res_eval_path = os.path.join(data_dir, "derivatives", output_dir)

    try:
        os.makedirs(res_eval_path)
    except OSError:
        print("res_eval_path {} already exists".format(res_eval_path))

    results = []
    for sub in subjects:
        for ses in sessions:

            print("**** Running multiclass sub {} ses {} ****".format(sub, ses))

            man_mask_file = os.path.join(data_dir, "derivatives", man_analysis_name, "sub-{}".format(sub), "ses-{}".format(ses), "anat", "sub-{}_ses-{}_{}.nii.gz".format(sub, ses, man_suffix))

            for auto_analysis_name in auto_analysis_names:

                if len(tab_auto_suffix.keys())==0:
                    auto_suffix = "space-native_desc-brain_dseg"
                elif auto_analysis_name in tab_auto_suffix.keys():
                    auto_suffix = tab_auto_suffix[auto_analysis_name]
                else:
                    print("Error could not find {} in {}".format(auto_analysis_name,tab_auto_suffix.keys()))

                auto_mask_file = os.path.join(data_dir, "derivatives", auto_analysis_name, "sub-{}".format(sub), "ses-{}".format(ses), "anat", "sub-{}_ses-{}_{}.nii.gz".format(sub, ses, auto_suffix))

                if os.path.exists(auto_mask_file) and os.path.exists(man_mask_file):

                    eval_name = "manual-{}".format(auto_analysis_name)

                    print("Comparing multiclass {} and {}".format(man_analysis_name, auto_analysis_name))

                    list_res = compute_all_multiclass_metrics(
                        man_mask_file, auto_mask_file,
                        pref = os.path.join(res_eval_path, sub + "_" + ses + "_" + eval_name + "_"))

                    list_res.insert(0, eval_name)
                    list_res.insert(0, ses)
                    list_res.insert(0, sub)

                    results.append(list_res)
                else:
                    print("Skipping, {} and/or {} are missing".format(auto_mask_file, man_mask_file))


    #sub, ses, eval_name, VP, FP, VN, FN, kappa, dice, JC
    df = pd.DataFrame(results, columns = ["Subject", "Session", "Evaluation", "ICC", "VP", "FP", "VN", "FN", "Kappa", "Dice", "Jaccard", "LCE", "GCE"])

    csv_name = "multiclass_eval_res.csv"

    df.to_csv(os.path.join(res_eval_path, csv_name))

    return df


if __name__ == '__main__':

    for dataset in dataset_dirs:

        print(dataset)

        df_dataset = new_eval_multiclass_metrics_dataset(dataset_name=dataset, man_analysis_name = man_analysis_name, tab_auto_suffix=tab_auto_suffix, man_suffix=man_suffix)


