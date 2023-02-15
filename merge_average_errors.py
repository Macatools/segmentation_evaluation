
import os
import glob

from itertools import product

import numpy as np
import nibabel as nib
import pandas as pd

####################################################################################################################################################################################

def merge_all_average_errors(data_path, analysis_names, ref_img_file, datasets,  keep_indiv = False, error_type = "erreurMartin", tissue_type = "_"):

        ref_img = nib.load(ref_img_file)
        ref_img_data = ref_img.get_fdata()

        #### counting errors
        count_shape = ref_img_data.shape

        if len(count_shape) == 4:
            count_shape = count_shape[:3]

        count_errors_data = np.zeros(shape=count_shape, dtype=float)

        print(count_errors_data.shape)

        if keep_indiv:
            res_path = os.path.join(data_path, "average_errors_" + analysis_name)

        else:
            res_path = os.path.join(data_path, "average_errors_" + analysis_name + "_noindiv")

        count_img = 0
        for dataset in datasets:

            df_file = os.path.join(res_path, "avail_wf_{}_{}.csv".format(dataset, analysis_name))

            assert os.path.exists(df_file), "Error with df_file {}".format(df_file)

            df = pd.read_csv(df_file, dtype = "str")

            print(df)

            for sub, ses, cur_workflow_name in  list(zip(df.subject, df.Session, df.Workflow)):

                print(sub, ses, analysis_name, error_type)


                if analysis_name in ["ants", "ants_t1"]:
                    if cur_workflow_name.startswith("macapype_indiv_params_"):
                        norm_file = os.path.join(res_path, "{}_{}_manual-{}{}_{}_roi_Nwarp_allineate.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))

                    else:
                        norm_file= os.path.join(res_path, "{}_{}_manual-{}{}_{}_res_ROI_Nwarp_allineate.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))

                elif analysis_name == "spm_native":

                    if cur_workflow_name.startswith("macapype_indiv_params_"):
                        #TODO
                        norm_file = os.path.join(res_path, "{}_{}_manual-{}{}_{}_roi_flirt.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))

                    else:
                        norm_file = os.path.join(res_path, "{}_{}_manual-{}{}_{}_res_flirt.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))

                assert os.path.exists(norm_file), "error with file {}".format(norm_file)

                warped = nib.load(norm_file).get_fdata()

                count_errors_data += warped

                count_img+=1


        nib.save(nib.Nifti1Image(count_errors_data, header = ref_img.header, affine = ref_img.affine),os.path.join(res_path, "count{}_manual-{}_{}.nii.gz".format(tissue_type, analysis_name, error_type)))

        nib.save(nib.Nifti1Image(count_errors_data / float(count_img), header = ref_img.header, affine = ref_img.affine),os.path.join(res_path, "percent{}_manual-{}_{}.nii.gz".format(tissue_type, analysis_name, error_type)))


#######################################################################################################################################################################################################
# main
from define_variables import data_path, analysis_names, tab_ref_img

if __name__ == '__main__':


    data_path = "/envau/work/nit/users/meunier.d/Data_BIDS/PrimeDE_all_macaque"

    datasets = ["ucdavis", "ucdavis2", 'sbri', 'cerimed']

    for analysis_name in analysis_names:

        merge_all_average_errors(data_path, analysis_name, tab_ref_img[analysis_name], datasets)

