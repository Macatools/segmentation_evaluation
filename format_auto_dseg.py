import os, glob
import nibabel as nib
import numpy as np
import json

from pad_back import padding_cropped_img

from reindex_tissues import merge_tissue

########################################################################################################################################################################################

if __name__ == '__main__':

    from define_variables import tissue_tab, subjects, sessions, auto_analysis_names, data_path, dataset_dirs

    ses = sessions[0]

    for sub in subjects:

        sub = str(sub)

        print("subject {}".format(sub))

        dataset_path = os.path.join(data_path, dataset_dirs[0])
        "/envau/work/nit/users/meunier.d/Data_BIDS/PrimeDE_ucdavisBIDS2/"

        #dataset_path = "/hpc/crise/meunier.d/Data/Baboon_Adults_Cerimed_Adrien/derivatives/semimanual_segmentation"
        #dataset_path = "/hpc/crise/meunier.d/Data/macaque_prime-de/semimanual_segmentation/sub-{}/ses-001/anat".format(sub)

        for analysis_name in auto_analysis_names:

                print("subject {}, merge_tissue for auto analysis_name {} with tab {}".format(sub, analysis_name, tissue_tab))

                auto_dseg_file = os.path.join(dataset_path, "derivatives/{analysis_name}/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_space-native_desc-brain_dseg.nii.gz".format(analysis_name=analysis_name, sub=sub, ses=ses))

                ### reindex auto dseg
                new_auto_dseg_file = os.path.join(dataset_path, "derivatives/{analysis_name}/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_space-native_desc-brain_desc-3classes_dseg.nii.gz".format(analysis_name=analysis_name, sub=sub, ses=ses))

                if os.path.exists(auto_dseg_file):
                    merge_tissue(auto_dseg_file, new_auto_dseg_file, tissue_tab = tissue_tab)
                else:
                    print("Not found {}, Skipping {} {}".format(auto_dseg_file, analysis_name, sub))
