import os, glob
import nibabel as nib
import numpy as np
import json

from pad_back import padding_cropped_img
from reindex_tissues import merge_tissue

if __name__ == '__main__':

    from define_variables import tissue_tab, subjects, sessions, man_analysis_name, auto_analysis_names, data_path, dataset_dirs, man_tissue_tab, crop_file

    merge_hemi = 0
    reindex = 0
    pad_back = 1

    #dataset = "ucdavis2"
    dataset = "sbri"
    #dataset = "ucdavis"
    #dataset = "amu"
    #dataset = "sinai"

    ses = sessions[0]

    for sub in subjects:

        sub = str(sub)

        print("subject {}".format(sub))

        dataset_path = os.path.join(data_path, dataset_dirs[0])
        "/envau/work/nit/users/meunier.d/Data_BIDS/PrimeDE_ucdavisBIDS2/"

        #dataset_path = "/hpc/crise/meunier.d/Data/Baboon_Adults_Cerimed_Adrien/derivatives/semimanual_segmentation"
        #dataset_path = "/hpc/crise/meunier.d/Data/macaque_prime-de/semimanual_segmentation/sub-{}/ses-001/anat".format(sub)

        if merge_hemi:
            seg_file = os.path.join(dataset_path, "derivatives",  "{}/{}_{}_segmentation.nii.gz".format(man_analysis_name, dataset, sub.lower()))

            print("merge_hemi")

            # from baboon (and brainvisa format)
            #LH_seg_file = os.path.join(dataset_path, "derivatives", "{}/_session_{}_subject_{}_segmentation_LH_corrected.nii.gz".format(man_analysis_name, ses, sub))
            #RH_seg_file = os.path.join(dataset_path, "derivatives", "{}/_session_{}_subject_{}_segmentation_RH_corrected.nii.gz".format(man_analysis_name, ses, sub))

            # older (macaque cerimed, sbri)
            #LH_seg_file = os.path.join(dataset_path, "sub-{}/cerimed_{}_segmentation_LH_good.nii.gz".format(sub, sub.lower()))
            #RH_seg_file = os.path.join(dataset_path, "sub-{}/cerimed_{}_segmentation_RH_good.nii.gz".format(sub, sub.lower()))

            # ucdavis
            LH_seg_file = os.path.join(dataset_path, "derivatives", "{}/{}_{}_segmentation_LH_good.nii.gz".format(man_analysis_name, dataset, sub.lower()))
            RH_seg_file = os.path.join(dataset_path, "derivatives", "{}/{}_{}_segmentation_RH_good.nii.gz".format(man_analysis_name, dataset, sub.lower()))

            # sbri
            LH_seg_file = os.path.join(dataset_path, "derivatives", "{}/{}_{}_segmentation_LH.nii.gz".format(man_analysis_name, dataset, sub.lower()))
            RH_seg_file = os.path.join(dataset_path, "derivatives", "{}/{}_{}_segmentation_RH.nii.gz".format(man_analysis_name, dataset, sub.lower()))

            #LH_seg_file = os.path.join(dataset_path, "derivatives/semimanual_segmentation", "sub-{}/ucdavis_{}_segmentation_LH_good.nii.gz".format(sub, sub.lower()))
            #RH_seg_file = os.path.join(dataset_path, "derivatives/semimanual_segmentation", "sub-{}/ucdavis_{}_segmentation_RH_good.nii.gz".format(sub, sub.lower()))

            os.system("fslmaths {} -add {} {}".format(LH_seg_file, RH_seg_file, seg_file))
        else:

            seg_file = os.path.join(dataset_path, "derivatives", "{}/{}_{}_segmentation.nii.gz".format(man_analysis_name, dataset, sub.lower()))

        ## export file and dir
        data_dirpath = os.path.join(dataset_path, "derivatives","{}/sub-{sub}/ses-{ses}/anat/".format(man_analysis_name, sub=sub, ses=ses))

        if reindex:

            assert os.path.exists(seg_file), "{} not found".format(seg_file)

            try:
                os.makedirs(data_dirpath)
            except OSError:
                print("res_eval_path {} already exists".format(data_dirpath))

            cropped_img_file = os.path.join(data_dirpath, "sub-{sub}_ses-{ses}_space-native_desc-brain_dseg.nii.gz".format(sub=sub, ses=ses))

            #################### reindex

            merge_tissue(seg_file, cropped_img_file, tissue_tab = man_tissue_tab)
        else:

            cropped_img_file = os.path.join(data_dirpath, "sub-{sub}_ses-{ses}_space-native_desc-brain_dseg.nii.gz".format(sub=sub, ses=ses))



        #################### pad_back
        if pad_back:

            ### datapath
            #assert os.path.exists(cropped_img_file), "Error with {}".format(cropped_img_file)


            #indiv_crop_file = os.path.join(dataset_path,"indiv_params_segment_macaque_{}.json".format(dataset))
            indiv_crop_file = os.path.join(dataset_path,crop_file)

            assert os.path.exists(indiv_crop_file), "{} not found".format(indiv_crop_file)


            data_dict = json.load(open(indiv_crop_file))
            assert "sub-" + sub in data_dict.keys(), "{}".format(data_dict.keys())
            assert "ses-" + ses in data_dict["sub-" + sub].keys(), "{}".format(data_dict["sub-" + sub].keys())
            indiv_crop = data_dict["sub-" + sub]["ses-" + ses]
            print(indiv_crop)

            ## padding manual_dseg to orig T1w
            orig_files = glob.glob(os.path.join(dataset_path, "sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}*_T1w.nii*".format(sub=sub, ses=ses)))

            print(orig_files)

            assert len(orig_files)>0, "Not found BIDS T1w for {} {}, skipping".format(sub, ses)
            orig_file = orig_files[0]

            padded_img_file = padding_cropped_img(cropped_img_file, orig_file, indiv_crop, export_path=data_dirpath)
            print(padded_img_file)

