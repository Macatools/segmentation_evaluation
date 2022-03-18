
import os
import glob

from itertools import product

import numpy as np
import nibabel as nib

import nipype.interfaces.afni as afni
import nipype.interfaces.fsl as fsl

from macapype.nodes.register import NwarpApplyPriors

from define_variables import *


def apply_xfm(in_file, xfm_file, ref_file):

    applyxfm = fsl.ApplyXFM()
    applyxfm.inputs.in_file = in_file
    applyxfm.inputs.in_matrix_file = xfm_file
    applyxfm.inputs.reference = ref_file
    applyxfm.inputs.apply_xfm = True

    output_file = applyxfm.run().outputs.out_file
    print(output_file)
    return output_file

def apply_warp(in_file, aff_file, warp_file):

    align_masks = NwarpApplyPriors()
    #align_masks = afni.NwarpApply()
    align_masks.inputs.in_file = in_file
    align_masks.inputs.out_file = in_file

    align_masks.inputs.interp = "NN"
    align_masks.inputs.args = "-overwrite"

    align_masks.inputs.master = aff_file
    align_masks.inputs.warp = warp_file

    output_file = align_masks.run().outputs.out_file
    print(output_file)
    return output_file


def alllineate_afni_template_space(in_file, ref_img_file, transfo_file):

    # align_NMT
    align_NMT = afni.Allineate()
    align_NMT.inputs.final_interpolation = "nearestneighbour"
    align_NMT.inputs.overwrite = True
    align_NMT.inputs.outputtype = "NIFTI_GZ"

    align_NMT.inputs.in_file = in_file
    align_NMT.inputs.reference = ref_img_file
    align_NMT.inputs.in_matrix = transfo_file

    output_file = align_NMT.run().outputs.out_file
    print(output_file)
    return output_file

def average_errors_ants(dataset_dir, ref_img_file):

    analysis_name = "ants"

    #workflow_name = "test_pipeline_single_indiv_params_{}".format(analysis_name) # old version of macapype
    #workflow_name = "macapype_indiv_params_crop_T1_T2_{}".format(analysis_name) # baboon (crop_T1 and T2)
    workflow_name = "fast/macapype_indiv_params_crop_T1_{}".format(analysis_name) # baboon (crop_T1 and T2)
    #workflow_name = "fast/macapype_indiv_params_crop_T1_T2_{}".format(analysis_name) # baboon (crop_T1 and T2)
    #workflow_name = "macapype_indiv_params_crop_T1_{}".format(analysis_name) # macaque_prime-de

    ref_img = nib.load(ref_img_file)
    ref_img_data = ref_img.get_fdata()

    error_type = "erreurEs"

    for suffix in [ "_fast"]:
    #for suffix in ["_no_priors", "_priors_w0", "_priors_w1", "_priors_w3", "_bug_NMT1.3better"]:

        full_analysis_name = analysis_name+suffix

        #for tissue_type in ["", "_1", "_2", "_3"]:
        for tissue_type in ["_1", "_2", "_3"]:

            for sub, ses in product(subjects, sessions):

                # warp
                in_file = os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/","{}_{}_manual-macapype_{}{}_{}.nii.gz".format(sub, ses, full_analysis_name, tissue_type, error_type))

                #ses+="_run-1" # when averaging was done, maybe required
                #fname_root = "sub-{}_ses-{}*_T1w_roi_noise_corrected_maths_masked_corrected".format(sub, ses)
                fname_root = "sub-{}_ses-{}*_T1w_roi_restore_debiased_brain_corrected".format(sub, ses)
                #fname_root = "sub-{}_ses-{}*_T1w_roi_corrected_restore_debiased_brain_corrected".format(sub, ses)
                #fname_root = "sub-{}_ses-{}*_T1w_roi_corrected_maths_masked_corrected".format(sub, ses) # erreur, should be denoised as well


                NMT_subject_align_path = os.path.join(data_path, dataset_dir, "{}/full_ants_subpipes/brain_segment_from_mask_pipe/register_NMT_pipe/_session_{}_subject_{}/NMT_subject_align/".format(workflow_name, ses, sub))

                glob_cmd = os.path.join(NMT_subject_align_path, "{}_affine.nii.gz".format(fname_root))

                aff_files = glob.glob(glob_cmd)
                assert len(aff_files) == 1, "Error with {}, glob({})".format(aff_files, glob_cmd)

                aff_file = aff_files[0]
                assert os.path.exists(aff_file), "Error, {} could not be found".format(aff_file)


                warp_files = glob.glob(os.path.join(NMT_subject_align_path, "{}_WARP.nii.gz".format(fname_root)))
                assert len(warp_files) == 1, "Error with {}".format(warp_files)
                warp_file = warp_files[0]
                assert os.path.exists(warp_file), "Error, {} could not be found".format(warp_file)

                warped_file = apply_warp(in_file, aff_file, warp_file)
                print(warped_file)

                # allineate
                transfo_files = glob.glob(os.path.join(NMT_subject_align_path, "{}_composite_linear_to_NMT.1D".format(fname_root)))
                assert len(transfo_files) == 1, "Error with {}".format(transfo_files)
                transfo_file = transfo_files[0]
                assert os.path.exists(transfo_file), "Error, {} could not be found".format(transfo_file)

                warped_allineate_file = alllineate_afni_template_space(in_file=warped_file, ref_img_file=ref_img_file , transfo_file=transfo_file )
                print(warped_allineate_file)

            ### counting errors
            count_errors_data = np.zeros(shape=ref_img_data.shape, dtype=float)

            print(count_errors_data.shape)

            for sub, ses in product(subjects, sessions):

                print(sub, ses, full_analysis_name, error_type)

                warped_allineate_file = "{}_{}_manual-macapype_{}{}_{}_Nwarp_allineate.nii.gz".format(sub, ses, full_analysis_name, tissue_type, error_type)

                warped_allineate = nib.load(warped_allineate_file).get_fdata()

                count_errors_data[warped_allineate==1.0] += 1.0

            print(np.unique(count_errors_data))

            nib.save(nib.Nifti1Image(count_errors_data, header = ref_img.header, affine = ref_img.affine),os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/", "count{}_manual-{}_{}.nii.gz".format(tissue_type, full_analysis_name, error_type)))

            nib.save(nib.Nifti1Image(count_errors_data / 5.0, header = ref_img.header, affine = ref_img.affine),os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/", "percent{}_manual-{}_{}.nii.gz".format(tissue_type, full_analysis_name, error_type)))


def average_errors_ants_t1(dataset_dir, ref_img_file):

    analysis_name = "ants_t1"


    #workflow_name = "test_pipeline_single_indiv_params_{}".format(analysis_name) # old version of macapype
    workflow_name = "macapype_indiv_params_crop_T1_{}".format(analysis_name)

    ref_img = nib.load(ref_img_file)
    ref_img_data = ref_img.get_fdata()

    for error_type in error_types:

        for tissue_type in ["", "_1", "_2", "_3"]:

            for sub, ses in product(subjects, sessions):

                # warp
                in_file = os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/","{}_{}_manual-macapype_{}{}_{}.nii.gz".format(sub, ses, analysis_name, tissue_type, error_type))

                #ses+="_run-1" # when averaging was done, maybe required
                fname_root = "sub-{}_ses-{}*_T1w_roi_noise_corrected_masked_corrected".format(sub, ses)
                #fname_root = "sub-{}_ses-{}*_T1w_roi_corrected_maths_masked_corrected".format(sub, ses) # erreur, should be denoised as well


                NMT_subject_align_path = os.path.join(data_path, dataset_dir, "{}/full_T1_ants_subpipes/brain_segment_from_mask_T1_pipe/register_NMT_pipe/_session_{}_subject_{}/NMT_subject_align/".format(workflow_name, ses, sub))

                aff_files = glob.glob(os.path.join(NMT_subject_align_path, "{}_affine.nii.gz".format(fname_root)))
                assert len(aff_files) == 1, "Error with {}".format(aff_files)
                aff_file = aff_files[0]
                assert os.path.exists(aff_file), "Error, {} could not be found".format(aff_file)


                warp_files = glob.glob(os.path.join(NMT_subject_align_path, "{}_WARP.nii.gz".format(fname_root)))
                assert len(warp_files) == 1, "Error with {}".format(warp_files)
                warp_file = warp_files[0]
                assert os.path.exists(warp_file), "Error, {} could not be found".format(warp_file)

                warped_file = apply_warp(in_file, aff_file, warp_file)
                print(warped_file)

                # allineate
                transfo_files = glob.glob(os.path.join(NMT_subject_align_path, "{}_composite_linear_to_NMT.1D".format(fname_root)))
                assert len(transfo_files) == 1, "Error with {}".format(transfo_files)
                transfo_file = transfo_files[0]
                assert os.path.exists(transfo_file), "Error, {} could not be found".format(transfo_file)

                warped_allineate_file = alllineate_afni_template_space(in_file=warped_file, ref_img_file=ref_img_file , transfo_file=transfo_file )
                print(warped_allineate_file)

            ### counting errors
            count_errors_data = np.zeros(shape=ref_img_data.shape, dtype=float)

            print(count_errors_data.shape)

            for sub, ses in product(subjects, sessions):

                print(sub, ses, analysis_name, error_type)

                warped_allineate_file = "{}_{}_manual-macapype_{}{}_{}_Nwarp_allineate.nii.gz".format(sub, ses, analysis_name, tissue_type, error_type)

                warped_allineate = nib.load(warped_allineate_file).get_fdata()

                count_errors_data[warped_allineate==1.0] += 1.0

            print(np.unique(count_errors_data))

            nib.save(nib.Nifti1Image(count_errors_data, header = ref_img.header, affine = ref_img.affine),os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/", "count{}_manual-{}_{}.nii.gz".format(tissue_type, analysis_name, error_type)))

            nib.save(nib.Nifti1Image(count_errors_data / 5.0, header = ref_img.header, affine = ref_img.affine),os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/", "percent{}_manual-{}_{}.nii.gz".format(tissue_type, analysis_name, error_type)))

def average_errors_spm_native(dataset_dir, ref_img_file):

    analysis_name = "spm_native"


    #workflow_name = "test_pipeline_single_indiv_params_{}".format(analysis_name) # old version of macapype
    workflow_name = "macapype_indiv_params_crop_T1_T2_{}".format(analysis_name)

    fname_root = "T1w_roi_corrected_debiased_BET_FLIRT-to_Haiko89_Asymmetric.Template_n89"
    #"T1w_roi_noise_corrected_debiased_BET_FLIRT-to_inia19-t1-brain"

    ref_img = nib.load(ref_img_file)
    ref_img_data = ref_img.get_fdata()

    for error_type in error_types:

        for tissue_type in ["", "_1", "_2", "_3"]:

            for sub, ses in product(subjects, sessions):

                # warp
                in_file = os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/","{}_{}_manual-macapype_{}{}_{}.nii.gz".format(sub, ses, analysis_name, tissue_type, error_type))

                assert os.path.exists(in_file)

                reg_path = os.path.join(data_path, dataset_dir, "{}/full_spm_subpipes/_session_{}_subject_{}/reg/".format(workflow_name, ses, sub))
                print(reg_path)

                xfm_files = glob.glob(os.path.join(reg_path, "sub-{}_ses-{}*_{}.xfm".format(sub, ses, fname_root)))
                assert len(xfm_files)==1,  "Error with {}".format(xfm_files)

                xfm_file=xfm_files[0]
                assert os.path.exists(xfm_file)

                warped_file = apply_xfm(in_file, xfm_file, ref_img_file)
                assert os.path.exists(warped_file)

                print (warped_file)

            ### counting errors
            count_errors_data = np.zeros(shape=ref_img_data.shape, dtype=float)

            print(count_errors_data.shape)

            for sub, ses in product(subjects, sessions):

                print(sub, ses, analysis_name, error_type)

                warped_file = "{}_{}_manual-macapype_{}{}_{}_flirt.nii.gz".format(sub, ses, analysis_name, tissue_type, error_type)

                warped_data = nib.load(warped_file).get_fdata()

                count_errors_data[warped_data==1.0] += 1.0

            print(np.unique(count_errors_data))

            nib.save(nib.Nifti1Image(count_errors_data, header = ref_img.header, affine = ref_img.affine),os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/", "count{}_manual-{}_{}.nii.gz".format(tissue_type, analysis_name, error_type)))

            nib.save(nib.Nifti1Image(count_errors_data / 5.0, header = ref_img.header, affine = ref_img.affine),os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/", "percent{}_manual-{}_{}.nii.gz".format(tissue_type, analysis_name, error_type)))



    #analysis_name = "spm_native"

    #for error_type in error_types:

        #for sub, ses in product(subjects, sessions):
            ## applyxfm
            #in_file = os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/","{}_{}_manual-macapype_{}_{}.nii.gz".format(sub, ses, analysis_name, error_type))

            #assert os.path.exists(in_file)

            #reg_path = os.path.join(data_path, dataset_dir, "/hpc/crise/meunier.d/Data/macaque_prime-de/macapype_indiv_params_crop_T1_{}/full_spm_subpipes/_session_{}_subject_{}/reg/".format(analysis_name, ses, sub))
            #xfm_file = os.path.join(reg_path, "sub-{}_ses-{}_run-1_T1w_roi_noise_corrected_debiased_BET_FLIRT-to_inia19-t1-brain.xfm".format(sub, ses))

            #assert os.path.exists(xfm_file)

            #warped_file = apply_xfm(in_file, xfm_file, ref_img_file)
            #assert os.path.exists(warped_file)

            #print (warped_file)

        #### counting errors
        #ref_img = nib.load(ref_img_file)

        #ref_img_data = ref_img.get_fdata()

        #count_errors_data = np.zeros(shape=ref_img_data.shape, dtype=float)
        #print(count_errors_data.shape)

        #for sub, ses in product(subjects, sessions):

            #print(sub, ses, analysis_name, error_type)

            #warped_file = "{}_{}_manual-macapype_{}_{}_flirt.nii.gz".format(sub, ses, analysis_name, error_type)

            #warped_data = nib.load(warped_file).get_fdata()

            #count_errors_data[warped_data==1.0] += 1.0

        #print(np.unique(count_errors_data))

        #nib.save(nib.Nifti1Image(count_errors_data, header = ref_img.header, affine = ref_img.affine),os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/", "count_manual-macapype_" + analysis_name + "_" + error_type + ".nii.gz"))

        #nib.save(nib.Nifti1Image(count_errors_data / 5.0, header = ref_img.header, affine = ref_img.affine),os.path.join(data_path, dataset_dir, "derivatives/evaluation_results/", "percent_manual-macapype_" + analysis_name + "_" + error_type + ".nii.gz"))

if __name__ == '__main__':

    dataset_dir = dataset_dirs[0]

    # macaque
    ## NMT_v1.3better for ants and ants_t1, inia19 for spm_native
    ref_img_file = "/home/meunier.d/data_macapype/NMT_v1.3better/NMT.nii.gz"
    #ref_img_file = "/home/meunier.d/data_macapype/inia19/inia19-brainmask.nii"

    ## baboon
    #ref_img_file = "/home/meunier.d/data_macapype/haiko89_template/Haiko89_Asymmetric.Template_n89.nii.gz"

    average_errors_ants(dataset_dir, ref_img_file)
    ##average_errors_ants_t1(dataset_dir, ref_img_file)
    #average_errors_spm_native(dataset_dir, ref_img_file)

