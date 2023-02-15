
import os
import glob

from itertools import product

import numpy as np
import nibabel as nib
import pandas as pd

import json
######## should be in macapype or even in nipype....

from nipype.interfaces.afni.base import (AFNICommandBase,
                                         AFNICommandOutputSpec,
                                         isdefined)

from nipype.utils.filemanip import split_filename as split_f

from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    TraitedSpec, traits, File)

# NwarpApplyPriors
class NwarpApplyPriorsInputSpec(CommandLineInputSpec):
    in_file = traits.Either(
        File(exists=True),
        traits.List(File(exists=True)),
        mandatory=True,
        argstr='-source %s',
        desc='the name of the dataset to be warped '
        'can be multiple datasets')
    warp = traits.String(
        desc='the name of the warp dataset. '
        'multiple warps can be concatenated (make sure they exist)',
        argstr='-nwarp %s',
        mandatory=True)
    inv_warp = traits.Bool(
        desc='After the warp specified in \'-nwarp\' is computed, invert it',
        argstr='-iwarp')
    master = traits.File(
        exists=True,
        desc='the name of the master dataset, which defines the output grid',
        argstr='-master %s')
    interp = traits.Enum(
        'wsinc5',
        'NN',
        'nearestneighbour',
        'nearestneighbor',
        'linear',
        'trilinear',
        'cubic',
        'tricubic',
        'quintic',
        'triquintic',
        desc='defines interpolation method to use during warp',
        argstr='-interp %s',
        usedefault=True)
    ainterp = traits.Enum(
        'NN',
        'nearestneighbour',
        'nearestneighbor',
        'linear',
        'trilinear',
        'cubic',
        'tricubic',
        'quintic',
        'triquintic',
        'wsinc5',
        desc='specify a different interpolation method than might '
        'be used for the warp',
        argstr='-ainterp %s')
    out_file = traits.Either(
        File(),
        traits.List(File()),
        mandatory=True,
        argstr='-prefix %s',
        desc='output image file name')
    short = traits.Bool(
        desc='Write output dataset using 16-bit short integers, rather than '
        'the usual 32-bit floats.',
        argstr='-short')
    quiet = traits.Bool(
        desc='don\'t be verbose :(', argstr='-quiet', xor=['verb'])
    verb = traits.Bool(
        desc='be extra verbose :)', argstr='-verb', xor=['quiet'])


class NwarpApplyPriorsOutputSpec(AFNICommandOutputSpec):
    out_file = traits.Either(
            File(),
            traits.List(File()))


class NwarpApplyPriors(AFNICommandBase):
    """
    Over Wrap of NwarpApply (afni node) in order to generate files in the right
    node directory (instead of in the original data directory, or the script
    directory as is now)

    Modifications are made over inputs and outputs
    """

    _cmd = '3dNwarpApply'
    input_spec = NwarpApplyPriorsInputSpec
    output_spec = NwarpApplyPriorsOutputSpec

    def _format_arg(self, name, spec, value):

        import os
        import shutil

        cur_dir = os.getcwd()

        new_value = []
        if name == 'in_file':
            if isinstance(value, list):

                print("A list for in_file")
                for in_file in value:
                    print(in_file)

                    # copy en local
                    shutil.copy(in_file, cur_dir)

                    new_value.append(os.path.join(cur_dir, in_file))

            else:
                print("A single file for in_file {}".format(value))

                if os.path.split(value)[1] not in os.listdir(cur_dir):
                    shutil.copy(value, cur_dir)

                    path, fname, ext = split_f(value)
                    new_value = os.path.join(cur_dir, fname + ext)
                    print(new_value)
                else:
                    new_value = value

            value = new_value

        elif name == 'out_file':
            if isinstance(value, list):
                print("A list for out_file")
                print("out_file:", value)

                for out_file in value[:1]:
                    print(out_file)

                    path, fname, ext = split_f(out_file)
                    new_value.append(os.path.join(cur_dir,
                                                  fname + "_Nwarp" + ext))

                for i in range(1, 4):
                    new_value.append(os.path.join(cur_dir,
                                                  "tmp_%02d.nii.gz" % i))

                print("after out_file:", new_value)

            else:
                print("A single file for out_file {}".format(value))

                path, fname, ext = split_f(value)
                new_value = os.path.abspath(fname + "_Nwarp" + ext)
                print(new_value)

            self.new_value = new_value
            value = new_value

        return super(NwarpApplyPriors, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.new_value):
            outputs['out_file'] = self.new_value
            print(outputs['out_file'])
        return outputs

##################################################
# applying transfos

import nipype.interfaces.afni as afni
import nipype.interfaces.fsl as fsl

from nipype.interfaces.niftyreg import regutils

def apply_aladin(in_file, ref_file, trans_file):

    apply_crop_aladdin = regutils.RegResample()

    apply_crop_aladdin.inputs.flo_file = in_file
    apply_crop_aladdin.inputs.ref_file = ref_file
    apply_crop_aladdin.inputs.trans_file = trans_file

    print(apply_crop_aladdin.cmdline)
    output_file = apply_crop_aladdin.run().outputs.out_file
    print(output_file)

    return output_file

def apply_crop_z(in_file, brainsize):

    apply_crop_z = fsl.RobustFOV()

    apply_crop_z.inputs.in_file = in_file
    apply_crop_z.inputs.brainsize = brainsize

    output_file = apply_crop_z.run().outputs.out_roi
    print(output_file)

    return output_file


def apply_crop(in_file, crop_args):

    apply_crop = fsl.ExtractROI()

    apply_crop.inputs.in_file = in_file
    apply_crop.inputs.args = crop_args

    output_file = apply_crop.run().outputs.roi_file
    print(output_file)

    return output_file


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

    print(align_masks.cmdline)
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


    print(align_NMT.cmdline)
    output_file = align_NMT.run().outputs.out_file
    print(output_file)

    return output_file


def apply_warp_and_1D(in_file, ref_file, warp_file, transfo_file):

    align_masks = NwarpApplyPriors()
    #align_masks = afni.NwarpApply()
    align_masks.inputs.in_file = in_file
    align_masks.inputs.out_file = in_file

    align_masks.inputs.interp = "NN"
    align_masks.inputs.args = "-overwrite"

    align_masks.inputs.master = ref_file
    align_masks.inputs.warp = warp_file + " " + transfo_file

    print(align_masks.cmdline)
    output_file = align_masks.run().outputs.out_file
    print(output_file)

    return output_file


####################################################################################################################################################################################
# averaging

def check_alt_wf(eval_path, sub, ses, analysis_name, tissue_type, error_type, alt_wf_roots = ["macapype_orig_{}", "macapype_indiv_params_{}"]):

    alt_wf_names = [alt_wf_root.format(analysis_name) for alt_wf_root in alt_wf_roots]

    for alt_wf_name in alt_wf_names:

        in_file = os.path.join(eval_path,"{}_{}_manual-{}{}_{}.nii.gz".format(sub, ses, alt_wf_name, tissue_type, error_type))


        if os.path.exists(in_file):
            print("Found {} for alt {}".format(in_file, alt_wf_name))
            cur_workflow_name = alt_wf_name
            return cur_workflow_name

    return False

def average_errors(data_path, working_path, subjects, sessions, dataset_dir, ref_img_file, template_aladin_file, crop_file,
                              analysis_name = "spm_native", suf = "T1w_res_ROI_restore_masked_corrected", error_type = "erreurMartin", NMT_version = "NMT_v2", keep_indiv = False, local_transfos = False):

    workflow_name = "macapype_crop_aladin_{}".format(analysis_name)

    if analysis_name == "ants_t1":
        pipeline_name = "full_T1_ants_subpipes"
        short_pipe_name = "short_preparation_T1_pipe"
        seg_pipe_name = "brain_segment_from_mask_T1_pipe"

    elif analysis_name == "ants":
        pipeline_name = "full_ants_subpipes"
        short_pipe_name = "short_preparation_pipe"
        seg_pipe_name = "brain_segment_from_mask_pipe"

    elif analysis_name == "spm_native":
        pipeline_name = "full_spm_subpipes"
        short_pipe_name = "short_preparation_pipe"

    dataset_path = os.path.join(data_path, dataset_dir)

    eval_path = os.path.join(dataset_path, "derivatives/new_evaluation_results/")
    assert os.path.exists(eval_path), "Error, {} not found".format(eval_path)

    res_dir = "average_errors_" + analysis_name

    if keep_indiv == False:
        res_dir += "_noindiv"

    res_path = os.path.join(eval_path, res_dir)

    try:
        os.makedirs(res_path)
    except OSError:
        print("res_path {} already exists".format(res_path))

    os.chdir(res_path)

    print(os.getcwd())


    if keep_indiv:
        alt_wf_roots = ["macapype_orig_{}", "macapype_indiv_params_{}"]
    else:
        alt_wf_roots = ["macapype_orig_{}"]

    ref_img = nib.load(ref_img_file)
    ref_img_data = ref_img.get_fdata()

    for tissue_type in ["_"]:

        list_avail_ses = []

        for sub, ses in product(subjects, sessions):

            print(("**** running average_errors_ants for {} {} and {}".format(sub, ses, analysis_name)))

            cur_workflow_name = workflow_name

            in_file = os.path.join(eval_path,"{}_{}_manual-{}{}_{}.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))

            if not os.path.exists(in_file):

                print ("changing in_file={} to alternative wf names (orig or indiv_params)".format(in_file))
                cur_workflow_name = check_alt_wf(eval_path=eval_path, sub=sub, ses=ses, analysis_name=analysis_name, tissue_type=tissue_type, error_type=error_type, alt_wf_roots=alt_wf_roots)
                if cur_workflow_name==0:

                    print( "error, {} does not exists in orig or indiv_params".format(in_file))
                    continue
                else:
                    print( "found {} for in_file".format(cur_workflow_name))
                    in_file = os.path.join(eval_path,"{}_{}_manual-{}{}_{}.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))


            print("cur_workflow_name = {}".format(cur_workflow_name))

            #### reg_transform if crop_aladin
            if cur_workflow_name.startswith("macapype_indiv_params_"):

                print("skipping reg_aladin aff step")

                indiv_crop_file = os.path.join(dataset_path,crop_file)

                assert os.path.exists(indiv_crop_file), "{} not found".format(indiv_crop_file)


                data_dict = json.load(open(indiv_crop_file))
                assert "sub-" + sub in data_dict.keys(), "{}".format(data_dict.keys())
                assert "ses-" + ses in data_dict["sub-" + sub].keys(), "{}".format(data_dict["sub-" + sub].keys())
                indiv_crop = data_dict["sub-" + sub]["ses-" + ses]
                print(indiv_crop)


                assert 'crop_T1' in indiv_crop.keys(), "Error, could not find crop_T1 in {}".format(indiv_crop.keys())
                assert 'args' in indiv_crop['crop_T1'].keys(), "Error, could not find args in {}".format(indiv_crop['crop_T1'].keys())

                cropped_file = apply_crop(in_file, indiv_crop['crop_T1']['args'])


                fname_root = "sub-{}_ses-{}*_T1w_roi_{}".format(sub, ses, suf)

            elif cur_workflow_name.startswith("macapype_crop_aladin_") or cur_workflow_name.startswith("macapype_orig_"):

                if local_transfos:

                    ## copying in local
                    glob_cmd = os.path.join(working_path, cur_workflow_name, "only_transfos", "crop_aladin_T1", "sub-{sub}_ses-{ses}*_T1w_aff.txt".format(ses=ses, sub=sub))

                else:

                    ## copying full analysis
                    glob_cmd = os.path.join(working_path, cur_workflow_name, pipeline_name, short_pipe_name, "_session_{ses}_subject_{sub}/crop_aladin_T1/*sub-{sub}_ses-{ses}*_T1w_aff.txt".format(ses=ses, sub=sub))


                # reg_transform
                aladdin_files = glob.glob(glob_cmd)

                assert len(aladdin_files) == 1, "Error with {}, glob({})".format(aladdin_files, glob_cmd)

                aladdin_file = aladdin_files[0]
                assert os.path.exists(aladdin_file), "Error, {} could not be found".format(aladdin_file)

                reg_file = apply_aladin(in_file=in_file, ref_file=template_aladin_file, trans_file=aladdin_file)
                print(reg_file)


                cropped_file = apply_crop_z(in_file=reg_file, brainsize = brainsize)
                print(cropped_file)

                if analysis_name == "spm_native":

                    fname_root = "sub-{}_ses-{}*_T1w_res_ROI_{}".format(sub, ses, suf)

                elif analysis_name in ['ants', 'ants_t1']:

                    fname_root = "sub-{}_ses-{}*_T1w_res_ROI_{}".format(sub, ses, suf)

            if analysis_name == "spm_native":

                    glob_cmd = os.path.join(working_path, "{}/full_spm_subpipes/_session_{}_subject_{}/reg/*{}.xfm".format(cur_workflow_name, ses, sub, fname_root))
                    print(glob_cmd)

                    xfm_files = glob.glob(glob_cmd)
                    assert len(xfm_files)==1,  "Error with {}".format(glob_cmd)

                    xfm_file=xfm_files[0]
                    assert os.path.exists(xfm_file)

                    warped_file = apply_xfm(cropped_file, xfm_file, ref_img_file)
                    assert os.path.exists(warped_file)

                    print (warped_file)

            elif analysis_name in ['ants', 'ants_t1']:

                ########################################################################### new version
                if NMT_version == "animal_warp":

                    ## copying full analysis
                    if local_transfos:
                        # copying in local
                        NMT_subject_align_path = os.path.join(working_path, cur_workflow_name, "only_transfos", "NMT_subject_align".format(ses, sub))

                    else:
                        NMT_subject_align_path = os.path.join(working_path, cur_workflow_name, pipeline_name, seg_pipe_name, "register_NMT_pipe/_session_{}_subject_{}/NMT_subject_align/aw_results".format(ses, sub))

                    ## Nwarp and then allineate

                    ## target file
                    #glob_cmd = os.path.join(NMT_subject_align_path, "*{}_ns.nii.gz".format(fname_root))
                    #aff_files = glob.glob(glob_cmd)
                    #assert len(aff_files) == 1, "Error with {}, glob({})".format(aff_files, glob_cmd)
                    #aff_file = aff_files[0]
                    #assert os.path.exists(aff_file), "Error, {} could not be found".format(aff_file)

                    transfo_files = glob.glob(os.path.join(NMT_subject_align_path, "*{}_composite_linear_to_template.1D".format(fname_root)))
                    assert len(transfo_files) == 1, "Error with {}".format(transfo_files)
                    transfo_file = transfo_files[0]
                    assert os.path.exists(transfo_file), "Error, {} could not be found".format(transfo_file)


                    # warped file
                    warp_files = glob.glob(os.path.join(NMT_subject_align_path, "*{}_shft_WARP.nii.gz".format(fname_root)))
                    assert len(warp_files) == 1, "Error with {}".format(os.path.join(NMT_subject_align_path, "*{}_shft_WARP.nii.gz".format(fname_root)))
                    warp_file = warp_files[0]
                    assert os.path.exists(warp_file), "Error, {} could not be found".format(os.path.join(NMT_subject_align_path, "*{}_shft_WARP.nii.gz".format(fname_root)))

                    warped_file = apply_warp_and_1D(cropped_file, ref_file=ref_img_file, warp_file= warp_file, transfo_file = transfo_file)
                    print(warped_file)

                    ## allineate file
                    #warped_allineate_file = alllineate_afni_template_space(in_file=warped_file, ref_img_file=ref_img_file , transfo_file=transfo_file )
                    #print(warped_allineate_file)

                ################################################################################## old version
                elif NMT_version == "NMT_v2.0":

                    NMT_subject_align_path = os.path.join(working_path, cur_workflow_name, pipeline_name, seg_pipe_name, "register_NMT_pipe/_session_{}_subject_{}/NMT_subject_align".format(ses, sub))

                    glob_cmd = os.path.join(NMT_subject_align_path, "*{}_affine.nii.gz".format(fname_root))

                    aff_files = glob.glob(glob_cmd)
                    assert len(aff_files) == 1, "Error with {}, glob({})".format(aff_files, glob_cmd)

                    aff_file = aff_files[0]
                    assert os.path.exists(aff_file), "Error, {} could not be found".format(aff_file)


                    warp_files = glob.glob(os.path.join(NMT_subject_align_path, "*{}_WARP.nii.gz".format(fname_root)))
                    assert len(warp_files) == 1, "Error with {}".format(warp_files)
                    warp_file = warp_files[0]
                    assert os.path.exists(warp_file), "Error, {} could not be found".format(warp_file)

                    warped_file = apply_warp(cropped_file, aff_file, warp_file)
                    print(warped_file)

                    # allineate
                    transfo_files = glob.glob(os.path.join(NMT_subject_align_path, "*{}_composite_linear_to_NMT.1D".format(fname_root)))
                    assert len(transfo_files) == 1, "Error with {}".format(transfo_files)
                    transfo_file = transfo_files[0]
                    assert os.path.exists(transfo_file), "Error, {} could not be found".format(transfo_file)

                    warped_allineate_file = alllineate_afni_template_space(in_file=warped_file, ref_img_file=ref_img_file , transfo_file=transfo_file )
                    print(warped_allineate_file)

                else:
                    print("Error, NMT_version should be either animal_warp or NMT_v2.0")
                    exit(-1)

            list_avail_ses.append((sub, ses, cur_workflow_name))

        print("************************** Counting errors for all subjects *******************************************************************")
        #### counting errors
        count_shape = ref_img_data.shape

        if len(count_shape) == 4:
            count_shape = count_shape[:3]

        count_errors_data = np.zeros(shape=count_shape, dtype=float)

        print(count_errors_data.shape)

        for sub, ses, cur_workflow_name in list_avail_ses:

            print(sub, ses, analysis_name, error_type)


            if analysis_name in ["ants", "ants_t1"]:
                if cur_workflow_name.startswith("macapype_indiv_params_"):
                    norm_file = os.path.abspath("{}_{}_manual-{}{}_{}_roi_Nwarp.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))
                else:
                    norm_file= os.path.abspath("{}_{}_manual-{}{}_{}_res_ROI_Nwarp.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))

            elif analysis_name == "spm_native":

                if cur_workflow_name.startswith("macapype_indiv_params_"):
                    #TODO
                    norm_file = os.path.abspath("{}_{}_manual-{}{}_{}_roi_flirt.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))

                else:
                    norm_file = os.path.abspath("{}_{}_manual-{}{}_{}_res_ROI_flirt.nii.gz".format(sub, ses, cur_workflow_name, tissue_type, error_type))

            assert os.path.exists(norm_file), "error with file {}".format(norm_file)

            warped = nib.load(norm_file).get_fdata()

            count_errors_data += warped

        print(np.unique(count_errors_data))

        nib.save(nib.Nifti1Image(count_errors_data, header = ref_img.header, affine = ref_img.affine),os.path.join(res_path, "count{}_manual-{}_{}.nii.gz".format(tissue_type, analysis_name, error_type)))

        nib.save(nib.Nifti1Image(count_errors_data / float(len(list_avail_ses)), header = ref_img.header, affine = ref_img.affine),os.path.join(res_path, "percent{}_manual-{}_{}.nii.gz".format(tissue_type, analysis_name, error_type)))

        df_avail_ses = pd.DataFrame(list_avail_ses, columns = ["subject", "Session", "Workflow"])
        df_avail_ses.to_csv(os.path.join(res_path, "avail_wf_{}.csv".format(analysis_name)))


#######################################################################################################################################################################################################
# main

from define_variables import data_path, working_path, subjects, sessions, dataset_dirs, brainsize, analysis_names
from define_variables import tab_ref_img, template_aladin_file, tabs_suf, NMT_version, crop_file, local_transfos

if __name__ == '__main__':

    dataset_dir = dataset_dirs[0]

    #analysis_name= "ants"

    #average_errors(
            #data_path=data_path,
            #working_path=working_path,
            #subjects=subjects,
            #sessions=sessions,
            #dataset_dir=dataset_dir,
            #ref_img_file=tab_ref_img[analysis_name],
            #template_aladin_file=template_aladin_file,
            #crop_file=crop_file,
            #analysis_name=analysis_name,
            #suf = tabs_suf[analysis_name],
            #NMT_version=NMT_version,
            #keep_indiv = False,
            #local_transfos = local_transfos)

    analysis_name= "ants_t1"

    average_errors(
            data_path=data_path,
            working_path=working_path,
            subjects=subjects,
            sessions=sessions,
            dataset_dir=dataset_dir,
            ref_img_file=tab_ref_img[analysis_name],
            #template_aladin_file="/home/meunier.d/data_macapype/NMT_v2.0_asym/NMT_v20_asym.nii.gz", # this is a bug, should be corrected
            template_aladin_file=template_aladin_file,
            crop_file=crop_file,
            analysis_name=analysis_name,
            suf = tabs_suf[analysis_name],
            NMT_version=NMT_version, keep_indiv = False,
            local_transfos=local_transfos)

    #analysis_name= "spm_native"

    #average_errors(
            #data_path=data_path,
            #working_path=working_path,
            #subjects=subjects,
            #sessions=sessions,
            #dataset_dir=dataset_dir,
            #ref_img_file=tab_ref_img[analysis_name],
            #template_aladin_file=template_aladin_file,
            #crop_file=crop_file,
            #analysis_name=analysis_name,
            #suf = tabs_suf[analysis_name],
            #NMT_version=NMT_version, keep_indiv = False, local_transfos = False)