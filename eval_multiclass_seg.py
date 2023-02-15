# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:11:05 2021

@author: essmaile
modif David Meunier 07/01/2022

"""

#### Importation --------------------------------------------------------------------------------------------------------
import nibabel as nib
import numpy as np

import os

from metrics import intraclass_correlation, perf_measure_seg_multimodale, coff_kappa, coff_dice, coff_jaccard, erreurs_martin, carte_erreur_martin, erreurs_martin_multiclasse

def compute_all_multiclass_metrics (seg_auto,seg_ref, pref = "img_") :

    assert os.path.exists(seg_auto)
    assert os.path.exists(seg_ref)

    img_set_auto =  nib.load(seg_auto)

    data_set_auto =  np.array(np.round(img_set_auto.get_data()), dtype = 'int')
    data_set_ref =  np.array(np.round(nib.load(seg_ref).get_data()), dtype = 'int')



    x, y, z = data_set_ref.shape

    # convertir array 3D to 1D
    auto = data_set_auto.reshape(-1)
    ref = data_set_ref.reshape(-1)

    assert len(auto) == len(ref), "error with orig shapes {} and {}".format(data_set_auto.shape, data_set_ref.shape)
    # Nombre total des voxels
    N = len(auto)

    ICC=0
    ICC = intraclass_correlation (seg_auto, seg_ref)
    print("ICC : ", ICC)



    VP, FP, VN, FN = perf_measure_seg_multimodale(auto, ref)
    print("vrai_positif : ",VP)
    print("faux_positif : ",FP)
    print("vrai_négatif : ",VN)
    print("faux_négatif : ",FN)

    kappa = coff_kappa(VP, FP, VN, FN, N)
    print("kappa : ", kappa)

    dice = coff_dice(VP, FP, VN, FN)
    print("dice : ", dice)


    JC = coff_jaccard(VP, FP, VN, FN)
    print("jaccard : ", JC)


    LCE, GCE, Emin = erreurs_martin_multiclasse(auto, ref)
    carteEmin=Emin.reshape(x, y, z)
    imaEmin = nib.Nifti1Image(carteEmin, img_set_auto.affine)
    res_file = pref + '_erreurMartin.nii.gz'
    nib.save(imaEmin, res_file)

    return [ICC, VP, FP, VN, FN, kappa, dice, JC, LCE, GCE]



#### Main ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #### data préparation ---------------------------------------------------------------------------------------------------
    # téléchargement des données NIFTI

    #Segmentation_auto = nib.load("/hpc/meca/users/essamlali.a/manuel_segmentation/sinia_seg_ref/sub-032155_ses-001_run-1_brain_mask_T2.nii.gz")
    #Segmentation_ref = nib.load("/hpc/meca/users/essamlali.a/manuel_segmentation/sinia_seg_ref/sub-032155_ses-001_run-1_brain_mask_T2.nii.gz")

    #seg_auto = nib.load("D:/Stage_INT/Script_metriques/Cube1.nii.gz")
    #seg_ref = nib.load("D:/Stage_INT/Script_metriques/Cube3.nii.gz")

    #seg_auto = "/hpc/crise/meunier.d/Data/Baboon_Adults_Cerimed_Adrien/derivatives/semimanual_segmentation/sub-Arthur/ses-01/anat/sub-Arthur_ses-01_space-orig_desc-brain_dseg.nii.gz"
    #seg_ref = "/hpc/crise/meunier.d/Data/Baboon_Adults_Cerimed_Adrien/derivatives/macapype_ants/sub-Arthur/ses-01/anat/sub-Arthur_ses-01_space-orig_desc-brain_dseg.nii.gz"

    seg_auto = "/Users/olivier/Downloads/autoCrop.nii.gz"
    seg_ref = "/Users/olivier/Downloads/manualCrop.nii.gz"

    compute_all_multiclass_metrics (seg_auto,seg_ref)






