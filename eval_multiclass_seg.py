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

from metrics import intraclass_correlation, perf_measure_seg_multimodale, coff_kappa, coff_dice, coff_jaccard, erreurs_martin, carte_erreur_martin

def compute_all_multiclass_metrics (seg_auto,seg_ref, pref = "img_") :
    
    assert os.path.exists(seg_auto)
    assert os.path.exists(seg_ref)
    
    img_set_auto =  nib.load(seg_auto)
    data_set_auto =  img_set_auto.get_data()

    data_set_ref =  nib.load(seg_ref).get_data()

    x, y, z = data_set_ref.shape
    
    # convertir array 3D to 1D
    auto = data_set_auto.reshape(-1)
    ref = data_set_ref.reshape(-1)

    # Nombre total des voxels 
    N = len(auto)

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
    
    #looping over tissues
    print (np.unique(auto))
    print (np.unique(ref))
    
    assert np.all(np.unique(auto) == np.unique(ref))
    
    for index in np.unique(auto)[1:]:
        
        auto_ind = np.zeros(shape=auto.shape, dtype=int)
        auto_ind[auto==index] = 1
        print(np.sum(auto_ind==1))
              
        ref_ind = np.zeros(shape = ref.shape, dtype=int)
        ref_ind[ref==index] = 1
        print(np.sum(ref_ind==1))
     
        LCE, GCE, Es, Es_prim = erreurs_martin(auto_ind, ref_ind, N)
        print("LCE : ", LCE)
        print("GCE : ", GCE)

    
        carte_erreurEs = carte_erreur_martin (Es, x, y, z)
        carte_erreurEs_prim = carte_erreur_martin (Es_prim, x, y, z)
        
        img_erreurEs = nib.Nifti1Image(carte_erreurEs, img_set_auto.affine)
        nib.save(img_erreurEs, pref + str(int(index)) + '_erreurEs.nii.gz')

        img_erreurEs_prim = nib.Nifti1Image(carte_erreurEs_prim, img_set_auto.affine)
        nib.save(img_erreurEs_prim, pref+ str(int(index)) + '_erreurEs_prim.nii.gz')
    
    return [ICC, VP, FP, VN, FN, kappa, dice, JC]



#### Main ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    #### data préparation ---------------------------------------------------------------------------------------------------
    # téléchargement des données NIFTI

    #Segmentation_auto = nib.load("/hpc/meca/users/essamlali.a/manuel_segmentation/sinia_seg_ref/sub-032155_ses-001_run-1_brain_mask_T2.nii.gz")
    #Segmentation_ref = nib.load("/hpc/meca/users/essamlali.a/manuel_segmentation/sinia_seg_ref/sub-032155_ses-001_run-1_brain_mask_T2.nii.gz")

    #seg_auto = nib.load("D:/Stage_INT/Script_metriques/Cube1.nii.gz")
    #seg_ref = nib.load("D:/Stage_INT/Script_metriques/Cube3.nii.gz")
    
    seg_auto = "/hpc/crise/meunier.d/Data/Baboon_Adults_Cerimed_Adrien/derivatives/semimanual_segmentation/sub-Arthur/ses-01/anat/sub-Arthur_ses-01_space-orig_desc-brain_dseg.nii.gz"
    seg_ref = "/hpc/crise/meunier.d/Data/Baboon_Adults_Cerimed_Adrien/derivatives/macapype_ants/sub-Arthur/ses-01/anat/sub-Arthur_ses-01_space-orig_desc-brain_dseg.nii.gz"
    
    compute_all_multiclass_metrics (seg_auto,seg_ref)






