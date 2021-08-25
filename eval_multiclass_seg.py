# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:11:05 2021

@author: essmaile
"""




#### Importation --------------------------------------------------------------------------------------------------------
import nibabel as nib
from metrics import intraclass_correlation, perf_measure_seg_multimodale, coff_kappa, coff_dice, coff_jaccard





def compute_all_multiclass_metrics (seg_auto,seg_ref) :
    
    img_set_auto =  nib.load(seg_auto)
    data_set_auto =  img_set_auto.get_data()

    data_set_ref =  nib.load(seg_ref).get_data()

    # convertir array 3D to 1D
    tab1 = data_set_auto.reshape(-1)
    tab2 = data_set_ref.reshape(-1)

    # Nombre total des voxels 
    N = len(tab1)


    ICC = intraclass_correlation (seg_auto, seg_ref)
    print("ICC : ", ICC)

    VP, FP, VN, FN = perf_measure_seg_multimodale(tab1, tab2)
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
    
    return [ICC, VP, FP, VN, FN, kappa, dice, JC]



#### Main ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    #### data préparation ---------------------------------------------------------------------------------------------------
    # téléchargement des données NIFTI

    #Segmentation_auto = nib.load("/hpc/meca/users/essamlali.a/manuel_segmentation/sinia_seg_ref/sub-032155_ses-001_run-1_brain_mask_T2.nii.gz")
    #Segmentation_ref = nib.load("/hpc/meca/users/essamlali.a/manuel_segmentation/sinia_seg_ref/sub-032155_ses-001_run-1_brain_mask_T2.nii.gz")


    seg_auto = nib.load("D:/Stage_INT/Script_metriques/Cube1.nii.gz")
    seg_ref = nib.load("D:/Stage_INT/Script_metriques/Cube3.nii.gz")
    
    compute_all_multiclass_metrics (seg_auto,seg_ref)






