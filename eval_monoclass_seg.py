# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:15:55 2021

@author: essmaile
"""

#### Importation --------------------------------------------------------------------------------------------------------
import nibabel as nib
from metrics import perf_measure_seg_monomodale, coff_kappa, coff_dice, coff_jaccard, erreurs_martin, carte_erreur_martin



#### data préparation ---------------------------------------------------------------------------------------------------
# téléchargement des données NIFTI

#Segmentation_auto = nib.load("/hpc/meca/users/essamlali.a/manuel_segmentation/sinia_seg_ref/sub-032155_ses-001_run-1_brain_mask_T2.nii.gz")
#Segmentation_ref = nib.load("/hpc/meca/users/essamlali.a/manuel_segmentation/sinia_seg_ref/sub-032155_ses-001_run-1_brain_mask_T2.nii.gz")

def compute_all_metrics (seg_auto, seg_ref, pref = "img_") :
    
    data_set_auto = seg_auto.get_data()     
    data_set_ref = seg_ref.get_data()

    # Size_data
    x, y, z = data_set_ref.shape
    #print(x, y, z)

    # convertir array 3D to 1D
    auto = data_set_auto.reshape(-1)
    ref = data_set_ref.reshape(-1)
    
    # Nombre total des voxels 
    N = len(auto)





    VP, FP, VN, FN = perf_measure_seg_monomodale(auto, ref)
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

    LCE, GCE, Es, Es_prim = erreurs_martin(auto, ref, N)
    print("LCE : ", LCE)
    print("GCE : ", GCE)


    carte_erreurEs = carte_erreur_martin (Es, x, y, z)
    carte_erreurEs_prim = carte_erreur_martin (Es_prim, x, y, z)


    img_erreurEs = nib.Nifti1Image(carte_erreurEs, seg_auto.affine)
    nib.save(img_erreurEs, pref+'erreurEs.nii.gz')

    img_erreurEs_prim = nib.Nifti1Image(carte_erreurEs_prim, seg_auto.affine)
    nib.save(img_erreurEs_prim, pref+'erreurEs_prim.nii.gz')
    
    return [VP, FP, VN, FN, kappa, dice, JC, LCE, GCE]




#### Main ---------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':

    seg_auto = nib.load("D:/Stage_INT/Script_metriques/Cube1.nii.gz")
    seg_ref = nib.load("D:/Stage_INT/Script_metriques/Cube2.nii.gz")

    compute_all_metrics (seg_auto,seg_ref)
