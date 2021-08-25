# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:17:14 2021

@author: essmaile
"""




# fonction de Coefficient de corrélation intra-classe ICC.
def intraclass_correlation (path_data1,path_data2) :
    #import os
    #ICC_cmd = "MeasureImageSimilarity 3 1 {} {} ".format(path_data1, path_data2)
    #val1 = os.system(ICC_cmd)


    import subprocess
    ICC_cmd = "MeasureImageSimilarity -d 3 -m CC[{},{},1,1]".format(path_data1, path_data2)
    print (ICC_cmd)

    proc = subprocess.Popen(ICC_cmd, shell = True, stdin = None, stdout = subprocess.PIPE, stderr = subprocess.PIPE, encoding = 'utf8')
    stdout_data, stderr_data = proc.communicate(0)
    val1 = float(stdout_data)

    return val1

# fonction de calcule des 4 cardinaliés VP, FP, VN, FN, pour la segmentation Monomodale (WM ou GM ou CSF ou Brain_Mask)
def perf_measure_seg_monomodale(tab1, tab2):
    VP = 0
    FP = 0
    VN = 0
    FN = 0

    print(tab1.shape)

    print(tab2.shape)

    for i in range(len(tab2)): 
        if tab1[i]==1:
            if tab2[i]==1:
                VP += 1
            else :
                FP += 1
        else :
            if tab2[i]==1:
                FN += 1
            else :
                VN += 1

    return(VP, FP, VN, FN)



# fonction de calcule des 4 cardinaliés VP, FP, VN, FN, pour la segmentation multimodale (WM+GM+CSF) 
def perf_measure_seg_multimodale(tab1, tab2):
    VP = 0
    FP = 0
    VN = 0
    FN = 0

    for i in range(len(tab2)): 
        if tab1[i]==tab2[i]==1 or tab1[i]==tab2[i]==2 or tab1[i]==tab2[i]==3:
            VP += 1
        elif tab2[i]==1 and tab1[i]!=tab2[i]:
            FP += 1
        elif tab2[i]==2 and tab1[i]!=tab2[i]:
            FP += 1
        elif tab2[i]==3 and tab1[i]!=tab2[i]:
            FP += 1
        elif tab1[i]==tab2[i]==0:
            VN += 1
        elif tab2[i]==0 and tab1[i]!=tab2[i]:
            FN += 1

    return(VP, FP, VN, FN)



# fonction de Coffecient de Kappa 
def coff_kappa(VP, FP, VN, FN, N):
    
    fa = VP + VN
    fc = (((VN+FN)*(VN+FP)) + ((FP+VP)*(FN+VP))) / N

    kappa = (fa-fc) / (N-fc)

    return kappa


# fonction de Coffecient de DICE 
def coff_dice(VP, FP, VN, FN): 
    dice = 2*VP / (2*VP + FP + FN)
    return dice


# fonction de Coffecient de Jaccard
def coff_jaccard(VP, FP, VN, FN):
    JC = VP / (VP + FP + FN)
    return JC


# fonction de calcule des erreurs de Martin LCE et GCE 
def erreurs_martin(tab1, tab2, N):
    import numpy as np
    Es = np.zeros(N)
    Es_prim = np.zeros(N)
    min_Es_Es_prim = np.zeros(N)
    LCE = 0
    GCE = 0  
    for i in range(len(tab2)): 
        VP = 0
        FP = 0
        VN = 0
        FN = 0
        if tab1[i]==tab2[i]==1:
            VP = 1
        if tab2[i]==1 and tab1[i]!=tab2[i]:
            FP = 1
        if tab1[i]==tab2[i]==0:
            VN = 1
        if tab2[i]==0 and tab1[i]!=tab2[i]:
            FN = 1       
        if (VP+FN)==0 and (VN+FP)!=0:
            Es[i] = ((FP*(FP+(2*VN))) / (VN+FP))
        if (VN+FP)==0 and (VP+FN)!=0 :
            Es[i] = ((FN*(FN+(2*VP))) / (VP+FN))
        if (VP+FN)==0 and (VN+FP)==0:
            Es[i] = 0
        if (VP+FN)!=0 and (VN+FP)!=0:
            Es[i] = ((FN*(FN+(2*VP))) / (VP+FN)) + ((FP*(FP+(2*VN))) / (VN+FP))
        if (VP+FP)==0 and (VN+FN)!=0 :
            Es_prim[i] = ((FN*(FN+(2*VN))) / (VN+FN))
        if (VN+FN)==0 and (VP+FP)!=0 :
            Es_prim[i] = ((FP*(FP+(2*VP))) / (VP+FP))
        if (VP+FP)==0 and (VN+FN)==0 :
            Es_prim[i] = 0
        if (VP+FP)!=0 and (VN+FN)!=0 :
            Es_prim[i] = ((FP*(FP+(2*VP))) / (VP+FP)) + ((FN*(FN+(2*VN))) / (VN+FN))
            
        min_Es_Es_prim[i] = min(Es[i], Es_prim[i])
            
    LCE = 1/N * sum (min_Es_Es_prim)
    GCE = 1/N * min(sum(Es), sum(Es_prim)) 
    return(LCE, GCE, Es, Es_prim)


def carte_erreur_martin (tab, x, y, z) :
    carte_erreur = tab.reshape(x, y, z)
    return carte_erreur
    
    














