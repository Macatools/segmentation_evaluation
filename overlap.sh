#!/bin/bash

seg_man=$1
seg_auto=$2
anat=$3


# clean sform/qform (copy from T1 to mask)

CopyImageHeaderInformation $3.nii.gz $1.nii.gz $1_clean.nii.gz 1 1 1
CopyImageHeaderInformation $3.nii.gz $2.nii.gz $2_clean.nii.gz 1 1 1

LabelOverlapMeasures 3 $1_clean.nii.gz $2_clean.nii.gz 
