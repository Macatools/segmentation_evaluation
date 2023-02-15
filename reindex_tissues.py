import os
import nibabel as nib
import numpy as np

def merge_tissue(old_dseg_file, new_dseg_file, tissue_tab = [0, 1, 2, 3, 3]):

    img = nib.load(old_dseg_file)
    img_data = img.get_fdata()
    print(np.unique(img_data))

    new_img_data = np.zeros(shape = img_data.shape, dtype = int)

    for index, i in enumerate(np.unique(img_data)):

        print(i, index, tissue_tab[index])

        new_img_data[img_data == i] = tissue_tab[index]

    print(np.unique(new_img_data))
    new_img = nib.Nifti1Image(new_img_data, header = img.header, affine = img.affine)
    nib.save(new_img, new_dseg_file)


def modify_tissue_order(data_path, fname):

    os.chdir(data_path)

    img = nib.load(fname)
    img_data = img.get_fdata()
    np.unique(img_data)

    img_data[img_data == 1] = 5
    img_data[img_data == 3] = 1
    img_data[img_data == 2] = 3
    img_data[img_data == 5] = 2
    new_img = nib.Nifti1Image(img_data, header = img.header, affine = img.affine)
    nib.save(new_img, "new_dseg.nii.gz")


def mask_non_zeros(data_path, fname):

    print("Creating mask for {}".format(fname))
    os.chdir(data_path)

    img = nib.load(fname)
    img_data = img.get_fdata()
    np.unique(img_data)

    img_data[img_data != 0] = 1
    new_img = nib.Nifti1Image(img_data, header = img.header, affine = img.affine)
    nib.save(new_img, "new_mask.nii.gz")

