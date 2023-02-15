import os
import json

def padding_cropped_img(cropped_img_file, orig_img_file, indiv_crop, export_path = "", padded_fname = ""):

    import os

    import nibabel as nib
    import numpy as np

    from nipype.utils.filemanip import split_filename as split_f

    # cropping params
    assert 'crop_T1' in indiv_crop.keys(), "Error, could not find crop_T1 in {}".format(indiv_crop.keys())
    assert 'args' in indiv_crop['crop_T1'].keys(), "Error, could not find args in {}".format(indiv_crop['crop_T1'].keys())

    crop = indiv_crop['crop_T1']['args'].split()
    print("cropping params {}".format(crop))

    xmin = int(crop[0])
    xmax = xmin + int(crop[1])

    ymin = int(crop[2])
    ymax = ymin + int(crop[3])

    zmin = int(crop[4])
    zmax = zmin + int(crop[5])

    # orig image
    orig_img = nib.load(orig_img_file)

    data_orig = orig_img.get_data()
    header_orig = orig_img.header
    affine_orig = orig_img.affine

    print("Orig img shape:", data_orig.shape)

    # cropped image
    cropped_img = nib.load(cropped_img_file)
    data_cropped = cropped_img.get_data()

    print("Cropped img shape:", data_cropped.shape)

    if len(data_orig.shape) == 3 and len(data_cropped.shape) == 4:
        print("Padding with 3D params on a 4D image")

        padded_img_data_shape = (*data_orig.shape, data_cropped.shape[3])
        print(padded_img_data_shape)

        padded_img_data = np.zeros(shape=padded_img_data_shape,
                                   dtype=data_cropped.dtype)
        print("Broscasted padded img shape:", padded_img_data_shape)

        for t in range(data_cropped.shape[3]):
            padded_img_data[xmin:xmax, ymin:ymax, zmin:zmax, t] = \
                data_cropped[:, :, :, t]

    else:

        padded_img_data = np.zeros(shape=data_orig.shape,
                                   dtype=data_cropped.dtype)
        print("Padded img shape:", padded_img_data.shape)

        padded_img_data[xmin:xmax, ymin:ymax, zmin:zmax] = data_cropped

    # saving padded image
    fpath, fname, ext = split_f(cropped_img_file)

    if padded_fname == "":
        padded_fname = fname + "_padded" + ext

    if export_path == "":
        padded_img_file = os.path.abspath(padded_fname)

    else:
        padded_img_file = os.path.join(export_path, padded_fname)

    img_padded_res = nib.Nifti1Image(padded_img_data, affine=affine_orig,
                                     header=header_orig)
    nib.save(img_padded_res, padded_img_file)

    return padded_img_file