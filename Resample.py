import nibabel as nib
from nilearn import image
import numpy as np
# from nibabel import processing
import os

def resample(img_array, target_voxel = (1, 1, 1)):
    '''
    :param img_array:input type; nifti file format
    :param target_voxel: resample size
    '''
    target_voxel_size = np.diag(target_voxel)
    img_resampled = image.resample_img(img_array, target_affine=target_voxel_size)
    voxel_size = check_voxel_size(img_resampled)
    img_resampled_img = image.resample_img(img_array, target_affine=target_voxel_size).get_data()
    img_resampled_img = np.rot90(img_resampled_img,2)
    
    return img_resampled_img, voxel_size

def mask2binary(mask_resampled_img):
    '''
    :param mask_resampled_img: numpy array; Resampled numpy array
    :return: mask_binary_img: numpy array with binary data
    '''
    roi = np.where(mask_resampled_img > 0.7)
    roi = list(roi)
    size = mask_resampled_img.shape
    mask_binary_img = np.zeros(size)
    for x, y, z in zip(roi[0], roi[1], roi[2]):
        mask_binary_img[x][y][z] = 1
    mask_binary_img = mask_binary_img.astype(dtype='<i2')

    return mask_binary_img

def check_voxel_size(img):
    '''
    input param
    img : nifti file format
    '''
    img_header = dict(img.header)
    voxel_size = img_header['pixdim'][1:4]
    return voxel_size

def make_affine(voxel_size):
    fin = np.zeros((4,4))
    fin[3][3] = 1
    for i, a in enumerate(voxel_size):
        fin[i][i] = a
    return fin

def numpy2nii(array, affine):
    nii_file = nib.Nifti1Image(array, affine)
    return nii_file

def save_file(name, array, path, voxel):
    affine=make_affine(voxel)
    nii_file = nib.Nifti1Image(array, affine)
    nib.save(nii_file, os.path.join(path, name))