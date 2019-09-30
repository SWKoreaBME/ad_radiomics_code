from radiomics import featureextractor
import SimpleITK as sitk
import pandas as pd
import os
import nibabel as nib
import numpy as np
import argparse
import sys
import pywt
import imageio
import warnings

import warnings
warnings.filterwarnings("ignore")

from skimage.transform import resize as resize
import time

# 0. get arguments

parser = argparse.ArgumentParser(description="AD radiomics haralick + wavelet feature extraction parameters")
parser.add_argument("-o", dest = "file_name", help="output file name")
parser.add_argument("-i", dest = "image_folder", help="image directory")
parser.add_argument("-m", dest = "mask_folder", help="mask directory")
parser.add_argument("-p", dest = "params", help="parameter setting file name", default = 'featureConfig.json')

args = parser.parse_args()

# 1. read images

def read_image(file):
    if '.nii' in file:
        img_arr = nib.load(file).get_data()
    else:
        img_arr = imageio.imread(file)

    im_3d = sitk.GetImageFromArray(img_arr)
    return im_3d

def result2val(result, wv=False):
    vals, columns = [], []

    for key, val in result.items():
        if 'diagnostics' in key : continue
            
        if wv:
            if 'shape' in key : continue
        
        columns.append(key.split('original_')[1])
        vals.append(val)

    return vals, columns

def apply_wavelet(img_arr):
    coeffs2 = pywt.dwtn(img_arr, 'sym4')
    return coeffs2

# 2. feature settings

params = args.params
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
feature_dict = dict()

img_folder = args.image_folder
mask_folder = args.mask_folder

errors = []

if len(os.listdir(img_folder)) != len(os.listdir(mask_folder)):
    raise Exception (' length of images and masks should be same ')
    sys.exit()

start = time.time()
errors = []

for image, mask in zip(sorted(os.listdir(img_folder)),sorted(os.listdir(mask_folder))):

    print(image, mask)
    
    whole_vals = []
    whole_cols = []
    
    subject_name = image.split('.nii')[0]
    
#     image_nii = nib.load(os.path.join(img_folder, image))
#     mask_nii = nib.load(os.path.join(mask_folder, mask))
    
    # check image size and mask size
    
#     if image_nii.shape != mask_nii.shape : 
#         print('shape difference in ', image)
#         continue
    
    # intensity normalization
#     normalized_img_nii = itn.normalize.zscore.zscore_normalize(image_nii)
#     nib.save(normalized_img_nii, os.path.join(zscore_dir, image+'.gz'))

    # original feature extraction

    img = read_image(os.path.join(img_folder, image))
    mask = read_image(os.path.join(mask_folder, mask))
    
    print('image loaded')

    try:
    
        result = extractor.execute(img, mask)
        vals, columns = result2val(result, wv=False)

        whole_cols.extend(columns)
        whole_vals.extend(vals)
        
        img_arr = sitk.GetArrayFromImage(img)
        wv_imgs = apply_wavelet(img_arr)

        print('wavelet image loaded')
        for index, (key, wv_img) in enumerate(wv_imgs.items()):

            resized_arr = resize(wv_img, img_arr.shape, preserve_range=True, order=1) # bilinear interpolation
            wv_img = sitk.GetImageFromArray(resized_arr)
            
            result = extractor.execute(wv_img, mask) 
            wv_vals, wv_columns = result2val(result, wv=True)
                
            wv_columns = [a + '_wv_' + str(index+1) for a in wv_columns]

            whole_cols.extend(wv_columns)
            whole_vals.extend(wv_vals)

    #         break
        
        feature_dict[subject_name] = whole_vals

    except:
        errors.append(image)
        continue
    
    # break
    
end = time.time()
print(end - start, ' seconds for feature extraction')
print('feature extraction done')

# 4. Convert dict  -> Dataframe -> excel

df = pd.DataFrame.from_dict(feature_dict).T
df.columns = whole_cols
df.to_csv(args.file_name)

print('saved')
print('errors : ', errors)

if os.path.exists(args.file_name):
    sys.exit()