import warnings
warnings.filterwarnings("ignore")
import os
import re
import subprocess
import numpy as np
import pandas as pd
from time import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from skimage import exposure
from skimage import morphology
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import binary_dilation
from scipy import ndimage as ndi
import nilearn
from nilearn import image
from nilearn.input_data import NiftiMasker

mpl.rcParams['figure.figsize'] = [15, 15]
mpl.rcParams.update({'font.size': 14})
np.set_printoptions(formatter={'float_kind':'{:0.4f}'.format})

import itk
from distutils.version import StrictVersion as VS
if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)


########################## ITK Functions ##########################
def run_itk_multiscale(in_image, TOF_img_path, brain_mask, objectness_filter, VED_outfolder, itk_params, config):
    start = time()

    # Load itk_params
    sigma_minimum = itk_params['sigma_minimum'] 
    sigma_maximum = itk_params['sigma_maximum']
    number_of_sigma_steps = itk_params['number_of_sigma_steps']

    mip_axis = config['mip_axis']
    mip_view = config['mip_view']
    
    ImageType = type(in_image)
    Dimension = in_image.GetImageDimension()  #would be 3

    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]

    # Load TOF image for plotting only
    TOF_img  = image.load_img(TOF_img_path)
    TOF = TOF_img.get_fdata()
    xflipped_TOF_no_skull = np.flip(TOF*brain_mask, axis=0)

    print("+-+-+- DOING MULTI-SCALE METHOD")
    
    for step, sigma in enumerate(np.geomspace(sigma_minimum, sigma_maximum, number_of_sigma_steps)):

        scale_str = str(int(sigma*100000)).zfill(8)
        out_path = os.path.join(VED_outfolder, "scale_" + scale_str + "_VED.nii.gz")

        if not os.path.exists(out_path):
            print(" +-+- DOING sigma scale " + str(sigma) + ": " + str(step+1) + " of " + str(number_of_sigma_steps))
            multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
            multi_scale_filter.SetInput(in_image)
            multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
            multi_scale_filter.SetSigmaStepMethodToLogarithmic()
            multi_scale_filter.SetSigmaMinimum(sigma)
            multi_scale_filter.SetSigmaMaximum(sigma)
            multi_scale_filter.SetNumberOfSigmaSteps(1)
            multi_scale_filter.SetGenerateHessianOutput(True)
            multi_scale_filter.Update()

            print("  +- ..write VED of sigma " + str(sigma) + ": " + str(step+1) + " of " + str(number_of_sigma_steps))
            Hessian_output=multi_scale_filter.GetHessianOutput()
            VED_output=multi_scale_filter.GetOutput()
            size=Hessian_output.GetBufferedRegion().GetSize()

            hessian_view = itk.GetArrayViewFromImage(Hessian_output)
            hessian_array = np.transpose(hessian_view, (2,1,0,3))
            ved_view = itk.GetArrayViewFromImage(VED_output)
            ved_array = np.transpose(ved_view, (2,1,0))
            ved_array = ved_array*brain_mask

            #print(np.min(ved_array))
            #print(np.max(ved_array))
            p0, p99 = np.percentile(ved_array, (0, 99.8)) 
            ved_array = exposure.rescale_intensity(ved_array, in_range=(p0,p99))
            #print(np.min(ved_array))
            #print(np.max(ved_array))
        

            ved_img = nilearn.image.new_img_like(TOF_img, ved_array, copy_header=True)
            ved_img.to_filename(out_path)

            xflipped_ved_array = np.flip(ved_array, axis=0)

            ############### Plotting #################
            VED_title = "VED for: " + str(sigma) + ": " + str(step+1) + " of " + str(number_of_sigma_steps)
            plot_two_MIPs(xflipped_TOF_no_skull, "TOF", xflipped_ved_array, VED_title, out_path.replace(".nii.gz", "_MIP.png"), mip_axis, mip_view)
        else:
            print(f"{out_path} exists.")
        
    print("Finished run_itk_multiscale. Total time (min):", (time() - start)/60)


def run_itk_classic(in_image, TOF_img_path, brain_mask, objectness_filter, classic_outfolder, itk_params, config, out_classic_prefix="classic"):
    start = time()

    if not os.path.exists(classic_outfolder):
        os.makedirs(classic_outfolder)
    out_path = os.path.join(classic_outfolder, "classic")

    if not os.path.exists(out_classic_prefix + "_VED.nii.gz"):
        print("+-+-+- DOING CLASSIC METHOD")
        

        # Load itk_params
        sigma_minimum = itk_params['sigma_minimum'] 
        sigma_maximum = itk_params['sigma_maximum']
        number_of_sigma_steps = itk_params['number_of_sigma_steps']

        mip_axis = config['mip_axis']
        mip_view = config['mip_view']
        ImageType = type(in_image)
        Dimension = in_image.GetImageDimension()  #would be 3

        HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
        HessianImageType = itk.Image[HessianPixelType, Dimension]

        # Load TOF image
        TOF_img  = image.load_img(TOF_img_path)
        TOF = TOF_img.get_fdata()
        xflipped_TOF_no_skull = np.flip(TOF*brain_mask, axis=0)

        multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
        multi_scale_filter.SetInput(in_image)
        multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
        multi_scale_filter.SetSigmaStepMethodToLogarithmic()
        multi_scale_filter.SetSigmaMinimum(sigma_minimum)
        multi_scale_filter.SetSigmaMaximum(sigma_maximum)
        multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)
        multi_scale_filter.SetGenerateHessianOutput(True)
        multi_scale_filter.Update()
        print("+-+-+- DONE CLASSIC. TIME (min):", (time() - start)/60)

        print("+-+- Check output size")
        Hessian_output=multi_scale_filter.GetHessianOutput()
        VED_output=multi_scale_filter.GetOutput()
        size=Hessian_output.GetBufferedRegion().GetSize()
        print("%d,%d,%d"%(size[0],size[1],size[2]))


        print("+-+- Making new image")
        hessian_view = itk.GetArrayViewFromImage(Hessian_output)
        hessian_array = np.transpose(hessian_view, (2,1,0,3))
        ved_view = itk.GetArrayViewFromImage(VED_output)
        ved_array = np.transpose(ved_view, (2,1,0))
        print("hessian_array shape:", hessian_array.shape)

        hessian_img = nilearn.image.new_img_like(TOF_img, hessian_array, copy_header=True)
        hessian_img.to_filename(out_classic_prefix + "_best_hessian.nii.gz")
        
        ved_array = np.transpose(ved_view, (2,1,0))
        ved_array = ved_array*brain_mask

        print("Min max before rescaling:", np.min(ved_array), np.max(ved_array))
        p0, p99 = np.percentile(ved_array, (0, 99.8)) 
        ved_array = exposure.rescale_intensity(ved_array, in_range=(p0,p99))
        print("Min max after rescaling:", np.min(ved_array), np.max(ved_array))

        

        ved_img = nilearn.image.new_img_like(TOF_img, ved_array, copy_header=True)
        ved_img.to_filename(out_classic_prefix + "_VED.nii.gz")

        xflipped_ved_array = np.flip(ved_array, axis=0)
        
        # Plotting
        plot_two_MIPs(xflipped_TOF_no_skull, "TOF", xflipped_ved_array, "Filtered Output", out_classic_prefix + '_MIP.png', mip_axis, mip_view)

    else:
        print(out_classic_prefix + "_VED.nii.gz from CLASSIC method exists.")
    
    print("Finished run_itk_classic. Total time (min):", (time() - start)/60)

########################## Plotting ##########################

def add_MIP_config(config):
    mip_dir = config['mip_dir']
    mip_idx_range = config['mip_idx_range']
    if mip_dir == 'sag':
        config['mip_axis'] = 0
        config['mip_view'] = tuple([ slice(mip_idx_range[0],mip_idx_range[1],1), slice(None,None), slice(None,None)  ])
    elif mip_dir == 'cor':
        config['mip_axis'] = 1
        config['mip_view'] = tuple([ slice(None,None), slice(mip_idx_range[0],mip_idx_range[1],1), slice(None,None)  ])
    else :
        config['mip_axis'] = 2
        config['mip_view'] = tuple([ slice(None, None), slice(None, None), slice(mip_idx_range[0],mip_idx_range[1],1) ])

    return config


def FA_combine_scales(img_4D, sigma_min, sigma_max):
    masker = NiftiMasker()
    data_masked = masker.fit_transform(img_4D)
    # print(np.max(data_masked))
    assert np.max(data_masked) > 0, "Mask and data are not in same space/orient"

    
    p0, p99 = np.percentile(data_masked, (0, 99.8)) 
    data_masked = exposure.rescale_intensity(data_masked, in_range=(p0,p99))

    #print(data_masked.shape)

    # print(np.max(data_masked))

    noise_variance_init=np.geomspace(sigma_min, sigma_max, data_masked.shape[0]) - sigma_min #[0:16]
    # print("Noise variance: ", noise_variance_init)

    method = FactorAnalysis(n_components=1, 
                            noise_variance_init=noise_variance_init,
                            #noise_variance_init=np.geomspace(sigma_minimum, sigma_maximum, number_of_sigma_steps),
                            tol=0.005, max_iter=200)
    method.fit(data_masked.T)
    X_reduced = method.transform(data_masked.T)
    X_reduced[X_reduced<0] = 0
    #print(X_reduced.shape)

    #print(X_reduced.noise_variance_array_)
    fact_img = masker.inverse_transform(X_reduced.T)
    fact_img = image.index_img(fact_img,0)
    
    factimg=nilearn.image.math_img("np.where(x<0, 0, x)", x=fact_img)
    
    p0, p99 = np.percentile(factimg.get_fdata(), (0, 99.8)) 
    factdata = exposure.rescale_intensity(factimg.get_fdata(), in_range=(p0,p99))

    return factdata

def otsu_3D(i_img, offset=0.2, input_is_array=False):
    if input_is_array:
        i_data = i_img
    else:
        i_data = i_img.get_fdata()
        
    p0, p99 = np.percentile(i_data, (0, 99.8)) 
    datacopy = exposure.rescale_intensity(i_data, in_range=(p0,p99))

    #datacopy = fact_img.get_fdata()
    threshold_x = np.zeros_like(datacopy) # Return an array of zeros with the same shape and type as a given array
    threshold_y = np.zeros_like(datacopy)
    threshold_z = np.zeros_like(datacopy)

    # offset = 0.2

    #threshold_local
    for x in range(datacopy.shape[0]):
        if np.max(datacopy[x, :, :]) == 0.0:
            val = 1.0
        else:
            img_adapteq = exposure.equalize_adapthist(datacopy[x,:,:], clip_limit=0.01)
            val = threshold_otsu(img_adapteq, nbins=400 )
            val = max(0.0, val-offset)
            threshold_x[x, :, :] = val

    for y in range(datacopy.shape[1]):
        if np.max(datacopy[:, y, :]) == 0.0:
            val = 1.0
        else:
            img_adapteq = exposure.equalize_adapthist(datacopy[:,y,:], clip_limit=0.01)
            val = threshold_otsu(img_adapteq, nbins=400 )
            val = max(0.0, val-offset)
            threshold_y[:, y, :] = val

    for z in range(datacopy.shape[2]):
        if np.max(datacopy[:, :, z]) == 0.0:
            val = 1.0
        else:
            img_adapteq = exposure.equalize_adapthist(datacopy[:,:,z], clip_limit=0.01)
            val = threshold_otsu(img_adapteq, nbins=400 )
            val = max(0.0, val-offset)
            threshold_z[:, :, z] = val

    threshold = np.stack((threshold_x, threshold_y, threshold_z), axis=3)
    threshold_min = np.min(threshold, axis=3)

    return datacopy > threshold_min

def afni_3dresample(input_path, output_filename, isotropic=True, target_voxel_sizes=None, binary=False):
    print("Resampling", input_path)
    if isotropic:
        img = image.load_img(input_path)
        voxel_sizes = img.header.get_zooms()
        target_voxel_sizes = [min(voxel_sizes), min(voxel_sizes), min(voxel_sizes)]
    else:
        assert target_voxel_sizes!=None
        #TO DO: Finish this part
    
    if binary:
        rmode = "NN"
    else:
        rmode = "Cu"

    afni_command = "3dresample -overwrite -dxyz {} {} {} -rmode {} -prefix {} -inset {}".format(
        target_voxel_sizes[0], 
        target_voxel_sizes[1], 
        target_voxel_sizes[2], 
        rmode,
        output_filename, 
        input_path
        )
    subprocess.run(afni_command, shell=True)


def plot_two_MIPs(img1, img1_name, img2, img2_name, savepath, mip_axis, mip_view, img2_color='viridis', show_img=False, figsize=None):
    if figsize != None:
        fig, ax = plt.subplots(2,1, figsize=figsize)
    else:
        fig, ax = plt.subplots(2,1, figsize=(15,8))
    ax[0].imshow(np.max(img1[mip_view], axis=mip_axis).T, origin='lower', cmap=plt.cm.gray, alpha=1.0)
    ax[0].set_title(img1_name)
    ax[0].set_axis_off()

    ax[1].imshow(np.max(img2[mip_view], axis=mip_axis).T, origin='lower', cmap=img2_color, alpha=1.0)
    ax[1].set_title(img2_name)
    ax[1].set_axis_off()

    fig.tight_layout()
    if show_img:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()

def plot_MIP(imgs, img_names, mip_axis, mip_view, cmaps, save_path, show_img=False, figsize=None, flip_x=True, hide_axis=True):
    num_img = len(imgs)
    if figsize != None:
        fig, ax = plt.subplots(num_img, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(num_img, 1, figsize=(15,4*num_img))
    if num_img > 1:
        for i in range(num_img):
            mip = np.max(imgs[i][mip_view], axis=mip_axis)
            if flip_x:
                mip = np.flip(mip, axis=0)
            ax[i].imshow(mip.T,
                         origin='lower',
                         cmap=cmaps[i],
                         alpha=1.0)
            ax[i].set_title(img_names[i])
            if hide_axis:
                ax[i].set_axis_off()
    else:
        mip = np.max(imgs[0][mip_view], axis=mip_axis)
        if flip_x:
            mip = np.flip(mip, axis=0)
        ax.imshow(mip.T,
                  origin='lower',
                  cmap=cmaps[0],
                  alpha=1.0)
        ax.set_title(img_names[0])
        if hide_axis:
            ax.set_axis_off()
    
    fig.tight_layout()
    if show_img:
        plt.show()
    elif save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.close()


if __name__ == '__main__':
    print("Running utils.py")
