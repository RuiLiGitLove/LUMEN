import os
import sys  
import json 
from glob import glob
import warnings
warnings.filterwarnings("ignore")
import shutil
import numpy as np
import matplotlib as mpl
import nilearn
from nilearn import image
import scipy.ndimage as ndi
from skimage import morphology
from skimage.filters import threshold_local
import itk
from distutils.version import StrictVersion as VS
if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)

from MSFDF_functions import run_itk_multiscale, run_itk_classic, FA_combine_scales, afni_3dresample, add_MIP_config, plot_MIP, otsu_3D


mpl.rcParams['figure.figsize'] = [15, 15]
mpl.rcParams.update({'font.size': 20})
np.set_printoptions(formatter={'float_kind':'{:0.4f}'.format})

def run_MSFDF(ID, out_folder, orig_TOF_path, hyperparams, config, brain_mask_path=None):
    # This is adapted from https://github.com/braincharter/vasculature_notebook
    #############################################################################
    ######################## Step 1: Data Preperation ###########################
    #############################################################################
    print("Running MSFDF on ID", ID)
    print("---------------------------------")
    print(f"Step 1: Data preperation ...")
    
    ### Check if TOF is isotropic ###
    TOF_img  = image.load_img(orig_TOF_path)  
    image_size = TOF_img.shape
    voxel_sizes = TOF_img.header.get_zooms()
    print("Original voxel size is: ", voxel_sizes)
    print("Original image size is: ", image_size)
    is_isotropic = all(size == voxel_sizes[0] for size in voxel_sizes)
    if is_isotropic==False:
        print("Provided TOF is not isotropic. Please make sure that it is isotropic.")
        exit()
    
    # Check brain mask if given
    if brain_mask_path:
        brain_mask_img = image.load_img(brain_mask_path)
        brain_mask_voxel_sizes = brain_mask_img.header.get_zooms()
        is_isotropic = all(size == brain_mask_voxel_sizes[0] for size in brain_mask_voxel_sizes)
        if is_isotropic==False:
            print("Provided TOF brain mask is not isotropic. Please make sure that it is isotropic.")
            exit()
        brain_mask = brain_mask_img.get_fdata()
        assert TOF_img.shape == brain_mask_img.shape
    else:
        brain_mask = np.ones(TOF_img.shape)
    
    ### Passed checking ###
    TOF = TOF_img.get_fdata()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 

    # Experiment - Rescale intensities inside TOF to fixed range 0-1
    min_value = np.min(TOF)
    max_value = np.max(TOF)
    if abs(max_value-min_value) < 1e-6:
        TOF[:] = 0.001
    else:   
        TOF = (TOF - min_value) / (max_value - min_value)
    
    # Save normalised skull-stripped TOF for vessel filtering step
    TOF_no_skull = TOF*brain_mask
    TOF_path = os.path.join(out_folder, "masked_upsampled_TOF.nii.gz")
    TOF_no_skull_img = image.new_img_like(TOF_img, TOF_no_skull, copy_header=True)
    TOF_no_skull_img.to_filename(TOF_path)
    
    ######## Specify VED filter parameters ###########
    print("Setting VED filter parameters...")
    brightvessels = True   # True for TOF

    if (hyperparams["sigma_minimum_scalar"]==None) or (hyperparams["sigma_maximum_scalar"]==None):
        sigma_minimum_scalar = 1
        sigma_maximum_scalar = 8
        print("User did not provide range of scales. Using default setting: sigma_min={}*voxelsize, sigma_max={}*voxelsize".format(sigma_minimum_scalar, sigma_maximum_scalar))
    else:
        sigma_minimum_scalar = hyperparams["sigma_minimum_scalar"]
        sigma_maximum_scalar = hyperparams["sigma_maximum_scalar"]

    sigma_minimum = np.min(voxel_sizes)*sigma_minimum_scalar    # min voxel size (or mean), can be a little higher
    sigma_maximum = sigma_minimum*sigma_maximum_scalar

    print("sigma_minimum is: {:.2f}".format(sigma_minimum))
    print("sigma_maximum is: {:.2f}".format(sigma_maximum))

    number_of_sigma_steps = config["number_of_sigma_steps"] 

    alpha=0.8                     #In theory, no need to adjust this
    beta=1.0                      #In theory, no need to adjust this
    gamma = hyperparams["gamma"]  #50 #Play with this to adjust for the noise/blur in the vascular segmentation
    scaleoutput = True            #Do. Not. Touch. That.
   
    ########## Specify MIP parameters ##################
    if config["plot_mip"]:
        config = add_MIP_config(config)
        mip_axis = config['mip_axis']
        mip_view = config['mip_view']
        plot_MIP([TOF_no_skull],
                 ['TOF'],
                 mip_axis,
                 mip_view,
                 ['gray'],
                 os.path.join(out_folder, "TOF_MIP.png"))


    #############################################
    ######### Step 2: Vessel filtering ##########
    #############################################
    print("---------------------------------")
    print(f"Step 2: Vessel filtering with {config['filtering_method']} method...")

    ######################## Option 2.1: ITK #############################
    if config["filtering_method"] == "ITK_Multiscale":

        VED_outfolder = os.path.join(out_folder, "ITK_Multiscale_filtered_images")
        if not os.path.exists(VED_outfolder):
            os.makedirs(VED_outfolder)
            in_image   = itk.imread(TOF_path, itk.F)      
            ImageType = type(in_image)
            Dimension = in_image.GetImageDimension()  #would be 3

            HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
            HessianImageType = itk.Image[HessianPixelType, Dimension]

            objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
            objectness_filter.SetBrightObject(brightvessels)
            objectness_filter.SetScaleObjectnessMeasure(scaleoutput)  
            objectness_filter.SetAlpha(alpha) 
            objectness_filter.SetBeta(beta)   
            objectness_filter.SetGamma(gamma) 

            itk_params = {
                'sigma_minimum': sigma_minimum,
                'sigma_maximum': sigma_maximum,
                'number_of_sigma_steps': number_of_sigma_steps,
            }

            ######################### 2.1.1 MULTI-SCALE METHOD ################################
            run_itk_multiscale(in_image, TOF_path, brain_mask, objectness_filter, VED_outfolder, itk_params, config)
        else:
            print("ITK_Multiscale already exists.")

    ######################### 2.1.2 CLASSIC METHOD ################################
    elif config["filtering_method"] == "ITK_Classic":
        classic_outfolder =os.path.join(out_folder, "CLASSIC")
        run_itk_classic(in_image, TOF_path, brain_mask, objectness_filter, classic_outfolder , itk_params, config)

    else:
        print("Unrecognised filtering method.")
        return None


    #######################################################################
    ####### Step 3: Combining VED images from ITK_Multiscale METHOD #######
    #######################################################################
    out_combined_path = os.path.join(out_folder, "combined_filtered.nii.gz")
    if (config["filtering_method"] == "ITK_Multiscale") and (os.path.exists(out_combined_path)==False):
        ### Combine VED series ###
        print("---------------------------------")
        print("Step 3: Combining VED images from ITK_Multiscale METHOD...")
        VED_images=nilearn.image.load_img(os.path.join(VED_outfolder, "*_VED.nii.gz"), wildcards=True)
        if config["combination_method"] == "Maxpooling":
            print("+-+- Compute max image")
            VED_image_files = glob(os.path.join(VED_outfolder, "*_VED.nii.gz"))
            VED_max_img = image.load_img(VED_image_files[0])
            for f in VED_image_files[1:]:
                img = image.load_img(f)
                VED_max_img = nilearn.image.math_img("np.maximum(a, b)", a=VED_max_img, b=img)
            VED_max_img.to_filename(out_combined_path)

            if config['plot_mip']:
                plot_MIP([TOF_no_skull, VED_max_img.get_fdata()],
                         ["TOF", "VED max"],
                         mip_axis,
                         mip_view,
                         ['gray', 'viridis'],
                         os.path.join(out_folder, "combined_filtered_MIP.png"))

        elif config["combination_method"] == "FA":
            VED_images=nilearn.image.load_img(os.path.join(VED_outfolder, "*_VED.nii.gz"), wildcards=True)
            print("+-+- Compute factory analysis")
            fact_data = FA_combine_scales(VED_images, sigma_minimum, sigma_maximum)
            fact_img = image.new_img_like(TOF_img, fact_data)
            fact_img.to_filename(out_combined_path) 
        
            if config['plot_mip']:
                plot_MIP([TOF_no_skull, fact_img.get_fdata()],
                         ["TOF", "VED FA"],
                         mip_axis,
                         mip_view,
                         ['gray', 'viridis'],
                         os.path.join(out_folder, "combined_filtered_MIP.png"))
        else:
            print("Unrecognised VED combination method:", config["combination_method"])
            return None
    else:
        print("No need to combine VEDs. Skipping Step 3.")
    
    ###############################################
    ########### Step 4: Thresholding   ############
    ###############################################
    out_seg_path = os.path.join(out_folder, "thresholded.nii.gz")
    if not os.path.exists(out_seg_path):
        print("---------------------------------")
        print("Step 4: Thresholding...")
        filtered_img = image.load_img(out_combined_path)
        filtered = filtered_img.get_fdata()
        if hyperparams['thres_method'] == 'local_gaussian':
            local_thres = threshold_local(
                filtered, 
                block_size=hyperparams['local_thres_block_size'], 
                method='gaussian', 
                offset=hyperparams['local_thres_offset'])
            seg = (filtered>local_thres).astype(np.int8)
        
        elif hyperparams['thres_method'] == 'global':
            seg = (filtered>hyperparams['global_thres']).astype(np.int8)

        elif hyperparams['thres_method'] == 'otsu_3D':
            seg = otsu_3D(filtered, input_is_array=True).astype(np.int8)

        else:
            print("Unrecognised thresholding method:", hyperparams['thres_method'])
            return None
        
        # Save segmentation
        seg_img = nilearn.image.new_img_like(filtered_img, seg, copy_header=True)
        seg_img.to_filename(out_seg_path)

        if config['plot_mip']:
            plot_MIP([TOF_no_skull, seg], 
                     ["TOF", "Segmentation"], 
                     mip_axis,
                     mip_view, 
                     ["gray", "gray"], 
                     os.path.join(out_folder,"thresholded.png"))
    else:
        print("Thresholded mask already exists. Skipping Step 4.")


    #################################################
    ########### Step 5: Post-processing  ############
    #################################################
    out_postprocessed_path = os.path.join(out_folder, f"{ID}_final_segmentation.nii.gz")
    if not os.path.exists(out_postprocessed_path):
        print("---------------------------------")
        print("Step 5: Post-processing...")
        seg_img = image.load_img(out_seg_path)
        seg = seg_img.get_fdata()

        seg_int = (seg>0.5).astype(int)

        connectivity = hyperparams['closing_conn']
        structure = ndi.generate_binary_structure(seg_int.ndim, connectivity)
        postprocessed_int = seg_int.copy()
        postprocessed_int = ndi.binary_dilation(postprocessed_int, structure=structure, iterations=1)
        postprocessed_int = morphology.remove_small_objects(postprocessed_int, min_size=hyperparams['pp_minsize'], connectivity=1)
        postprocessed_int = ndi.binary_erosion(postprocessed_int, structure=structure, iterations=1)
        postprocessed_int = morphology.remove_small_holes(postprocessed_int, area_threshold=3000, connectivity=1)
        
        # Save postprocessed image
        postprocessed_img = nilearn.image.new_img_like(seg_img, postprocessed_int, copy_header=True)
        postprocessed_img.to_filename(out_postprocessed_path)

        if config['plot_mip']:
            plot_MIP([seg, postprocessed_int],
                     ['Original Mask', 'Postprocessed'],
                     mip_axis,
                     mip_view,
                     ['gray', 'gray'],
                     out_postprocessed_path.replace(".nii.gz", ".png"))
    else:
        print("Final postprocessed segmentation already exists. Skipping Step 5.")
    
    ##########################################
    ########### Step 6: Cleaning  ############
    ##########################################
    if config["clear_inter_output"]:
        print("Cleaning intermediate files...")
        paths_to_remove = [TOF_path,out_combined_path, out_seg_path]
        if config['plot_mip']:
            paths_to_remove += glob(os.path.join(out_folder, "*.png"))
        for p in paths_to_remove:
            if os.path.exists(p):
                os.remove(p)

        folders_to_remove = [VED_outfolder] if config["filtering_method"] == "ITK_Multiscale" else []
        for f in folders_to_remove:
            if os.path.exists(f):
                shutil.rmtree(f)

    print("Finished MSFDF pipeline")
    print("------------------------")


   
if __name__ == '__main__':
    print("MSFDF_main.py is not meant to be run directly.")