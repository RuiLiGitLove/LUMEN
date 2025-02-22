import os
import sys
import re
import nrrd
import json
import math
import numpy as np
import pandas as pd 
import nibabel as nib
import scipy.ndimage as ndi
from skimage import morphology
from skimage.morphology import skeletonize
from skimage import measure
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from nilearn.image import resample_img
from utils import crop_nifti, get_endpoints_in_3d

def get_ROI_location_from_mask(LSA_PA_mask_path:str, original_LSA_PA_mask_path:str) -> dict:
    '''
    This function gets the min and max idx for the bounding boxes from LSA PA mask
    '''
    # Load mask
    LSA_PA_mask_img = nib.load(LSA_PA_mask_path)
    LSA_PA_mask = LSA_PA_mask_img.get_fdata()
    LSA_PA_mask = (LSA_PA_mask>0.5).astype(np.int8)
    
    labelled_LSA_PA_mask, num_labels = measure.label(LSA_PA_mask, return_num=True, connectivity=1)
    regions = measure.regionprops(labelled_LSA_PA_mask)
    
    if num_labels>2: # Some island got split during registration and smoothing -> Keep the largest two
        print(f"Detected more than two islands. Keeping the largest two, and rewriting mask to {LSA_PA_mask_path}")

        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        LSA_PA_mask *= np.isin(labelled_LSA_PA_mask, [regions[0].label, regions[1].label])
    
        # Save modified LSA_PA_mask
        updated_LSA_PA_mask = (LSA_PA_mask>0.5).astype(np.int8)
        updated_LSA_PA_mask_nifti = nib.Nifti1Image(updated_LSA_PA_mask, LSA_PA_mask_img.affine, LSA_PA_mask_img.header)
        nib.save(updated_LSA_PA_mask_nifti, LSA_PA_mask_path)
    # else:
    #     updated_LSA_PA_mask = (LSA_PA_mask>0.5).astype(np.int8)


    island_1_min_idx = list(regions[0].bbox[:3])
    island_1_max_idx = [x - 1 for x in regions[0].bbox[3:]]
    island_2_min_idx = list(regions[1].bbox[:3])
    island_2_max_idx = [x - 1 for x in regions[1].bbox[3:]]

    if island_1_min_idx[0]<island_2_min_idx[0]: # compare minimum x location to decide which is left
        #island_1 is left
        ROI_location = {
            "Left":{
                "min_idx": island_1_min_idx,
                "max_idx": island_1_max_idx
            },
            "Right":{
                "min_idx": island_2_min_idx,
                "max_idx": island_2_max_idx
            }
        }
    else:
        ROI_location = {
            "Right":{
                "min_idx": island_1_min_idx,
                "max_idx": island_1_max_idx
            },
            "Left":{
                "min_idx": island_2_min_idx,
                "max_idx": island_2_max_idx
            }
        }
    return ROI_location

def get_distance_between_islands(labelled_arr, label_1, label_2):
    # This function computes the shortest distance between the two islands with the given labels in a labelled array
    dist_mat = cdist(np.argwhere(labelled_arr==label_1),np.argwhere(labelled_arr==label_2), "euclidean")
    return dist_mat.min()

def save_fudicial_points_for_slicer(endpoint_lst, affine_mat, save_path, label_pts=True, label_prefix="F-", locked=False, unselect_lowest_pt=False):
    if unselect_lowest_pt:
        endpoint_z_lst = [pt[2] for pt in endpoint_lst]
        lowest_pt_idx = endpoint_z_lst.index(min(endpoint_z_lst))
    else:
        lowest_pt_idx = -1
    
    output_endpoints = []
    for idx, pt_index_loc in enumerate(endpoint_lst):
        pt_RAS_loc = np.dot(affine_mat, pt_index_loc+[1])

        if label_pts:
            this_label = f"{label_prefix}{idx+1}"
        else:
            this_label = ""
        pt_info = {
            "label": this_label,
            "position": [pt_RAS_loc[0], pt_RAS_loc[1], pt_RAS_loc[2]],
            "locked": locked,
            "selected": idx!=lowest_pt_idx
        }
        output_endpoints.append(pt_info)

    endpoints_json = {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [{
            "type": "Fiducial", 
            "coordinateSystem": "RAS", 
            "controlPoints": output_endpoints
        }]
    }

    json.dump(endpoints_json, open(save_path, 'w'), indent=4)

def dilate_binary_seg(seg, connectivity=1):
    dilated = (seg>0.5).astype(np.int8)
    structure = ndi.generate_binary_structure(dilated.ndim, connectivity=connectivity)
    dilated = ndi.binary_dilation(dilated, structure=structure, iterations=1) # ndarray of bools
    dilated = dilated.astype(np.int8)
    return dilated 

def erode_binary_seg(seg, connectivity=1):
    eroded = (seg>0.5).astype(np.int8)
    structure = ndi.generate_binary_structure(eroded.ndim, connectivity=connectivity)
    eroded = ndi.binary_erosion(eroded, structure=structure, iterations=1) # ndarray of bools
    eroded = eroded.astype(np.int8)
    return eroded

def remove_islands_far_from_largest(seg, thres_dist):
    ## Dilation
    postprocessed = dilate_binary_seg(seg, 1)

    ## Remove islands far from the largest
    labelled_pp_seg, num_labels = measure.label(postprocessed, return_num=True, connectivity=1)
    island_sizes = [np.sum(labelled_pp_seg == label_id) for label_id in range(1, num_labels+1)]
    max_island_label = np.argmax(island_sizes)+1 # Assume this largest island is the main LSA 
    # Calculate the shortest distance from each island to this largest island
    distances_to_largest_island_map = {}
    distances = []
    for label in range(1, num_labels+1):
        if label != max_island_label:
            dist = get_distance_between_islands(labelled_pp_seg, max_island_label, label)
            distances_to_largest_island_map[label] = dist
            distances.append(dist)
    islands_to_remove = [a for a in list(distances_to_largest_island_map.keys()) if distances_to_largest_island_map[a]>=thres_dist]
    remaining_islands = np.setdiff1d(np.arange(1, num_labels+1), islands_to_remove)
    clean_labelled_pp_seg = labelled_pp_seg.copy()
    # print(f"{len(islands_to_remove)} islands removed")
    for label in islands_to_remove:
        clean_labelled_pp_seg[clean_labelled_pp_seg==label] = 0
    postprocessed = (clean_labelled_pp_seg>0.5).astype(np.int8)

    ## Erosion
    postprocessed = erode_binary_seg(postprocessed)
    return postprocessed

def remove_islands_touching_top_surface(mask):
    # Identify connected components in the 3D binary mask
    labelled_mask, num_labels = measure.label(mask, return_num=True, connectivity=1)
    regions = measure.regionprops(labelled_mask)
    x_max, y_max, z_max = tuple(i-1 for i in mask.shape)
    cleaned_mask = mask.copy()
    count = 0
    
    for region in regions:
        region_x_min, region_y_min, region_z_min = region.bbox[:3]
        region_x_max, region_y_max, region_z_max = tuple(i-1 for i in region.bbox[3:])
        touches_top = region_z_max == z_max

        # If the component touches only the top surface and no other surface, remove it
        if touches_top and region_z_min>(z_max/2):
            cleaned_mask[labelled_mask==region.label] = 0  # Remove the component from the mask
            count+=1

    print(f"Removed {count} islands touching the top surface")
    return cleaned_mask

def get_labelled_seg(cropped_seg_path, save_dir, closing=False, auto_remove_distant_islands=False, thres_dist=15, dilate=False, get_helper_endpoints=True, save_name="labelled_seg.nii.gz", remove_top_surface_islands=False):
    '''
        auto_remove_distant_islands: whether to remove islands that are too far from the largest one
        thres_dist: the distance threshold to remove those clusters that are too far away from the largest one 
        remove_top_surface_islands: whether to remove islands that are touching the top surface of the segmentation -- typically not LSAs
    '''

    cropped_seg_img = nib.load(cropped_seg_path)
    cropped_seg = cropped_seg_img.get_fdata()
    out_seg = cropped_seg.copy()

    ## Morphological closing -- disabled as this was found to connect adjacent branches too much
    if closing:
        out_seg = dilate_binary_seg(out_seg)
        out_seg = erode_binary_seg(out_seg)

    ## Remove small islands
    out_seg = morphology.remove_small_objects(out_seg>0.5, min_size=10, connectivity=1)

    ## Remove islands far from the largest
    if auto_remove_distant_islands:
        out_seg = remove_islands_far_from_largest(out_seg, thres_dist)

    ## Dilation
    if dilate:
        out_seg = dilate_binary_seg(out_seg, 1)
    
    ## Remove top islands
    if remove_top_surface_islands:
        out_seg = remove_islands_touching_top_surface(out_seg)

    ## Labelling
    labelled_out_seg, num_labels = measure.label(out_seg, return_num=True, connectivity=1)
    print(f"{num_labels} islands in the segmentation")
    labelled_out_seg_nifti = nib.Nifti1Image(labelled_out_seg, cropped_seg_img.affine, cropped_seg_img.header)
    nib.save(labelled_out_seg_nifti, os.path.join(save_dir, save_name))


    ## Get endpoints to help with manual correction
    if get_helper_endpoints:
        skeleton = (skeletonize(out_seg>0.5)>0.5).astype(np.int8)
        endpoint_mask, endpoint_lst = get_endpoints_in_3d(skeleton)

        # Ensure that each island has at least one endpoint
        for label in range(1, num_labels+1):
            num_endpoints = np.sum(endpoint_mask[labelled_out_seg==label])
            if num_endpoints == 0:
                positions = np.argwhere(labelled_out_seg==label)
                idx = np.random.choice(len(positions))
                endpoint_mask[positions[idx][0], positions[idx][1], positions[idx][2]] = 1
                endpoint_lst.append(list(positions[idx,:]))

        endpoint_save_path = os.path.join(save_dir, "helper_endpoints.json")
        save_fudicial_points_for_slicer(endpoint_lst, cropped_seg_img.affine, endpoint_save_path, label_pts=False, locked=True)



def upsample_seg(seg_img, factor=3, interpolation_method='nearest'):
    # Get new affine and shape
    new_affine = seg_img.affine.copy()
    new_affine[:3, :3] /= factor
    new_shape = tuple(int(x * factor) for x in seg_img.shape)

    # Resample
    upsampled_seg_img = resample_img(seg_img, target_affine=new_affine, target_shape=new_shape, interpolation=interpolation_method)
    if interpolation_method=='continuous': #assuming binary segmentation
        upsampled_seg = (upsampled_seg_img.get_fdata() > 0.5).astype(np.int8)
        assert np.max(upsampled_seg) == 1
        upsampled_seg_img = nib.Nifti1Image(upsampled_seg, upsampled_seg_img.affine, upsampled_seg_img.header)

    # Sanity check -- check the number of labels are identical after upsampling
    seg = seg_img.get_fdata().astype(np.int8)
    upsampled_seg = upsampled_seg_img.get_fdata().astype(np.int8)
    assert set(seg.ravel()) == set(upsampled_seg.ravel())
    return upsampled_seg_img


def get_LSA_with_MCA(corrected_nrrd_path, ref_cropped_seg_path, save_dir):
    ## 1. Load and binarise manually corrected segmentation
    corrected_seg, corrected_seg_header = nrrd.read(corrected_nrrd_path)
    ref_cropped_seg_img = nib.load(ref_cropped_seg_path)
    if corrected_seg.shape == ref_cropped_seg_img.shape:
        bin_corrected_seg = corrected_seg > 0.5
    else:
        bin_corrected_seg = np.max(corrected_seg, axis=0) > 0.5

    ## 2. Keep the largest island
    labelled_bin_corrected_seg, num_labels = measure.label(bin_corrected_seg, return_num=True, connectivity=1)
    island_sizes = [np.sum(labelled_bin_corrected_seg == label_id) for label_id in range(1, num_labels+1)]
    max_island_label = np.argmax(island_sizes)+1 # Assume this largest island is the main LSA we want to extract
    LSA_MCA_seg = (labelled_bin_corrected_seg==max_island_label) # checked: max=1

    # Refine the segmentation for the largest island
    LSA_MCA_seg = morphology.remove_small_holes(LSA_MCA_seg, area_threshold=3000, connectivity=1)
    LSA_MCA_seg = LSA_MCA_seg.astype(np.int8) # checked: max=1
    assert LSA_MCA_seg.shape == ref_cropped_seg_img.shape
    assert np.max(LSA_MCA_seg) == 1

    LSA_MCA_seg_savepath = os.path.join(save_dir, "LSA_MCA_seg.nii.gz")
    LSA_MCA_seg_img = nib.Nifti1Image(LSA_MCA_seg, ref_cropped_seg_img.affine, ref_cropped_seg_img.header)
    nib.save(LSA_MCA_seg_img, LSA_MCA_seg_savepath)
    return LSA_MCA_seg_savepath


def postprocess_LSA_seg(LSA_nrrd_path, ref_img_path, save_dir, out_seg_name, upsample=True, ROI_mask_path=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## 1. Load and binarise manually corrected segmentation
    LSA_seg, LSA_seg_header = nrrd.read(LSA_nrrd_path)
    ref_img = nib.load(ref_img_path)
    if LSA_seg.shape == ref_img.shape:
        LSA_seg = LSA_seg > 0.5
    else:
        LSA_seg = np.max(LSA_seg, axis=0) > 0.5
    assert LSA_seg.shape == ref_img.shape

    if ROI_mask_path != None:
        if ROI_mask_path.endswith("nii.gz"):
            ROI_mask_img = nib.load(ROI_mask_path)
            ROI_mask = ROI_mask_img.get_fdata() > 0.5
        else:
            ROI_mask, ROI_mask_header = nrrd.read(ROI_mask_path)
            ROI_mask = ROI_mask>0.5
            
        assert LSA_seg.shape == ROI_mask.shape
        LSA_seg = LSA_seg & ROI_mask

    ## 2. Refine the segmentation
    LSA_seg = morphology.remove_small_objects(LSA_seg, min_size=5, connectivity=1) # input should be bool
    LSA_seg = morphology.remove_small_holes(LSA_seg, area_threshold=3000, connectivity=1) # return boolean
    assert isinstance(LSA_seg[0,0,0], np.bool_) == True
    LSA_seg_img = nib.Nifti1Image(LSA_seg, ref_img.affine, ref_img.header)

    ## 3. Label islands and get endpoints for each island 
    ## This is done before upsampling because it was found that this produces fewer wrong endpoints
    labelled, num_islands = measure.label(LSA_seg>0.5, return_num=True, connectivity=1)
    for label in np.arange(num_islands)+1:
        island_seg = labelled==label

        island_skeleton = (skeletonize(island_seg)>0.5).astype(int)
        island_endpoint_mask, island_endpoint_lst = get_endpoints_in_3d(island_skeleton)

        # Save to json file
        island_endpoint_save_path = os.path.join(save_dir, f"F{label}.json")
        save_fudicial_points_for_slicer(island_endpoint_lst, ref_img.affine, island_endpoint_save_path, label_pts=True, label_prefix=f"F{label}-", locked=False, unselect_lowest_pt=True)

  
    # upsample labelled segmentation
    labelled_img = nib.Nifti1Image(labelled, ref_img.affine)
    if upsample:
        upsampled_labelled_img = upsample_seg(labelled_img, factor=3, interpolation_method='nearest') 
    else:
        upsampled_labelled_img = labelled_img

    postprocessed_save_path = os.path.join(save_dir, out_seg_name)
    nib.save(upsampled_labelled_img, postprocessed_save_path)
    return postprocessed_save_path



def crop_TOF_and_seg(ID, side, save_dir, upsampled_TOF_path, seg_path, LSA_ROI_dict, LSA_PA_mask_path=None, closing=False, auto_remove_distant_islands=False, remove_top_surface_islands=False):
    TOF_img = nib.load(upsampled_TOF_path)
    TOF = TOF_img.get_fdata()
    seg_img = nib.load(seg_path)
    seg = seg_img.get_fdata()
    if LSA_PA_mask_path != None:
        LSA_PA_mask_img = nib.load(LSA_PA_mask_path)
        LSA_PA_mask = LSA_PA_mask_img.get_fdata()


    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)

    cropped_TOF_prefix = f"{ID}_upsampled_TOF"
    cropped_seg_prefix = f"{ID}_seg"
    if LSA_PA_mask_path != None:
        cropped_mask_prefix = f"{ID}_LSA_PA_mask"

    # Read in ROI min and max idx
    if LSA_PA_mask_path == None:
        min_idx = LSA_ROI_dict[ID][side]['min_idx']
        max_idx = LSA_ROI_dict[ID][side]['max_idx']
    else:
        min_idx = LSA_ROI_dict[side]['min_idx']
        max_idx = LSA_ROI_dict[side]['max_idx']

    # Crop TOF in cube
    cropped_TOF, cropped_TOF_path= crop_nifti(
        TOF, 
        min_idx, 
        max_idx, 
        save_dir, 
        TOF_img.affine, 
        TOF_img.header, 
        new_dir_per_cube=False, 
        save_name=cropped_TOF_prefix, 
        return_savepath=True)

    # Crop seg in mask
    seg_to_crop = seg if LSA_PA_mask_path == None else seg*(LSA_PA_mask>0.5)
    cropped_seg, cropped_seg_path = crop_nifti(
        seg_to_crop, 
        min_idx, 
        max_idx, 
        save_dir, 
        seg_img.affine, 
        seg_img.header, 
        new_dir_per_cube=False, 
        save_name=cropped_seg_prefix, 
        return_savepath=True)
    
    if LSA_PA_mask_path != None:
        # Crop mask in cube
        cropped_mask, cropped_mask_path = crop_nifti(
            LSA_PA_mask, 
            min_idx, 
            max_idx, 
            save_dir, 
            LSA_PA_mask_img.affine, 
            LSA_PA_mask_img.header, 
            new_dir_per_cube=False, 
            save_name=cropped_mask_prefix, 
            return_savepath=True)
    
    # Label cropped segmentation
    get_labelled_seg(cropped_seg_path, save_dir, closing=closing, 
                    auto_remove_distant_islands=auto_remove_distant_islands, 
                    get_helper_endpoints=True, 
                    remove_top_surface_islands=remove_top_surface_islands)

    if LSA_PA_mask_path == None:
        return cropped_TOF_path, cropped_seg_path
    else:
        return cropped_TOF_path, cropped_seg_path, cropped_mask_path
    
        




############################################################################################
######### Functions for computing morphological metrics from extracted centerlines #########
############################################################################################

def get_euclidean_distance(point1, point2):
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(point1, point2)]))

def get_tortuosity(start, end, curve_length):
    straight_distance = get_euclidean_distance(start, end)
    return curve_length/straight_distance
 
def get_length(control_points): # control_points is a list of dicts in the format in .mrk.json files
    total_length = 0.0
    for i in range(len(control_points) - 1):
        point1 = control_points[i]
        point2 = control_points[i + 1]
        segment_length = get_euclidean_distance(point1, point2)
        total_length += segment_length
    return total_length

def get_curvature(control_points, downsample_distance=0.5, return_kappa_control_points=True):
    # Compute average distances between consecutive control points, as it is known that this was set fixed upon extraction
    # print(f"Computing curvature with downsample distance: {downsample_distance}")
    distances = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
    delta = round(np.mean(distances),2)
    if delta < downsample_distance: #downsample control points for more stable results
        factor = round(downsample_distance/delta)
        downsampled_control_points = control_points[::factor]
    else:
        downsampled_control_points = control_points
    
    n = downsampled_control_points.shape[0]
    if n < 3:
        kappa = np.zeros(1)

        return kappa, downsampled_control_points[0]
    T = np.zeros((n, 3))
    N = np.zeros((n, 3))
    kappa = np.zeros(n)

    # Compute tangents and normals using finite differences
    for i in range(1, n-1): #kappa[0] and kappa[-1] will remain 0
        # Tangent vector (first derivative)
        T[i, :] = (downsampled_control_points[i+1, :] - downsampled_control_points[i-1, :]) / 2
        
        # Normal vector (second derivative)
        N[i, :] = (downsampled_control_points[i+1, :] - 2 * downsampled_control_points[i, :] + downsampled_control_points[i-1, :])

    # Compute curvature at each control point
    for i in range(1, n-1):
        # Cross product of tangent and normal vectors
        cross_product = np.cross(T[i, :], N[i, :])
        
        # Curvature
        kappa[i] = np.linalg.norm(cross_product) / np.linalg.norm(T[i, :])**3
    
    if return_kappa_control_points:
        return kappa[1:n-1], downsampled_control_points[1:n-1,:]
    else:
        return kappa[1:n-1]

def get_segment_info_from_json(json_path, metrics, downsample_distance=0.5):
    data = json.load(open(json_path, 'r'))
    control_points = data["markups"][0]["controlPoints"]
    start_point = control_points[0]["position"]
    end_point = control_points[-1]["position"]
    control_point_positions = np.array([pt["position"] for pt in control_points])

    segment_info = {
        "start": start_point,
        "end": end_point,
        "control_point_positions": control_point_positions
    }
    if "length" in metrics:
        curve_length = get_length(control_point_positions)
        segment_info["length"] = curve_length
    if "tortuosity" in metrics:
        tortuosity = get_tortuosity(start_point, end_point, curve_length)
        segment_info["tortuosity"] = tortuosity
    if ("mean_curvature" in metrics) or ("max_curvature" in metrics):
        kappa, kappa_control_points = get_curvature(control_point_positions, downsample_distance=downsample_distance)
        
        mean_curvature = np.mean(kappa)
        max_curvature = np.max(kappa)
        segment_info["kappa"] = kappa
        segment_info["mean_curvature"] = mean_curvature
        segment_info["max_curvature"] = max_curvature
        segment_info["kappa_control_points"] = kappa_control_points

    return segment_info


def get_segments_and_tree_from_dir(root_dir, metrics, segment_name_pattern = r"^C\d+ \(\d+\)$", downsample_distance=0.5):
    original_dir = os.getcwd()
    os.chdir(root_dir)
    stem_json_name_pattern = segment_name_pattern.replace("$", r"\.mrk\.json$")

    all_segments = []
    LSA_tree = {}
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if re.match(stem_json_name_pattern, file_name): # if the file has a name that matches Cx (0).mkr.json where x is any number
                segment_name = file_name.replace(".mrk.json", "")
                segment_info = get_segment_info_from_json(file_name, metrics, downsample_distance=downsample_distance)
                all_segments.append(segment_info)
                LSA_tree[segment_name] = segment_info

        for dir_name in dirs: # If there is a directory for a branch, that branch has children, so we need to run this function recursively
            if re.match(segment_name_pattern, dir_name): # if the directory name matches Cx (0)
                dir_path = os.path.join(root_dir, dir_name)
                children_segments, LSA_tree[dir_name]["children"] = get_segments_and_tree_from_dir(dir_path, metrics, downsample_distance=downsample_distance)
                all_segments += children_segments

        # Prevent os.walk from going into subdirectories
        dirs[:] = []
    os.chdir(original_dir)
    return all_segments, LSA_tree


def get_full_branchs(key, node, metrics):
    '''This function gets the full branchs from a node'''
    if "children" in node:
        branchs = []
        for child_key in list(node["children"].keys()):
            child_branchs = get_full_branchs(child_key, node["children"][child_key], metrics)
            # Add current node to children branchs 
            for child_branch in child_branchs:
                seg_to_add = {
                    "segments" : [key]+child_branch["segments"],
                    "start" : node["start"],
                    "end" : child_branch["end"]
                }
                if "length" in metrics:
                    seg_to_add["length"] = node["length"]+child_branch["length"]
                if ("mean_curvature" in metrics) or ("max_curvature" in metrics):
                    seg_to_add["kappa"] = np.concatenate((node['kappa'], child_branch['kappa']))
                branchs.append(seg_to_add)
            
    else:
        branchs = [{
            "segments" : [key],
            "start" : node["start"],
            "end" : node["end"]
        }]
        if "length" in metrics:
            branchs[0]["length"] = node["length"]
        if ("mean_curvature" in metrics) or ("max_curvature" in metrics):
            branchs[0]["kappa"] = node["kappa"]

    for branch in branchs:
        if "tortuosity" in metrics:
            branch["tortuosity"] = get_tortuosity(branch["start"], branch["end"], branch["length"])
        if "mean_curvature" in metrics:
            branch["mean_curvature"] = np.mean(branch["kappa"])
        if "max_curvature" in metrics:
            branch["max_curvature"] = np.max(branch["kappa"])
    
    return branchs

def get_all_LSA_metrics(results_dir, metrics):
    all_segments, LSA_tree = get_segments_and_tree_from_dir(results_dir, metrics)
    n_stems = len(LSA_tree)

    ## Get segment-level results
    segment_metric_lsts = {}
    for metric in metrics:
        segment_metric_lsts[metric] = [segment[metric] for segment in all_segments]

    ## Get full-branch results
    # Get branchs
    all_branchs = []
    for stem in list(LSA_tree.keys()):
        all_branchs += get_full_branchs(stem, LSA_tree[stem], metrics)
    n_branchs = len(all_branchs)
    branch_metric_lsts = {}
    for metric in metrics:
        branch_metric_lsts[metric] = [branch[metric] for branch in all_branchs]

    results = {
        "n_stems" : n_stems,
        "n_segments": len(all_segments),
        "n_branchs": len(all_branchs),
    }

    # Get longest branch tortuosity
    if "longest_branch_tortuosity" in metrics:
        all_branch_lengths = [branch["length"] for branch in all_branchs]
        longest_branch_idx = np.argmax(all_branch_lengths)
        longest_branch_tortuosity = all_branchs[longest_branch_idx]["tortuosity"]
        results["longest_branch_tortuosity"] = longest_branch_tortuosity

    for metric in metrics:
        results[f'segment_{metric}_mean'] = np.mean(segment_metric_lsts[metric])
        results[f'segment_{metric}_std'] = np.std(segment_metric_lsts[metric])
        results[f'segment_{metric}_median'] = np.median(segment_metric_lsts[metric])
        results[f'segment_{metric}_max'] = np.max(segment_metric_lsts[metric])
        
        results[f'branch_{metric}_mean'] = np.mean(branch_metric_lsts[metric])
        results[f'branch_{metric}_std'] = np.std(branch_metric_lsts[metric])
        results[f'branch_{metric}_median'] = np.median(branch_metric_lsts[metric])
        results[f'branch_{metric}_max'] = np.max(branch_metric_lsts[metric])

    full_results = {"all_segments": all_segments,
                    "all_branchs": all_branchs}
    return results, full_results

def get_voxel_idx(world_idx, affine_matrix, world_coords="LPS", affine_coords="RAS"):
    # world_idx: Nx3 array of world coordinates
    world_idx_transformed = world_idx.copy()
    if world_coords[0]!=affine_coords[0]:
        world_idx_transformed[:,0] = -world_idx[:,0]
    if world_coords[1]!=affine_coords[1]:
        world_idx_transformed[:,1] = -world_idx[:,1]
    if world_coords[2]!=affine_coords[2]:
        world_idx_transformed[:,2] = -world_idx[:,2]

    inverse_affine = np.linalg.inv(affine_matrix) 

    assert world_idx_transformed.shape[1] == 3
    world_idx_transformed = np.concatenate([world_idx_transformed, np.ones((world_idx_transformed.shape[0],1))], axis=1).T # 4xN
    voxel_idx = np.round(np.dot(inverse_affine, world_idx_transformed)[:3,:]).astype(int) # 3xN
    return voxel_idx.T # Nx3


if __name__ == '__main__':
    print("Running centerline_functions.py")