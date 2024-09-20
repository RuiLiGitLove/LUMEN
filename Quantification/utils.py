import os
import re
import json
import subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

def run_command(command_lst, print_command=False):
    if print_command:
        print("Running command:", command_lst)
    # subprocess.run(command, shell=True)

    # Run the command and capture its output
    # subprocess.run(command_lst)
    completed_process = subprocess.run(command_lst, capture_output=True, text=True)

    # Check if the command was successful (return code 0)
    if completed_process.returncode == 0:
        # Access the captured output
        output = completed_process.stdout
        print("Command Output:")
        print(output)
    else:
        # Handle errors if the command failed
        print("Command failed with return code:", completed_process.returncode)
        print("Error Output:")
        print(completed_process.stderr)

def afni_3dresample(input_path, output_path, isotropic=True, target_voxel_sizes=None, binary=False):
    # print("Resampling", input_path)
    if isotropic:
        img = nib.load(input_path)
        voxel_sizes = img.header.get_zooms()
        target_voxel_sizes = [min(voxel_sizes), min(voxel_sizes), min(voxel_sizes)]
    else:
        assert target_voxel_sizes!=None
        
    if binary:
        rmode = "NN"
    else:
        rmode = "Cu"

    afni_command = "3dresample -overwrite -dxyz {} {} {} -rmode {} -prefix {} -inset {}".format(
        target_voxel_sizes[0], 
        target_voxel_sizes[1], 
        target_voxel_sizes[2], 
        rmode,
        output_path, 
        input_path
        )
    # The -overwrite flag allows the command to overwrite an existing output dataset with the same name
    subprocess.run(afni_command, shell=True)

def crop_nifti(data, min_idx, max_idx, save_dir, affine, header, new_dir_per_cube=False, save_name=None, return_savepath=False, save_cropped=True):
    # Crop image
    try:
        cropped = data[min_idx[0]:(max_idx[0]+1), min_idx[1]:(max_idx[1]+1), min_idx[2]:(max_idx[2]+1)]
    except:
        print("Provided min_idx and max_idx are invalid!")
        return None
  
    # Compute affine
    cropped_affine = affine.copy()
    for i in range(3):
        cropped_affine[i,3] += min_idx[i] * affine[i,i]

    # Save results
    location_str = "x_{}_{}_y_{}_{}_z_{}_{}".format(
            min_idx[0], max_idx[0], 
            min_idx[1], max_idx[1],
            min_idx[2], max_idx[2])
    
    if new_dir_per_cube:
        # Create a savedir named as the cropping location & save_name will be appended to the location string
        cropped_save_dir = os.path.join(save_dir,location_str)
        if not os.path.exists(cropped_save_dir):
            os.mkdir(cropped_save_dir)
        savepath = os.path.join(cropped_save_dir, f"{location_str}_{save_name}.nii.gz")
    else: # Not creating a new dir for each cube & save_name is used as prefix
        savepath = os.path.join(save_dir, f"{save_name}_{location_str}.nii.gz")

    # Save image
    if save_cropped:
        cropped_img = nib.Nifti1Image(cropped, cropped_affine, header)
        nib.save(cropped_img, savepath)

    if return_savepath:
        return cropped, savepath
    else:
        return cropped

def get_bounding_box(image_data, mask=None):
    if mask is None:
        ROI_pxl_idxs = np.argwhere(image_data>0)
    else:
        ROI_pxl_idxs = np.argwhere(mask>0)
    min_idxs = ROI_pxl_idxs.min(axis=0)
    max_idxs = ROI_pxl_idxs.max(axis=0)
    #print("Bounding box min_idxs={}, max_idxs={}.".format(min_idxs, max_idxs))
    if ROI_pxl_idxs.shape[1]==3: # 3D image
        ROI_block = image_data[min_idxs[0]:(max_idxs[0]+1), min_idxs[1]:(max_idxs[1]+1), min_idxs[2]:(max_idxs[2]+1)]
    else:
        print("Cannot handle non-3D image data! Image data shape:", image_data.shape)
        ROI_block = image_data
    return ROI_block, (min_idxs, max_idxs)



def get_location_str(min_idx, max_idx):
    location_str = "x_{}_{}_y_{}_{}_z_{}_{}".format(
        min_idx[0], max_idx[0],
        min_idx[1], max_idx[1],
        min_idx[2], max_idx[2]
    )
    return location_str

def parse_indices_from_filename(filename):
    # Regular expressions to extract numbers following x, y, and z
    x_matches = re.findall(r'x_(\d+)_(\d+)', filename)
    y_matches = re.findall(r'y_(\d+)_(\d+)', filename)
    z_matches = re.findall(r'z_(\d+)_(\d+)', filename)

    # Ensure at least one match for each coordinate
    if x_matches and y_matches and z_matches:
        x_min, x_max = int(x_matches[0][0]), int(x_matches[0][1])
        y_min, y_max = int(y_matches[0][0]), int(y_matches[0][1])
        z_min, z_max = int(z_matches[0][0]), int(z_matches[0][1])

        min_idx = np.array([x_min, y_min, z_min])
        max_idx = np.array([x_max, y_max, z_max])

        print(f"x_min: {x_min}, x_max: {x_max}")
        print(f"y_min: {y_min}, y_max: {y_max}")
        print(f"z_min: {z_min}, z_max: {z_max}")
        
        return min_idx, max_idx
    else:
        print("Could not find coordinate values.")
        return None



def read_in_binary_img(path):
    img = nib.load(path)
    data = img.get_fdata()
    data = (data>0.5).astype(int)
    return img, data
    
def change_img_affine(img_path, target_img_path, save_path):
    target_img = nib.load(target_img_path)
    img = nib.load(img_path)
    data = img.get_fdata()
    save_img = nib.Nifti1Image(data, target_img.affine, target_img.header)
    nib.save(save_img, save_path)

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
        if save_path:
            plt.imsave(save_path, mip.T,
                  origin='lower',
                  cmap=cmaps[0])
        else:
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
    elif save_path and (num_img > 1):
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.close()
       

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

def get_endpoints_in_3d(skel):
    skel = (skel>0.5).astype(int)
    filter = np.ones((3,3,3), dtype=int)
    convolved = convolve(skel, filter, mode='constant', cval=0)
    endpoint_mask = np.zeros_like(skel)
    endpoint_mask[(skel==1) & (convolved==2)] = 1
    endpoint_lst = np.argwhere(endpoint_mask==1).tolist()
    # print("Detected {} endpoints!".format(len(endpoint_lst)))
    return endpoint_mask, endpoint_lst

def get_branching_points_in_3d(skel):
    skel = (skel>0.5).astype(int)
    filter = np.ones((3,3,3), dtype=int)
    convolved = convolve(skel, filter, mode='constant', cval=0)
    branching_point_mask = np.zeros_like(skel)
    branching_point_mask[(skel==1) & (convolved>3)] = 1
    branching_point_lst = np.argwhere(branching_point_mask==1).tolist()
    # print("Detected {} branching_points!".format(len(branching_point_lst)))

    return branching_point_mask, branching_point_lst
 
def save_fudicial_points_for_slicer(endpoint_lst, affine_mat, save_path, label_prefix="F", locked="false"):
    #locked can be "false" or "true"
    selected_endpoints = []
    for idx, pt_index_loc in enumerate(endpoint_lst):
        pt_RAS_loc = np.dot(affine_mat, pt_index_loc+[1])

        pt_info = {
            "label":f"{label_prefix}-{idx+1}",
            "position": [pt_RAS_loc[0], pt_RAS_loc[1], pt_RAS_loc[2]]
        }
        selected_endpoints.append(pt_info)

    endpoints_json = {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [{
            "type": "Fiducial", 
            "coordinateSystem": "RAS", 
            "controlPoints": selected_endpoints,
            "locked": locked
        }]
    }

    json.dump(endpoints_json, open(save_path, 'w'), indent=4)

if __name__ == '__main__':
    print("Running utils.py")
