import os
import json
from glob import glob
from MSFDF_main import run_MSFDF

if __name__ == '__main__':
    ###### TODO: Input the following ######
    ID_lst = ['001'] # List of IDs

    # OR: read IDs from an input folder
    # input_folder = "path/to/input/tofs"
    # input_file_lst = glob(os.path.join(input_folder, "*_upsampled_TOF.nii.gz"))
    # ID_lst = [os.path.basename(p).replace("_upsampled_TOF.nii.gz", "") for p in input_file_lst]

    orig_TOF_path_format = "path/to/[ID]_upsampled_TOF.nii.gz" # path format to the isotropically upsampled TOF file. [ID] will be replaced by the IDs in ID_lst.
    out_folder_format = os.path.join("path/to/save/folder", "[ID]") # path format to the output folder for each ID.
    
    plot_MIP_images = False # By setting plot_MIP to True, you can plot MIP images of TOF and the final segmentation. But this requires saving the MIP index range for each ID in the json file 'MIP_range.json'
    MIP_dir = "cor"   # "cor" for coronal view; in MIP_range.json, you need to specify the range of slice indices in the second axis
    mip_idx_range = [0, 230] # change to your range. If this is specified, the same mip range will be applied to all subjects.
    mip_idx_range_path= None # only used when mip_idx_range is None
    clear_inter_output = True
    #######################################
    
    root_dir = "path/to/LUMEN/MSFDF" # path to the MSFDF folder
    brain_mask_path_format = None # optional path format to the brain mask file. 
    hyperparam_path = os.path.join(root_dir, "MSFDF_hyperparams.json")    
    hyperparams = json.load(open(hyperparam_path, "r"))

    if mip_idx_range is None:
        mip_idx_range_dict = json.load(open(mip_idx_range_path, "r"))

    # Usually don't change this config
    config = { 
        "filtering_method": "ITK_Multiscale", 
        "combination_method": "Maxpooling",  # tested that Maxpooling is better than FA
        "number_of_sigma_steps": 15,    # The more the better, between 10 and 15 is good
        "plot_mip": plot_MIP_images,
        "mip_dir": MIP_dir,
        "clear_inter_output": clear_inter_output
    }

    # Run MSFDF
    os.chdir(root_dir)
    for ID in ID_lst:
        orig_TOF_path = orig_TOF_path_format.replace("[ID]", ID)
        out_folder = out_folder_format.replace("[ID]", ID)
        brain_mask_path = brain_mask_path_format.replace("[ID]", ID) if brain_mask_path_format is not None else None
        if plot_MIP_images:
            if mip_idx_range is None:
                config["mip_idx_range"] = mip_idx_range_dict[ID]
            else:
                config["mip_idx_range"] = mip_idx_range
        run_MSFDF(ID, out_folder, orig_TOF_path, hyperparams, config, brain_mask_path)