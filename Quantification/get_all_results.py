import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import json
import nrrd
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from quantification_functions import get_all_LSA_metrics, fractal_dimension

def get_all_results(full_ID_lst, 
        save_path, 
        ID_upsampled_TOF_path_format, 
        results_dir_format, 
        sum_metrics, 
        morph_metrics, 
        reductions,
        LSA_nrrd_path_format=None, 
        top_n=None, 
        terminal_min_length=None, 
        min_branch_length=0):

    full_ID_lst = sorted(full_ID_lst)
    print(f"There are {len(full_ID_lst)} subjects in total")
    levels = ["segment", "branch", "major_branch"]

    results = []
    for ID in full_ID_lst:
        ID_results = {
            "ID": ID
        }
        # Initialize all metrics
        for sum_metric in sum_metrics:
            ID_results[sum_metric] = 0
        for morph_metric in morph_metrics:
            for level in levels:
                ID_results[f"{level}_{morph_metric}"] = []
                for rdc in reductions:
                    ID_results[f"{level}_{morph_metric}_{rdc}"] = 0
        if 'FD' in morph_metrics:
            ID_results["FD"] = 0
        if 'tortuosity' in morph_metrics:
            ID_results["longest_branch_tortuosity"] = 0
        
        ID_upsampled_TOF_path = ID_upsampled_TOF_path_format.replace("[ID]", ID)
        ID_upsampled_TOF_img = nib.load(ID_upsampled_TOF_path)
        affine_matrix = ID_upsampled_TOF_img.affine
        ID_upsampled_TOF = ID_upsampled_TOF_img.get_fdata()

        for side in ["Left", "Right"]:
            print("Processing", ID, side)
            results_dir = results_dir_format.replace("[ID]", ID).replace("[SIDE]", side)
            side_results = get_all_LSA_metrics(results_dir, morph_metrics, affine_matrix, ID_upsampled_TOF, top_n=top_n, terminal_min_length=terminal_min_length, min_branch_length=min_branch_length)

            for sum_metric in sum_metrics:
                ID_results[f"{side}_{sum_metric}"] = side_results[sum_metric]
                ID_results[sum_metric] += side_results[sum_metric]
            for morph_metric in morph_metrics:
                # gather all morph metric values for segments and branches on this side
                for level in levels:
                    ID_results[f"{side}_{level}_{morph_metric}"] = [seg[morph_metric] for seg in side_results[f'all_{level}s']]
                    ID_results[f"{level}_{morph_metric}"] += ID_results[f"{side}_{level}_{morph_metric}"]
                    # calculate summary stats for single side
                    if 'mean' in reductions:
                        ID_results[f"{side}_{level}_{morph_metric}_mean"] = np.mean(ID_results[f"{side}_{level}_{morph_metric}"]) 
                    if 'median' in reductions:
                        ID_results[f"{side}_{level}_{morph_metric}_median"] = np.median(ID_results[f"{side}_{level}_{morph_metric}"]) 
                    if 'max' in reductions:
                        ID_results[f"{side}_{level}_{morph_metric}_max"] = np.max(ID_results[f"{side}_{level}_{morph_metric}"])
                    if 'std' in reductions:
                        ID_results[f"{side}_{level}_{morph_metric}_std"] = np.std(ID_results[f"{side}_{level}_{morph_metric}"])
            
            # Fractal dimension
            if 'FD' in morph_metrics:
                LSA_nrrd_path = LSA_nrrd_path_format.replace("[ID]", ID).replace("[SIDE]", side)
                LSA_seg, LSA_seg_header = nrrd.read(LSA_nrrd_path)
                assert LSA_seg.ndim == 3
                LSA_seg = LSA_seg > 0.5
                FD = fractal_dimension(LSA_seg)
                ID_results[f"{side}_FD"] = FD

            # Longest branch tortuosity
            if 'longest_branch_tortuosity' in morph_metrics:
                all_branchs = side_results["all_branchs"]
                all_branch_lengths = [branch["length"] for branch in all_branchs]
                longest_branch_idx = np.argmax(all_branch_lengths)
                longest_branch_tortuosity = all_branchs[longest_branch_idx]["tortuosity"]
                ID_results[f"{side}_longest_branch_tortuosity"] = longest_branch_tortuosity

        # calculate summary stats for both sides
        for morph_metric in morph_metrics:
            for level in levels:
                for rdc in reductions:
                    ID_results[f"{level}_{morph_metric}_{rdc}"] = (ID_results[f"Left_{level}_{morph_metric}_{rdc}"] + ID_results[f"Right_{level}_{morph_metric}_{rdc}"])/2
        if 'FD' in morph_metrics:
            ID_results['FD'] = (ID_results[f"Left_FD"] + ID_results[f"Right_FD"])/2
        if 'longest_branch_tortuosity' in morph_metrics:
            ID_results['longest_branch_tortuosity'] = (ID_results[f"Left_longest_branch_tortuosity"] + ID_results[f"Right_longest_branch_tortuosity"])/2
        results.append(ID_results)

    results = pd.DataFrame(results)
    results.to_csv(save_path, index=False)

if __name__ == "__main__":
    ## Get a list of ID
    # # Example 1: extract it from a csv file
    # subject_info_path = "study_subjects.csv"
    # subject_info = pd.read_csv(subject_info_path)
    # full_ID_lst = subject_info["Participant ID"].to_list()

    # Example 2: get it from the folder names in the data dir
    full_ID_lst = [f.name for f in Path("LUMEN/Data").iterdir() if f.is_dir()]

    save_path = "LUMEN/Data/results.csv"
    ID_upsampled_TOF_path_format = "LUMEN/Data/[ID]/[ID]_upsampled_TOF.nii.gz"
    results_dir_format = "LUMEN/Data/[ID]/[SIDE]/Results"

    sum_metrics = ["n_stems", "n_segments", "n_branchs", "n_branchs_above_cutoff"] # counting metrics
    morph_metrics = ['length', 'tortuosity'] # morphological metrics
    reductions = ['mean', 'median', 'max', 'std']
    min_branch_length = 0  # mm. Set a positive value here to exclude very short branches.

    get_all_results(full_ID_lst, 
                    save_path, 
                    ID_upsampled_TOF_path_format, 
                    results_dir_format, 
                    sum_metrics, 
                    morph_metrics,
                    reductions, 
                    min_branch_length=min_branch_length)