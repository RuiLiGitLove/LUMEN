# Python commands in this file are executed on Slicer startup


## Change default settings for 3D view
for viewNode in slicer.util.getNodesByClass("vtkMRMLViewNode"):
  # Make 3D box invisible
  viewNode.SetBoxVisible(False)

  # Change background color to black
  viewNode.SetBackgroundColor(0,0,0)
  viewNode.SetBackgroundColor2(0,0,0)


## Change default volume rendering to MIP
for viewNode in slicer.util.getNodesByClass("vtkMRMLViewNode"):
    viewNode.SetRaycastTechnique(slicer.vtkMRMLViewNode.MaximumIntensityProjection)
    # Load self-defined MIP vp file
    mip_vp_file = r"/path/to/LSA_3D_Morph/Quantification/MIP_VolumeProperty.vp" # TODO: Change this to the path on your computer
    volRenLogic = slicer.modules.volumerendering.logic()
    volRenLogic.AddVolumePropertyFromFile(mip_vp_file)

## Define the function to get ROI slice indices
def get_ROI_slice_idx(ROI_node_name):
    import numpy as np
    roiNode = slicer.util.getNodes()[ROI_node_name]
    bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    roiNode.GetRASBounds(bounds) # This updates bounds to the current bounds in WORLD COORDINATE, in the form of [x_min, x_max, y_min, y_max, z_min, z_max]

    # Get affine of volume node
    volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    transformMatrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(transformMatrix) # This updates transformMatrix
    affine_matrix = np.array([
        [transformMatrix.GetElement(0, 0), transformMatrix.GetElement(0, 1), transformMatrix.GetElement(0, 2), transformMatrix.GetElement(0, 3)],
        [transformMatrix.GetElement(1, 0), transformMatrix.GetElement(1, 1), transformMatrix.GetElement(1, 2), transformMatrix.GetElement(1, 3)],
        [transformMatrix.GetElement(2, 0), transformMatrix.GetElement(2, 1), transformMatrix.GetElement(2, 2), transformMatrix.GetElement(2, 3)],
        [transformMatrix.GetElement(3, 0), transformMatrix.GetElement(3, 1), transformMatrix.GetElement(3, 2), transformMatrix.GetElement(3, 3)]
    ])
    assert affine_matrix.shape==(4,4)

    # Compute inverse of affine
    inverse_affine = np.linalg.inv(affine_matrix)
    min_world_idx = [bounds[0], bounds[2], bounds[4], 1.0]
    max_world_idx = [bounds[1], bounds[3], bounds[5], 1.0]
    min_slice_idx = np.round(np.dot(inverse_affine, min_world_idx)[:-1]).astype(int)
    max_slice_idx = np.round(np.dot(inverse_affine, max_world_idx)[:-1]).astype(int)
    ROI_size =max_slice_idx-min_slice_idx+1

    print("ROI location:")
    print(f"\"min_idx\":[{min_slice_idx[0]}, {min_slice_idx[1]}, {min_slice_idx[2]}],")
    print(f"\"max_idx\":[{max_slice_idx[0]}, {max_slice_idx[1]}, {max_slice_idx[2]}],")
    print(f"\"ROI_size\":[{ROI_size[0]}, {ROI_size[1]}, {ROI_size[2]}]")
    return None


## Define the function to load postprocessed segmentation and endpoints
import os
import re
def load_postprocessed_seg(folderPath):
    for fname in os.listdir(folderPath):
        path = os.path.join(folderPath, fname)
        baseName = os.path.splitext(fname)[0]

        # 1) NIfTI -> Segmentation
        if fname.lower().endswith((".nii", ".nii.gz")):
            segNode = slicer.util.loadSegmentation(
                path,
                properties={}  # no custom name: let Slicer pick
            )
            # Clean off any _n suffix Slicer added
            origName = segNode.GetName()
            cleanName = re.sub(r'_\d+$', '', origName)
            segNode.SetName(cleanName)
            print(f"Loaded segmentation: {cleanName}")

        # 2) JSON -> Markups
        elif fname.lower().endswith(".json"):
            mrkNode = slicer.util.loadMarkups(path)
            origName = mrkNode.GetName()
            cleanName = re.sub(r'_\d+$', '', origName)
            mrkNode.SetName(cleanName)
            print(f"Loaded markups: {cleanName}")