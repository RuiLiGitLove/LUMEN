import os
import vtk
import slicer
import argparse
import ExtractCenterline

def exportMarkupCurveNodes(shFolderItemId, outputFolder):
    slicer.util.showStatusMessage("Saving centerline curves...")
    # Get items in the folder
    childIds = vtk.vtkIdList() # Creates an empty list
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    shNode.GetItemChildren(shFolderItemId, childIds) # Populates the childIds list with the IDs of all children nodes under the current shFolderItemId
    if childIds.GetNumberOfIds() == 0:
        return

    # Prepare output folder
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Loop through each node in scene
    for itemIdIndex in range(childIds.GetNumberOfIds()):
        shItemId = childIds.GetId(itemIdIndex)
        # Write node to file (if it is a markup curve node)
        dataNode = shNode.GetItemDataNode(shItemId)
        if dataNode and dataNode.IsA("vtkMRMLMarkupsCurveNode"):
            storageNode = dataNode.GetStorageNode()
            if storageNode:
                filename = dataNode.GetName() + ".mrk.json" #os.path.basename(storageNode.GetFileName())
                filepath = os.path.join(outputFolder, filename)
                slicer.util.exportNode(dataNode, filepath)
            # Write all children of this child item
            grandChildIds = vtk.vtkIdList()
            shNode.GetItemChildren(shItemId, grandChildIds)
            if grandChildIds.GetNumberOfIds() > 0:
                exportMarkupCurveNodes(shItemId, os.path.join(outputFolder, shNode.GetItemName(shItemId)))

def smooth(inputModel, outputModel, method='Taubin', iterations=10, laplaceRelaxationFactor=0.5, taubinPassBand=0.1, boundarySmoothing=True):
    """Smoothes surface model using a Laplacian filter or Taubin's non-shrinking algorithm.
       This function was taken from the surface toolbox.
    """
    if method == "Laplace":
      smoothing = vtk.vtkSmoothPolyDataFilter()
      smoothing.SetRelaxationFactor(laplaceRelaxationFactor)
    else:  # "Taubin"
      smoothing = vtk.vtkWindowedSincPolyDataFilter()
      smoothing.SetPassBand(taubinPassBand)
    smoothing.SetBoundarySmoothing(boundarySmoothing)
    smoothing.SetNumberOfIterations(iterations)
    smoothing.SetInputData(inputModel.GetPolyData())
    smoothing.Update()
    outputModel.SetAndObservePolyData(smoothing.GetOutput())



def run_extract_centerline(labelled_LSA_seg_path, end_pts_path_format, cropped_TOF_path, results_dir, mip_vp_file_path):
    os.makedirs(results_dir, exist_ok=True)
    extract_centerline_parameters = {
        "curveSamplingDistance": 0.03
    }

    # Load TOF
    volume_node=slicer.util.loadVolume(cropped_TOF_path)
    volRenLogic = slicer.modules.volumerendering.logic()
    volume_display_node = volRenLogic.CreateDefaultVolumeRenderingNodes(volume_node)
    if mip_vp_file_path:
        volume_property_node = volRenLogic.AddVolumePropertyFromFile(mip_vp_file_path)
        volume_display_node.GetVolumePropertyNode().Copy(volume_property_node)
    slicer.app.processEvents()

    # Load LSA segmentation
    LSA_seg_node = slicer.util.loadSegmentation(labelled_LSA_seg_path)
    LSA_seg = LSA_seg_node.GetSegmentation()
    LSA_seg.SetConversionParameter("Smoothing factor", "0.3")
    LSA_seg_node.CreateClosedSurfaceRepresentation() # This creates the 3D view of the LSA_seg
    num_segments = LSA_seg.GetNumberOfSegments()

    # Export visible segments to model
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    exportFolderItemID = shNode.CreateFolderItem(shNode.GetSceneItemID(), f"{LSA_seg_node.GetName()}-models")
    slicer.modules.segmentations.logic().ExportAllSegmentsToModels(LSA_seg_node, exportFolderItemID)


    childIds = vtk.vtkIdList()
    shNode.GetItemChildren(exportFolderItemID, childIds)
    for segmentIndex in range(childIds.GetNumberOfIds()):
        modelShItemID = childIds.GetId(segmentIndex)
        modelNode = shNode.GetItemDataNode(modelShItemID)
        assert segmentIndex == (int(modelNode.GetName().split('_')[1])-1)
        segmentName = LSA_seg.GetNthSegment(segmentIndex).GetName()        

        #Create a surface model for this segment and smooth it
        surface_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"surface_{segmentIndex+1}")
        smooth(modelNode, surface_model_node)
         
        # Load endpoints for the current segment
        end_pts_path = end_pts_path_format.replace("[x]", str(segmentIndex + 1))
        end_pts = slicer.util.loadMarkups(end_pts_path)
        slicer.app.processEvents()
        ### Extract Centerline ###
        extractLogic = ExtractCenterline.ExtractCenterlineLogic()

        ## Preprocess ##
        inputSurfacePolyData = surface_model_node.GetPolyData()

        if not inputSurfacePolyData or inputSurfacePolyData.GetNumberOfPoints() == 0:
            raise ValueError(f"Valid input surface is required for segment '{segmentName}'")

        #Parameters#
        preprocessEnabled = True
        targetNumberOfPoints = 50000.0
        decimationAggressiveness = 1.0
        subdivideInputSurface = False
        curveSamplingDistance = 0.03
        slicer.util.showStatusMessage("Preprocessing surface before centerline extraction...")
        slicer.app.processEvents()
        preprocessedPolyData = extractLogic.preprocess(
            inputSurfacePolyData,
            targetNumberOfPoints,
            decimationAggressiveness,
            subdivideInputSurface,
        )
        ## Extract centerline ##
        centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", f"C{segmentIndex+1}")

        slicer.util.showStatusMessage(f"Extracting centerline for segment '{segmentName}'...")
        slicer.app.processEvents()  # force update
        centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(preprocessedPolyData, end_pts, curveSamplingDistance=curveSamplingDistance)
        
        slicer.util.showStatusMessage(f"Generating curves for segment '{segmentName}'...")
        slicer.app.processEvents()  # force update

        extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode=None, curveSamplingDistance=curveSamplingDistance)

        if centerlineCurveNode.GetNumberOfControlPoints() == 2:
            slicer.util.errorDisplay(f"Centerline generation failed for segment '{segmentName}'.")
        
        # Remove surface model
        slicer.mrmlScene.RemoveNode(surface_model_node)

    # Save the centerline markup nodes  
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    slicer.app.ioManager().addDefaultStorageNodes()
    exportMarkupCurveNodes(shNode.GetSceneItemID(), results_dir)  
 
    # Save scene
    LSA_seg_node.GetDisplayNode().SetVisibility(False) # Hide segmentation
    # slicer.mrmlScene.RemoveNode(LSA_seg_node)
    # exportFolderNode = shNode.GetItemDataNode(exportFolderItemID) # Hide export folder
    # exportFolderNode.SetVisibility(False)
    
    shNode.RemoveItem(exportFolderItemID) # Remove folder of exported models

    slicer.util.showStatusMessage("Saving scene ...")
    slicer.util.saveScene(os.path.join(results_dir, "scene.mrb"))

    slicer.util.showStatusMessage("Extract centerline finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script runs the centerline extraction in 3D Slicer.")
    parser.add_argument('-load_dir', help='Directory containing the labelled LSA segmentation and endpoint json files.')
    parser.add_argument('-seg_filename', help='Filename of the labelled LSA segmentation.')
    parser.add_argument('-TOF_path', help="Path to cropped TOF file")
    parser.add_argument('-results_dir', help="Directory to save results")
    parser.add_argument('-mip_volume_property', help="Path to MIP volume property")
    args = parser.parse_args()


    load_dir = args.load_dir
    results_dir = args.results_dir 
    upsampled_labelled_LSA_seg_path = os.path.join(load_dir, args.seg_filename)
    end_pts_path_format = os.path.join(load_dir, "F[x].json")
    cropped_TOF_path = args.TOF_path
    mip_vp_file_path = args.mip_volume_property
    run_extract_centerline(upsampled_labelled_LSA_seg_path, end_pts_path_format, cropped_TOF_path, results_dir, mip_vp_file_path)
