import slicer
import argparse

def smooth_mask(mask_path, volume_path, output_path):
    # slicer.util.mainWindow().showMinimized()
    # slicer.app.processEvents()
    slicer.util.showStatusMessage("Loading files")
    segmentationNode = slicer.util.loadSegmentation(mask_path)
    masterVolumeNode = slicer.util.loadVolume(volume_path)

    # Create segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(masterVolumeNode)

    # Smoothing
    slicer.util.showStatusMessage("Smoothing LSA PA mask")
    segmentEditorWidget.setActiveEffectByName("Smoothing")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("SmoothingMethod", "JOINT_TAUBIN")
    effect.setParameter("JointTaubinSmoothingFactor", 0.5)
    effect.self().onApply()

    # Clean up
    segmentEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    # Export segmentation to a labelmap
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode, masterVolumeNode)
    slicer.util.saveNode(labelmapVolumeNode, output_path)
    # slicer.util.saveNode(segmentationNode, output_path) # Saving it to an nrrd file will change the dimension of the mask due to unknown reasons
    slicer.util.showStatusMessage("Finished smoothing LSA PA mask")
    print(f"Finished smoothing LSA PA mask. Saved to {output_path}")

    slicer.util.exit()
    return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script postprocess the LSA PA mask in upsampled TOF space using functions from 3D Slicer.")
    parser.add_argument('-mask', help='Path to LSA PA mask')
    parser.add_argument('-volume', help='Path to the master volume the LSA PA mask corresponds to')
    parser.add_argument('-output', help='Path to save output mask')

    args = parser.parse_args()
    smooth_mask(args.mask, args.volume, args.output)