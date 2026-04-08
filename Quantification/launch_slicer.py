import argparse
import os
import slicer
from pathlib import Path

def launch_slicer(TOF_path=None, mip_volume_property_path=None, seg_path=None, seg_smoothing_factor=0.5, markup_pt_path=None):
    # Load TOF if provided
    if TOF_path is not None:
        volume_node=slicer.util.loadVolume(TOF_path)
        volRenLogic = slicer.modules.volumerendering.logic()
        volume_display_node = volRenLogic.CreateDefaultVolumeRenderingNodes(volume_node)
        
        # Apply custom volume property if given
        if mip_volume_property_path is not None:
            volume_property_node = volRenLogic.AddVolumePropertyFromFile(mip_volume_property_path)
            volume_display_node.GetVolumePropertyNode().Copy(volume_property_node)
        slicer.app.processEvents()

    # Load segmentation if provided
    if seg_path is not None:
        seg_node = slicer.util.loadSegmentation(seg_path)
        seg = seg_node.GetSegmentation()
        seg.SetConversionParameter("Smoothing factor", str(seg_smoothing_factor))
        seg_node.CreateClosedSurfaceRepresentation() # This creates the 3D view of the seg
    
    # Load fiducial points if provided
    if markup_pt_path is not None:
        fiducial_node = slicer.util.loadMarkups(markup_pt_path)
        fiducial_node_name = Path(file_path).name.replace(".mrk.json", "")
        fiducial_node.SetName(fiducial_node_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script launches 3D Slicer.")
    parser.add_argument('-TOF_path', default=None, help='Path to TOF volume')
    parser.add_argument('-mip_volume_property_path', default=None, help="Path to MIP volume property")
    parser.add_argument('-seg_path', default=None, help='Path to segmentation file')
    parser.add_argument('-seg_smoothing_factor', default=0.5, help='Smoothing factor for segmentation 3D display')
    parser.add_argument('-markup_pt_path', default=None, help='Path to markup points json file')
    args = parser.parse_args()

    launch_slicer(TOF_path=args.TOF_path, 
                  mip_volume_property_path=args.mip_volume_property_path,
                  seg_path=args.seg_path,
                  seg_smoothing_factor=args.seg_smoothing_factor,
                  markup_pt_path=args.markup_pt_path)
