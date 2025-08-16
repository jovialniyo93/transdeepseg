import numpy as np
#from tifffile import imread, imsave
import logging
from tifffile import imread, imwrite
from pathlib import Path
from tracking.export import ExportResults
from tracking.tracker import TrackingConfig, MultiCellTracker

def get_predict_files(img_dir):
    """Find predict_XXXXXX.tif files with error handling"""
    img_dir = Path(img_dir)
    files = {}
    for f in sorted(img_dir.glob('predict_*.tif')):
        try:
            frame_num = int(f.stem.split('_')[-1])
            files[frame_num] = str(f)
        except (ValueError, IndexError):
            continue
    return files

def track(predict_path, output_path):
    predict_path = Path(predict_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Clean output directory first
    for f in output_path.glob('*'):
        if f.name.startswith(('predict_', 'mask', 'res_track')):
            f.unlink()

    segm_files = get_predict_files(predict_path)
    if not segm_files:
        available = list(predict_path.glob('*'))
        raise ValueError(
            f"No predict_XXXXXX.tif files found in {predict_path}\n"
            f"Found files: {[f.name for f in available[:3]] + (['...'] if len(available)>3 else [])}"
        )

    # Use first image to determine parameters
    first_img = np.squeeze(imread(segm_files[list(segm_files.keys())[0]]))
    
    # Calculate image density
    density = np.count_nonzero(first_img) / first_img.size
    
    # Dynamic parameters based on image size and density
    min_dim = min(first_img.shape)
    
    # Adaptive parameters
    if density > 0.3:  # High density
        roi_size = (min_dim//6, min_dim//6) if len(first_img.shape) == 2 else (min_dim//8, min_dim//8, min_dim//8)
        cutoff_dist = min_dim//10
        delta_t = 3
    else:  # Normal density
        roi_size = (min_dim//5, min_dim//5) if len(first_img.shape) == 2 else (min_dim//6, min_dim//6, min_dim//6)
        cutoff_dist = min_dim//8
        delta_t = 2

    config = TrackingConfig(
        img_files=segm_files,
        segm_files=segm_files,
        seeds=None,
        roi_box_size=roi_size,
        delta_t=delta_t,
        cut_off_distance=cutoff_dist,
        allow_cell_division=True
    )

    tracker = MultiCellTracker(config)
    tracks = tracker()
    
    # Analysis output
    print("\nTracking Statistics:")
    print(f"Total tracks detected: {len(tracks)}")
    avg_length = np.mean([len(t.masks) for t in tracks.values()])
    print(f"Average track length: {avg_length:.1f} frames")
    divisions = sum(1 for t in tracks.values() if t.successors)
    print(f"Cell divisions detected: {divisions}")

    # Export results
    exporter = ExportResults()
    exporter(tracks, output_path, first_img.shape, time_steps=sorted(segm_files.keys()))

    # Generate lineage file
    with open(output_path/'res_track.txt', 'w') as f:
        for track_id, track in tracks.items():
            times = sorted(track.masks.keys())
            f.write(f"{track_id} {times[0]} {times[-1]} {track.pred_track_id}\n")

def generate_output_files(tracks, segm_files, output_path):
    """Generate res_track.txt and tracked images"""
    # Lineage data
    with open(output_path/'res_track.txt', 'w') as f:
        for track_id, track in tracks.items():
            times = sorted(track.masks.keys())
            f.write(f"{track_id} {times[0]} {times[-1]} {track.pred_track_id}\n")

if __name__ == "__main__":
    predict_result = "mask" 
    track_result = "track"  
    track(predict_result, track_result)
