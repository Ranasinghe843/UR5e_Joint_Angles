"""
Python Script to convert an fbx file containing motion capture data to a csv file
containing all the marker data (posiiotn and rotation) at each timestamp

Example Usage:
python fbx_to_csv.py /to/input.fbx path/to/output.csv
"""

import bpy
import csv
import sys

def export_motion_data_to_csv(fbx_file, output_csv):
    
    bpy.ops.import_scene.fbx(filepath=fbx_file)
    
    scene = bpy.context.scene

    action_start = min((action.frame_range[0] for action in bpy.data.actions), default=scene.frame_start)
    action_end = max((action.frame_range[1] for action in bpy.data.actions), default=scene.frame_end)
    scene.frame_start = int(action_start)
    scene.frame_end = int(action_end)
    
    fps = 60
    
    marker_names = [obj.name for obj in bpy.data.objects if obj.type == 'EMPTY']
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        header = ["Time"]
        for marker in marker_names:
            header.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z", f"{marker}_Rx", f"{marker}_Ry", f"{marker}_Rz"])
        writer.writerow(header)
        
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)  # Set the current frame
            
            time = frame / fps
            
            row = [f"{time:.4f}"]
            
            for marker in marker_names:
                obj = bpy.data.objects.get(marker)
                if obj:
                    loc = obj.location
                    rot = obj.rotation_euler  # Ensure rotation is in Euler format
                    row.extend([loc.x, loc.y, loc.z, rot.x, rot.y, rot.z])
                else:
                    row.extend([None] * 6)  # Placeholder for missing markers
            
            writer.writerow(row)
    
    print(f"Motion data exported from {fbx_file} to {output_csv}")

if __name__ == "__main__":
    try:
        export_motion_data_to_csv(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"An error occurred: {e}")