import numpy as np
import SimpleITK as sitk
import os
from scipy.ndimage import measurements
import networkx as nx
import time as timing

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

def cell_center(seg_img):
    """Calculate cell centers with additional features"""
    results = {}
    for label in np.unique(seg_img):
        if label != 0:
            mask = seg_img == label
            all_points = np.argwhere(mask)
            avg_pos = np.round(np.mean(all_points, axis=0))
            area = np.sum(mask)
            results[label] = {
                'center': avg_pos[:2],  # Only x,y for 2D
                'area': area
            }
    return results

def compute_cell_location(seg_img):
    """Enhanced graph construction with multiple features"""
    g = nx.Graph()
    centers = cell_center(seg_img)
    all_labels = np.unique(seg_img)

    # Add nodes with features
    for label in all_labels:
        if label != 0:
            g.add_node(label, **centers[label])
    
    # Add edges with combined distance and area similarity
    for i in all_labels:
        if i != 0:
            for j in all_labels:
                if j != 0 and i != j:
                    # Position distance
                    pos_dist = np.linalg.norm(centers[i]['center'] - centers[j]['center'])
                    g.add_edge(i, j, weight=pos_dist)
    return g

def tracklet(g1, g2, seg_img1, seg_img2, maxtrackid, time, linelist, tracksavedir):
    f1 = {}
    f2 = {}
    new_seg_img2 = np.zeros(seg_img2.shape)
    dict_associate = {}
    
    # Get cell features
    cellcenter1 = cell_center(seg_img1)
    cellcenter2 = cell_center(seg_img2)
    
    # Store features with graph properties
    loc1 = g1.degree(weight='weight')
    loc2 = g2.degree(weight='weight')
    
    for ele1 in loc1:
        cell = ele1[0]
        f1[cell] = {
            'center': cellcenter1[cell]['center'],
            'degree': ele1[1],
            'area': cellcenter1[cell]['area'],
            'original_id': cell  # Track original ID
        }
    
    for ele2 in loc2:
        cell = ele2[0]
        f2[cell] = {
            'center': cellcenter2[cell]['center'],
            'degree': ele2[1],
            'area': cellcenter2[cell]['area']
        }
    
    # Associate cells based on minimum distance
    for cell2 in f2.keys():
        min_distance = float('inf')
        best_match = None
        
        for cell1 in f1.keys():
            pos1 = f1[cell1]['center']
            pos2 = f2[cell2]['center']
            distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
            
            if distance < min_distance:
                min_distance = distance
                best_match = cell1
        
        dict_associate[cell2] = best_match
    
    # Create inverse mapping (parent -> daughters)
    inverse_dict_ass = {}
    for cell in dict_associate:
        parent = dict_associate[cell]
        if parent not in inverse_dict_ass:
            inverse_dict_ass[parent] = []
        inverse_dict_ass[parent].append(cell)
    
    # Process each parent cell
    for parent in inverse_dict_ass:
        daughters = inverse_dict_ass[parent]
        original_parent_id = f1[parent]['original_id'] if 'original_id' in f1[parent] else parent
        
        if len(daughters) > 1:
            # Division case - create new IDs
            for cellin2 in daughters[:2]:  # Limit to 2 daughters
                maxtrackid += 1
                new_seg_img2[seg_img2 == cellin2] = maxtrackid
                string = '{} {} {} {}'.format(maxtrackid, time + 1, time + 1, original_parent_id)
                linelist.append(string)
        else:
            # Continuation case - keep original ID
            cellin2 = daughters[0]
            new_seg_img2[seg_img2 == cellin2] = original_parent_id
            
            # Update the existing track
            updated = False
            for i, line in enumerate(linelist):
                parts = line.split()
                if int(parts[0]) == original_parent_id and int(parts[2]) == time:
                    new_string = '{} {} {} {}'.format(
                        parts[0], parts[1], time + 1, parts[3])
                    linelist[i] = new_string
                    updated = True
                    break
            
            if not updated:
                string = '{} {} {} {}'.format(
                    original_parent_id, time + 1, time + 1, original_parent_id)
                linelist.append(string)
    
    # Save images
    img1 = sitk.GetImageFromArray(seg_img1.astype('uint16'))
    img2 = sitk.GetImageFromArray(new_seg_img2.astype('uint16'))
    filename1 = 'predict_' + '%0*d' % (6, time) + '.tif'
    filename2 = 'predict_' + '%0*d' % (6, time + 1) + '.tif'
    sitk.WriteImage(img1, os.path.join(tracksavedir, filename1))
    sitk.WriteImage(img2, os.path.join(tracksavedir, filename2))
    
    return maxtrackid, linelist

def predict_dataset_2(path, output_path):
    folder1 = output_path
    folder2 = path
    times = len(os.listdir(folder2))
    maxtrackid = 0
    linelist = []
    total_start_time = timing.time()
    
    for time in range(times):
        print('Processing frame {}'.format(time))
        start_time = timing.time()
        threshold = 100
        
        # Initialize first frame
        if time == 0:
            file1 = 'predict_000000.tif'
            img1 = sitk.ReadImage(os.path.join(folder2, file1))
            img1 = sitk.GetArrayFromImage(img1)
            img1_label, img1_counts = np.unique(img1, return_counts=True)

            # Remove small objects
            for l in range(len(img1_label)):
                if img1_counts[l] < threshold:
                    img1[img1 == img1_label[l]] = 0
            
            # Assign sequential IDs starting from 1
            labels = np.unique(img1)
            current_id = 1
            new_img1 = np.zeros_like(img1)
            for label in labels:
                if label != 0:
                    new_img1[img1 == label] = current_id
                    string = '{} {} {} {}'.format(current_id, time, time, 0)
                    linelist.append(string)
                    current_id += 1
            
            maxtrackid = current_id - 1
            img1 = new_img1
            img1_img = sitk.GetImageFromArray(img1.astype('uint16'))
            sitk.WriteImage(img1_img, os.path.join(folder1, file1))
            print('--------%s seconds-----------' % (timing.time() - start_time))
            continue
        
        # Process subsequent frames
        file1 = 'predict_' + '%0*d' % (6, time - 1) + '.tif'
        file2 = 'predict_' + '%0*d' % (6, time) + '.tif'
        
        img1 = sitk.ReadImage(os.path.join(folder1, file1))
        img2 = sitk.ReadImage(os.path.join(folder2, file2))
        img1 = sitk.GetArrayFromImage(img1)
        img2 = sitk.GetArrayFromImage(img2)
        
        if len(np.unique(img2)) < 2:  # Empty frame
            img2 = img1
            img2_img = sitk.GetImageFromArray(img2.astype('uint16'))
            sitk.WriteImage(img2_img, os.path.join(folder2, file2))
            continue
        
        # Remove small objects in current frame
        img2_label_counts = np.array(np.unique(img2, return_counts=True)).T
        for label, count in img2_label_counts:
            if count < threshold and label != 0:
                img2[img2 == label] = 0
        
        # Compute graphs for tracking
        g1 = compute_cell_location(img1)
        g2 = compute_cell_location(img2)
        
        maxtrackid, linelist = tracklet(g1, g2, img1, img2, maxtrackid, time - 1, linelist, folder1)
        print('--------%s seconds-----------' % (timing.time() - start_time))
    
    # Post-processing to ensure proper end frames
    final_linelist = []
    track_dict = {}
    
    # First pass to collect all appearances
    for line in linelist:
        parts = line.split()
        cell_id = int(parts[0])
        start = int(parts[1])
        end = int(parts[2])
        parent = int(parts[3])
        
        if cell_id not in track_dict:
            track_dict[cell_id] = {
                'start': start,
                'end': end,
                'parent': parent,
                'appearances': [(start, end)]
            }
        else:
            if start < track_dict[cell_id]['start']:
                track_dict[cell_id]['start'] = start
            if end > track_dict[cell_id]['end']:
                track_dict[cell_id]['end'] = end
            track_dict[cell_id]['appearances'].append((start, end))
    
    # Second pass to create final tracks
    for cell_id in sorted(track_dict.keys()):
        info = track_dict[cell_id]
        # Find the earliest parent reference
        parent = info['parent']
        if parent != 0:
            for app in info['appearances']:
                if app[0] == info['start']:
                    parent = track_dict[cell_id]['parent']
                    break
        
        final_linelist.append('{} {} {} {}'.format(
            cell_id, info['start'], info['end'], parent))
    
    # Save the final results
    filetxt = open(os.path.join(folder1, 'res_track.txt'), 'w')
    for line in final_linelist:
        filetxt.write(line)
        filetxt.write("\n")
    filetxt.close()
    
    print('Whole time sequence running time %s' % (timing.time() - total_start_time))

if __name__ == "__main__":
    predict_result = "data/res_result/"
    track_result = "data/track_result/"
    predict_dataset_2(predict_result, track_result)
