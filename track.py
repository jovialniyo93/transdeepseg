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
    results = {}
    for label in np.unique(seg_img):
        if label != 0:
            all_points_x, all_points_y = np.where(seg_img == label)
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_y))
            results[label] = [avg_x, avg_y]  # Only x and y for 2D images
    return results
            
def compute_cell_location(seg_img):
    g = nx.Graph()
    centers = cell_center(seg_img)
    all_labels = np.unique(seg_img)

    # Compute vertices
    for i in all_labels:
        if i != 0:
            g.add_node(i)
    
    # Compute edges based on distance between centers
    for i in all_labels:
        if i != 0:
            for j in all_labels:
                if j != 0 and i != j:
                    pos1 = centers[i]
                    pos2 = centers[j]
                    # Compute Euclidean distance for 2D (x, y) coordinates
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    g.add_edge(i, j, weight=distance)
    return g

def tracklet(g1, g2, seg_img1, seg_img2, maxtrackid, time, linelist, tracksavedir):
    f1 = {}
    f2 = {}
    new_seg_img2 = np.zeros(seg_img2.shape)
    dict_associate = {}
    cellcenter1 = cell_center(seg_img1)
    cellcenter2 = cell_center(seg_img2)
    loc1 = g1.degree(weight='weight')
    loc2 = g2.degree(weight='weight')
    for ele1 in loc1:
        cell = ele1[0]
        f1[cell] = [cellcenter1[cell], ele1[1]]
    for ele2 in loc2:
        cell = ele2[0]
        f2[cell] = [cellcenter2[cell], ele2[1]]
    for cell in f2.keys():
        tmp_center = f2[cell][0]
        min_distance = seg_img2.shape[0]**2 + seg_img2.shape[1]**2
        for ref_cell in f1.keys():
            ref_tmp_center = f1[ref_cell][0]
            distance = (tmp_center[0] - ref_tmp_center[0])**2 + (tmp_center[1] - ref_tmp_center[1])**2
            if distance < min_distance:
                dict_associate[cell] = ref_cell
                min_distance = distance
    inverse_dict_ass = {}
    for cell in dict_associate:
        if dict_associate[cell] in inverse_dict_ass:
            inverse_dict_ass[dict_associate[cell]].append(cell)
        else:
            inverse_dict_ass[dict_associate[cell]] = [cell]
    
    maxtrackid = max(maxtrackid, max(inverse_dict_ass.keys()))
    for cell in inverse_dict_ass.keys():
        if len(inverse_dict_ass[cell]) > 1:
            for cellin2 in inverse_dict_ass[cell]:
                maxtrackid += 1
                new_seg_img2[seg_img2 == cellin2] = maxtrackid
                string = '{} {} {} {}'.format(maxtrackid, time + 1, time + 1, cell)
                linelist.append(string)
        else:
            cellin2 = inverse_dict_ass[cell][0]
            new_seg_img2[seg_img2 == cellin2] = cell
            i = 0
            for line in linelist:
                i += 1
                if i == cell:
                    list_tmp = line.split()
                    new_string = '{} {} {} {}'.format(list_tmp[0], list_tmp[1], time + 1, list_tmp[3])
                    linelist[i - 1] = new_string
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
    
    for time in range(times - 1):
        print('Linking frame {} to previous tracked frames'.format(time + 1))
        start_time = timing.time()
        threshold = 100
        
        # Initialize first frame
        if time == 0:
            file1 = 'predict_000000.tif'
            img1 = sitk.ReadImage(os.path.join(folder2, file1))
            img1 = sitk.GetArrayFromImage(img1)
            img1_label, img1_counts = np.unique(img1, return_counts=True)

            for l in range(len(img1_label)):
                if img1_counts[l] < threshold:
                    img1[img1 == img1_label[l]] = 0
            labels = np.unique(img1)
            start_label = 0
            for label in labels:
                img1[img1 == label] = start_label
                start_label += 1
            img1 = sitk.GetImageFromArray(img1)
            sitk.WriteImage(img1, os.path.join(folder1, file1))
        
        # Process subsequent frames
        file1 = 'predict_' + '%0*d' % (6, time) + '.tif'
        file2 = 'predict_' + '%0*d' % (6, time + 1) + '.tif'
        img1 = sitk.ReadImage(os.path.join(folder1, file1))
        img2 = sitk.ReadImage(os.path.join(folder2, file2))
        img1 = sitk.GetArrayFromImage(img1)
        img2 = sitk.GetArrayFromImage(img2)
        
        if len(np.unique(img2)) < 2:
            img2 = img1
            img2_img = sitk.GetImageFromArray(img2)
            sitk.WriteImage(img2_img, os.path.join(folder2, file2))
            continue
        
        img2_label_counts = np.array(np.unique(img2, return_counts=True)).T
        i = 0
        for label in img2_label_counts[:, 0]:
            if img2_label_counts[i, 1] < threshold:
                img2[img2 == label] = 0
            i += 1
        labels = np.unique(img1)
        g1 = compute_cell_location(img1)
        g2 = compute_cell_location(img2)
        
        if time == 0:
            for cell in np.unique(img1):
                if cell != 0:
                    string = '{} {} {} {}'.format(cell, time, time, 0)
                    linelist.append(string)
                maxtrackid = max(cell, maxtrackid)
        
        maxtrackid, linelist = tracklet(g1, g2, img1, img2, maxtrackid, time, linelist, folder1)
        print('--------%s seconds-----------' % (timing.time() - start_time))
    
    # Save the results
    filetxt = open(os.path.join(folder1, 'res_track.txt'), 'w')
    for line in linelist:
        filetxt.write(line)
        filetxt.write("\n")
    filetxt.close()
    print('Whole time sequence running time %s' % (timing.time() - total_start_time))

if __name__ == "__main__":
    predict_result = "data/res_result/"
    track_result = "data/track_result/"
    predict_dataset_2(predict_result, track_result)
