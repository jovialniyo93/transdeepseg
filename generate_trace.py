import cv2
import os
import numpy as np
import random

# Function to compute centroids of each labeled cell in the segmented image
def get_center(serial, label, directory):
    track_picture = os.listdir(directory)
    track_picture = [file for file in track_picture if ".tif" in file]
    track_picture.sort()
    result_picture = cv2.imread(os.path.join(directory, track_picture[serial]), -1)
    label_picture = ((result_picture == label) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find and return centroid of the cell
    if contours:
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None

# Function to get a colored mask for each contour
def get_coloured_mask(mask):
    colours = [
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
        [255, 128, 0], [128, 255, 0], [0, 255, 128], [0, 128, 255], [128, 0, 255], [255, 0, 128],
        [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190], [0, 128, 0], [255, 165, 0]
    ]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    idx = random.randrange(0, len(colours))
    r[mask == 255], g[mask == 255], b[mask == 255] = colours[idx]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_trace(image_path, track_path, trace_path):
    track_picture = sorted([file for file in os.listdir(track_path) if ".tif" in file])
    test_image = sorted([file for file in os.listdir(image_path) if ".tif" in file])
    trace_image = []

    # Read parent-child relationship data from res_track.txt
    parent_data = {}
    file = os.path.join(track_path, "res_track.txt")
    with open(file, "r") as f:
        data = f.readlines()
    for line in data:
        parts = line.strip().split()
        cell_id = int(parts[0])
        parent_id = int(parts[3])
        parent_data[cell_id] = parent_id

    for i in range(len(test_image)):
        # Read and process the test image
        image_to_draw = cv2.imread(os.path.join(image_path, test_image[i]), -1)
        #image_to_draw = image_to_draw[6:742, 7:743]
        image_to_draw = np.stack((image_to_draw,) * 3, axis=2)

        # Read the corresponding tracking image
        result_picture = cv2.imread(os.path.join(track_path, track_picture[i]), -1)
        label_picture = ((result_picture >= 1) * 255).astype(np.uint8)

        # Draw white bounding boxes for each cell
        contours, _ = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # Draw the tracking label in each cell's centroid (white)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cell_id = result_picture[cy, cx]

            if cell_id == 0:
                continue

            # Check if the cell has a parent (indicating division)
            if cell_id in parent_data and parent_data[cell_id] != 0:
                label = f"{cell_id}({parent_data[cell_id]})"
            else:
                label = f"{cell_id}"

            cv2.putText(image_to_draw, label, (cx, cy), font, 0.5, (255, 255, 255), 1)

            # Apply a colored mask to each contour
            mask = np.zeros_like(image_to_draw[:, :, 0])
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            colored_mask = get_coloured_mask(mask)
            image_to_draw = cv2.addWeighted(image_to_draw, 1, colored_mask, 0.5, 0)

        # Add the frame number to the top left corner (white)
        #cv2.putText(image_to_draw, f"{i + 1}", (10, 30), font, 1, (255, 255, 255), 3)
        cv2.putText(image_to_draw, f"{i}", (10, 30), font, 1, (255, 255, 255), 3)
        
        trace_image.append(image_to_draw)

    # Read tracking data from res_track.txt
    lines = [line.strip('\n') for line in data]
    for line in lines:
        line = line.split()
        number = int(line[0])
        start = int(line[1])
        end = int(line[2])
        parent_number = int(line[3])

        # Process for cells that have tracks (multiple frames)
        if start != end:
            center = get_center(start, number, track_path)
            if center:
                cv2.circle(trace_image[start], center, 3, (0, 0, 255), -1)  # Red dot at the start
            start_point = center

            # Connect tracked centroids across frames (red lines)
            for i in range(start + 1, end + 1):
                center = get_center(i, number, track_path)
                if center:
                    cv2.circle(trace_image[i], center, 3, (0, 0, 255), -1)  # Red dot at each point
                    for j in range(start, i):
                        cv2.line(trace_image[j], start_point, center, (0, 0, 255), 1)  # Red line for all tracks
                    start_point = center

    # Save traced images
    for i in range(len(trace_image)):
        cv2.imwrite(os.path.join(trace_path, test_image[i]), trace_image[i])

# Function to create video from traced images
def get_video(trace_path):
    directory = trace_path
    pictures = sorted([name for name in os.listdir(directory) if "trace" not in name])
    print(pictures)
    if not pictures:
        print("No images found to generate video.")
        return

    fps = 1  # Frames per second
    image = cv2.imread(os.path.join(directory, pictures[0]), -1)
    size = (image.shape[1], image.shape[0])

    videowriter = cv2.VideoWriter(os.path.join(trace_path, "trace.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, name in enumerate(pictures):
        img = cv2.imread(os.path.join(directory, name), -1)
        cv2.putText(img, str(i), (10, 30), font, 1, (255, 255, 255), 1)
        videowriter.write(img)

    videowriter.release()
    print("Video generation completed.")

# Helper function to create a folder if it doesn't exist
def createFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"{path} has been created.")
    else:
        print(f"{path} already exists.")

# Main execution
if __name__ == "__main__":
    # Example directories (modify these paths as needed)
    print("Generating trace")
    test_folders = os.listdir("nuclear_dataset")
    test_folders = [os.path.join("nuclear_dataset/", folder) for folder in test_folders]
    test_folders.sort()

    for folder in test_folders:
        test_path = os.path.join(folder, "test")
        track_result_path = os.path.join(folder, "track_result")
        trace_path = os.path.join(folder, "trace1")
        createFolder(trace_path)

        # Ensure trace images are generated before creating the video
        get_trace(test_path, track_result_path, trace_path)
        get_video(trace_path)

    print("Generating trace")

    print("Processing completed.")
