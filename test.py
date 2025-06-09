import torch
#from model import DeepSeg
#from model import StarDistModel
#from model import Unet
#from model import FCN8s
#from model import Cellpose
#from model import UnetSegmentation
from model import TransformerDeepSeg
from data_utils import TestDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from torch import nn
import shutil
from track import predict_dataset_2
from generate_trace import get_trace, get_video
from matplotlib import pyplot as plt
import os
import cv2

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img = clahe.apply(img)
    return img

def enhance(img):
    img = np.clip(img * 1.2, 0, 255)
    img = img.astype(np.uint8)
    return img

def test(test_path, result_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    #model = DeepSeg(n_channels=1, n_classes=2, bilinear=True)
    #model = Unet(n_channels=1, n_classes=2)
    #model = StarDistModel(n_channels=1, n_classes=2, n_rays=32, bilinear=True)
    #model = FCN8s(n_channels=1, n_classes=2)
    #model = Cellpose(n_channels=1, n_classes=2)
    #model = UnetSegmentation(n_channels=1, n_classes=2)
    model = TransformerDeepSeg(n_channels=1, n_classes=2, bilinear=True)
    model.eval()
    model = model.to(device)

    model.load_state_dict(torch.load('tmp/segmentation.pth', map_location=device))
    
    test_data = TestDataset(test_path, transform=x_transforms)
    dataloader = DataLoader(test_data, batch_size=1, num_workers=2)

    with torch.no_grad():
        for index, (x, orig_h, orig_w, top, left) in enumerate(dataloader):
            x = x.to(device)
            # Convert to integers (batch_size=1, so take first element)
            orig_h, orig_w, top, left = orig_h.item(), orig_w.item(), top.item(), left.item()
            
            # Model returns (logits, edges) - use logits for segmentation
            logits, edges = model(x)
            
            # Get segmentation prediction
            pred = torch.argmax(logits, dim=1)  # Get class predictions
            pred = pred.cpu().squeeze().numpy()
            
            # Convert to binary mask (class 1 = foreground)
            img_y = (pred == 1).astype(np.uint8) * 255
            
            # Crop back to original size
            img_y = img_y[top:top+orig_h, left:left+orig_w]
            
            cv2.imwrite(os.path.join(result_path, "predict_" + str(index).zfill(6) + '.tif'), img_y)
    print(test_path, "prediction finish!")

def process_img():
    img_root = "data/test/"
    n = len(os.listdir(img_root))
    for i in range(n):
        img_path = os.path.join(img_root, str(i).zfill(6) + ".tif")
        img = cv2.imread(img_path, -1)
        img = np.uint8(np.clip((0.02 * img + 60), 0, 255))
        cv2.imwrite(img_path, img)

def process_predictResult(source_path, result_path):
    if not os.path.isdir(result_path):
        print('Creating RES directory')
        os.mkdir(result_path)

    names = os.listdir(source_path)
    names = [name for name in names if '.tif' in name]
    names.sort()

    for name in names:
        predict_result = cv2.imread(os.path.join(source_path, name), -1)
        # Binarize image
        ret, predict_result = cv2.threshold(predict_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connected components and labeling
        ret, markers = cv2.connectedComponents(predict_result)
        
        # Convert markers to uint16 for saving as .tif
        markers = np.uint16(markers)
        
        cv2.imwrite(os.path.join(result_path, name), markers)

def useAreaFilter(img, area_size):
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_new = np.stack((img, img, img), axis=2)

    for cont in contours:
        area = cv2.contourArea(cont)
        if area < area_size:
            img_new = cv2.fillConvexPoly(img_new, cont, (0, 0, 0))

    img = img_new[:, :, 0]
    return img

def delete_file(path):
    if not os.path.isdir(path):
        print(path, "does not exist!")
        os.mkdir(path)
        return
    file_list = os.listdir(path)
    for file in file_list:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print(path, "has been cleaned!")

def createFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(path, "has been created.")
    else:
        print(path, "has already existed.")

if __name__ == "__main__":
    test_folders = os.listdir("nuclear_dataset")
    test_folders = [os.path.join("nuclear_dataset/", folder) for folder in test_folders]
    test_folders.sort()
    
    for folder in test_folders:
        test_path = os.path.join(folder, "test")
        test_result_path = os.path.join(folder, "test_result")
        res_path = os.path.join(folder, "res")
        res_result_path = os.path.join(folder, "res_result")
        track_result_path = os.path.join(folder, "track_result")
        trace_path = os.path.join(folder, "trace")

        createFolder(test_result_path)
        createFolder(res_path)
        createFolder(res_result_path)
        createFolder(track_result_path)
        createFolder(trace_path)

        test(test_path, test_result_path)
        process_predictResult(test_result_path, res_path)

        result = os.listdir(res_path)
        for picture in result:
            image = cv2.imread(os.path.join(res_path, picture), -1)
            image = useAreaFilter(image, 100)
            cv2.imwrite(os.path.join(res_result_path, picture), image)
        
        print("Starting tracking")
        # Track
        predict_result = res_result_path
        predict_dataset_2(predict_result, track_result_path)

        get_trace(test_path, track_result_path, trace_path)
        get_video(trace_path)
