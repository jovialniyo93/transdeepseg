import torch

from model import TransformerDeepSeg
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import numpy as np
from torch import nn
import shutil
from track import track
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

class TestDataset(Dataset):
    def __init__(self, img_root, transform=None, model_input_size=(576, 576)):
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform=transform
        self.model_input_size = model_input_size

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = cv2.imread(x_path,-1)
        
        # Store original dimensions
        orig_h, orig_w = img_x.shape[:2]
        
        # Normalize image to 0-255 range
        img_x = img_x.astype(np.float32)
        img_x = (255 * ((img_x - img_x.min()) / (np.ptp(img_x) + 1e-6))).astype(np.uint8)
        
        # Resize to model input size
        img_x = cv2.resize(img_x, self.model_input_size)
        
        if self.transform is not None:
            img_x=self.transform(img_x)
        return img_x, orig_h, orig_w

    def __len__(self):
        return len(self.imgs)

def test_dataset(img_root):
    imgs=[]
    files = sorted([f for f in os.listdir(img_root) if f.endswith('.tif')])
    for i, file in enumerate(files):
        img = os.path.join(img_root, file)
        imgs.append(img)
    return imgs

def test(test_path, result_path, model_input_size=(576, 576)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    model = TransformerDeepSeg(n_channels=1, n_classes=2, bilinear=True, img_size=model_input_size[0])
    model.eval()
    model = model.to(device)

    checkpoint_path = 'tmp/segmentation.pth'
    if not os.path.exists(checkpoint_path):
        print("Warning: Checkpoint not found. Please ensure the model is trained.")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    test_data = TestDataset(test_path, transform=x_transforms, model_input_size=model_input_size)
    dataloader = DataLoader(test_data, batch_size=1, num_workers=2)

    with torch.no_grad():
        for index, (x, orig_h, orig_w) in enumerate(dataloader):
            x = x.to(device)
            orig_h, orig_w = orig_h.item(), orig_w.item()
            
            logits, edges = model(x)
            
            pred = torch.argmax(logits, dim=1)
            pred = pred.cpu().squeeze().numpy()
            
            img_y = (pred == 1).astype(np.uint8) * 255
            
            # Resize prediction back to original image size
            img_y = cv2.resize(img_y, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Save as uncompressed TIFF to avoid imagecodecs dependency
            cv2.imwrite(os.path.join(result_path, "predict_" + str(index).zfill(6) + '.tif'), 
                       img_y, [cv2.IMWRITE_TIFF_COMPRESSION, 1])  # 1 = no compression
    
    print(test_path, "prediction finish!")

def process_img():
    img_root = "data/test/"
    n = len(os.listdir(img_root))
    for i in range(n):
        img_path = os.path.join(img_root, str(i).zfill(6) + ".tif")
        img = cv2.imread(img_path, -1)
        img = np.uint8(np.clip((0.02 * img + 60), 0, 255))
        # Save as uncompressed TIFF
        cv2.imwrite(img_path, img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

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
        
        # Save as uncompressed TIFF
        cv2.imwrite(os.path.join(result_path, name), markers, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

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
    MODEL_INPUT_SIZE = (576, 576)
    
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

        test(test_path, test_result_path, model_input_size=MODEL_INPUT_SIZE)
        process_predictResult(test_result_path, res_path)

        result = os.listdir(res_path)
        for picture in result:
            image = cv2.imread(os.path.join(res_path, picture), -1)
            image = useAreaFilter(image, 100)
            # Save as uncompressed TIFF
            cv2.imwrite(os.path.join(res_result_path, picture), image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        
        print("Starting tracking")
        # Track
        predict_result = res_result_path
        track(predict_result, track_result_path)

        get_trace(test_path, track_result_path, trace_path)
        get_video(trace_path)