import torch
#from net import Unet
#from net import FCN8s
#from net import DeepSeg
from net import TransDeepSeg
#from net import StarDistUNet
#from net import Cellpose

from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from torch import nn
import shutil
from track import predict_dataset_2
from generate_trace import get_trace,get_video
from matplotlib import pyplot as plt

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img = clahe.apply(img)

    return img

def enhance(img):
    img=np.clip(img*1.2,0,255)
    img=img.astype(np.uint8)
    return img


def test(test_path,result_path):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_transforms = transforms.Compose([
        #transforms.Lambda(lambda img:clahe(img)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    #model=Unet(1,1)
    #model = FCN8s(1, 1)
    #model = DeepSeg(1, 1)
    model = TransDeepSeg(1, 1)
    #model = StarDistUNet(1, 1)
    #model = Cellpose(1, 1)

    model.eval()
    model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)
    model.load_state_dict(torch.load('checkpoints/CP_epoch50.pth'))
    test_data=TestDataset(test_path,transform=x_transforms)
    dataloader=DataLoader(test_data,batch_size=1)

    with torch.no_grad():
        for index,x in enumerate(dataloader):
            x=x.to(device)
            y=model(x)
            y=y.cpu()
            y=torch.squeeze(y)
            img_y=torch.sigmoid(y).numpy()
            img_y=(img_y*255).astype(np.uint8)
            cv2.imwrite(os.path.join(result_path,"predict_"+str(index).zfill(6)+'.tif'),img_y)
    print(test_path," prediction finish!")

def process_img():
    img_root = "data/test/"
    n = len(os.listdir(img_root))
    for i in range(n):
        img_path = os.path.join(img_root, str(i).zfill(6)+".tif")
        img = cv2.imread(img_path, -1)
        img = np.uint8(np.clip((0.02 * img + 60), 0, 255))
        #img = np.uint8(np.clip((0.04 * img + 50), 0, 255))
        '''
        min = np.unique(img)[0]
        max = np.unique(img)[-1]
        new_img = np.zeros((img.shape))

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[i][j] = (img[i][j] - min) * 255 / (max - min)
        new_img = new_img.astype(np.uint8)
        #cla = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        #new_img = cla.apply(new_img)
        new_img = np.uint8(np.clip((new_img*0.2 + 30), 0, 255))'''
        cv2.imwrite(img_path, img)

def processImg2():
    directory = "data/test"
    img_list = os.listdir(directory)
    imgs = []
    for img_name in img_list:
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path, -1)
        imgs.append(img)
    whole = imgs[0]

    for i in range(1, len(imgs)):
        whole = np.hstack((whole, imgs[i]))
    whole = clahe(whole)
    for i, img_name in enumerate(img_list):
        img_path = os.path.join(directory, img_name)
        img = whole[:, 770 * i:770 * (i + 1)]
        cv2.imwrite(img_path, img)


def add_blur(img):
    new_img = cv2.GaussianBlur(img, (21, 21),0)
    new_img = cv2.GaussianBlur(new_img, (5, 5), 0)

    #new_img = cv2.addWeighted(img, 0.2, img_mean, 0.8, 0)
    return new_img


def process_predictResult(source_path,result_path):
    if not os.path.isdir(result_path):
        print('creating RES directory')
        os.mkdir(result_path)

    names = os.listdir(source_path)
    names = [name for name in names if '.tif' in name]
    names.sort()

    for name in names:

        predict_result=cv2.imread(os.path.join(source_path,name),-1)
        ret,predict_result=cv2.threshold(predict_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ''' 
        ret, predict_result = cv2.threshold(predict_result, 20, 255, cv2.THRESH_BINARY)
        kernel_size=6
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        predict_result=cv2.morphologyEx(predict_result,cv2.MORPH_OPEN,kernel)
        '''
        ret, markers = cv2.connectedComponents(predict_result)
        cv2.imwrite(os.path.join(result_path,name),markers)

def useAreaFilter(img, area_size):
    # Convert the image to 8-bit single-channel (CV_8UC1)
    img_8bit = cv2.convertScaleAbs(img)
    
    # Find contours on the 8-bit image
    contours, hierarchy = cv2.findContours(img_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_new = np.stack((img_8bit, img_8bit, img_8bit), axis=2)
    for cont in contours:
        area = cv2.contourArea(cont)
        if area < area_size:
            img_new = cv2.fillConvexPoly(img_new, cont, (0, 0, 0))
    
    img = img_new[:, :, 0]
    return img


def delete_file(path):
    if not os.path.isdir(path):
        print(path," does not exist!")
        os.mkdir(path)
        return
    file_list=os.listdir(path)
    for file in file_list:
        file_path=os.path.join(path,file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print(path," has been cleaned!")

def createFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(path," has been created.")
    else:
        print(path," has already existed.")


if __name__=="__main__":
    test_folders=os.listdir("nuclear_dataset")
    test_folders=[os.path.join("nuclear_dataset/",folder) for folder in test_folders]
    test_folders.sort()
    for folder in test_folders:
        test_path=os.path.join(folder,"test")
        test_result_path=os.path.join(folder,"test_result")
        res_path=os.path.join(folder, "res")
        res_result_path=os.path.join(folder, "res_result")
        track_result_path=os.path.join(folder, "track_result")
        trace_path=os.path.join(folder,"trace")
        createFolder(test_result_path)
        createFolder(res_path)
        createFolder(res_result_path)
        createFolder(track_result_path)
        createFolder(trace_path)


        test(test_path,test_result_path)
        process_predictResult(test_result_path,res_path)

        result=os.listdir(res_path)
        for picture in result:
            image=cv2.imread(os.path.join(res_path,picture),-1)
            image=useAreaFilter(image,100)
            cv2.imwrite(os.path.join(res_result_path,picture),image)
        print("starting tracking")
        #track
        predict_result=res_result_path
        predict_dataset_2(predict_result,track_result_path)

        #get_trace(test_path,track_result_path,trace_path)





