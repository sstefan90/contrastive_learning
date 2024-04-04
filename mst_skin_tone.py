import numpy as np
import os
import sys
from PIL import Image
import math
import os
import time
import cv2
from itertools import product
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import random


class MST_loader:
    def __init__(self):
        self.dir = "MST_Swatches"
        self.MST_files = []
        for i in range(1, 11):
            self.MST_files.append(f"MST_{i}.png")

        self.MST_rgb = []

        for i in range(1, 11):
            img = Image.open(f"{self.dir}/{self.MST_files[i-1]}")
            np_img = np.asarray(img)
            #sample a single pixel and store it
            r = np_img[0,0,0]
            g = np_img[0,0,1]
            b = np_img[0,0,2]
            self.MST_rgb.append([r,g,b])
        
        self.label_path = "Data_Images/MST_labels.csv"
        self.num_tiles = 100
    
    @staticmethod
    def is_mostly_black(np_img):
        num_pixels = np_img.shape[0]*np_img.shape[1]
        #(f"num_pixels {num_pixels}")
        #sum R,G,B over 2nd axis
        np_img_collapsed = np.sum(np_img, axis=2)
        #print(f"shape of collapsed matrix:{np_img_collapsed.shape}")
        #count areas that are black (microscope)
        count = np.count_nonzero(np_img_collapsed<50)
        if count >= num_pixels // 1.2:
            return True
        return False

    @staticmethod
    def is_mostly_white(np_img):
        num_pixels = np_img.shape[0]*np_img.shape[1]
        count = np.count_nonzero(np_img>=215)
        if count >= num_pixels // 2:
            return True
        return False

    @staticmethod
    def read_image(img_file):
        im = Image.open(img_file)
        np_img = np.asarray(im)
        #print("shape of array np_img", np_img.shape)
        return np_img

    
    #split the image into 36 small squared
    def open_image_split(self, np_img):
        """
        takes a numpy_array and outputs an array of numpy matrixes
        """
        #print("open_image_split", np_img.shape)
        if len(np_img.shape) >= 3: 
            im = Image.fromarray(np_img.astype('uint8'), 'RGB')
        else:
            im = Image.fromarray(np_img.astype('uint8'), 'L')
        imgwidth, imgheight = im.size
        s = int(np.sqrt(self.num_tiles))
        height = round(imgheight / s)
        width = round(imgwidth / s)
        k = 0
        imgwidth 
        grid = product(range(0, imgwidth-imgwidth%width, width), range(0, imgheight-imgheight%width, height))
        list_of_crops = []
        for i, j in grid:
            box = (j, i, j+height, i+width)
            #save_path = out_path + f"{img_name}_{k}.jpg"
            crop = im.crop(box)
            list_of_crops.append(np.asarray(crop))
        return list_of_crops
            #k +=1

    @staticmethod
    def write_labels(img_name, label, file_name):
        with open(file_name, 'a') as f:
            f.write(f"{img_name}\t{label}\n")

    @staticmethod
    def color_difference(np_img, mst_patch):
        r_delta = np_img[:,:,0] - mst_patch[:,:,0]
        g_delta = np_img[:,:,1] - mst_patch[:,:,1]
        b_delta = np_img[:,:,2] - mst_patch[:,:,2]
        r_mean = (np_img[:,:,0] + mst_patch[:,:,0])*0.5
        r_diff = (2 + r_mean / 256)*r_delta**2
        g_diff = 4*g_delta**2
        b_diff = (2 + (255 - r_mean)/256)*b_delta**2
        diff_comined = (r_diff + g_diff + b_diff)**0.5
        #sum the difference
        total_difference = np.sum(diff_comined.reshape(-1), axis=0)
        return total_difference


    def match_mst(self, np_img):
        rows, cols = np_img.shape[0], np_img.shape[1]
        min_distance = float("inf")
        mst_bucket_chosen = 0
        for label, pixel in enumerate(self.MST_rgb):
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            #print(f"MST colors check: {r}, {g}, {b}, {label+1}")
            r_matrix = np.full((rows, cols), r, dtype=int)
            g_matrix = np.full((rows, cols), g, dtype=int)
            b_matrix = np.full((rows, cols), b, dtype=int)
            mst_patch = np.stack((r_matrix, g_matrix, b_matrix), axis=2)
            diff_norm = self.color_difference(np_img, mst_patch)
            if diff_norm < min_distance:
                mst_bucket_chosen = label
                min_distance = diff_norm
        return mst_bucket_chosen # 0 indexed!!

    @staticmethod
    # overwrite the image with white where any edge detector
    def edge_detection(input_path):
        img_name = input_path.split("/")[-1].split(".")[0]
        im = Image.open(input_path)
        crop = np.asarray(im)

        
        r = crop[:,:,0]
        g = crop[:,:,1]
        b = crop[:,:,2]
        img_greyscale = r * (299/1000) + g * (587/1000) + b * (114/1000)

        #make image black and white
        grey_img = 255 - img_greyscale
        grey_img[grey_img > 100] = 255
        grey_img[grey_img <= 100] = 0
        return grey_img


    def find_bucket(self, input_image):
        img_name = input_image.split("/")[-1]
        img_name  =img_name.split(".")[0]
        labels_path = self.label_path
        np_img = self.read_image(input_image)
        #print(np_img.shape)
        list_of_crops = self.open_image_split(np_img)
        grey_scale_img = self.edge_detection(input_image)
        print(img_name)
        plt.imshow(grey_scale_img, cmap='gray')
        plt.show()
        list_of_crops_greyscale = self.open_image_split(grey_scale_img)
        #print("list of crop greyscale", list_of_crops_greyscale[0].shape)
        bucket_labels = defaultdict(lambda: 0)
        #print(len(list_of_crops))
        
        #print(len(list_of_crops_greyscale))
        for crop, crop_greyscale in zip(list_of_crops, list_of_crops_greyscale):
            if self.is_mostly_black(crop):
                #print("black crop", crop)
                continue
            if self.is_mostly_white(crop_greyscale):
                continue
            label_guess = self.match_mst(crop)
            if label_guess != -1:
                bucket_labels[label_guess] +=1
        #print(bucket_labels)
        try:
            label_vote = max(bucket_labels, key=bucket_labels.get)
        except:
            label_vote = random.randint(0,9)
        #self.write_labels(img_name, label_vote, labels_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--restart', type=bool, default=False, help='')
    args = parser.parse_args()
    
    #filename = "/Users/stephone_christian/Downloads/ISIC_2019_Training_Input/ISIC_0073237.jpg"
    filename = "ISIC_0058514.jpg"
    filename = "ISIC_0073231.jpg"
    filename2 = "ISIC_0073251.jpg"
    filename3 = "ISIC_0000212.jpg"
    filename4 = "ISIC_0000211.jpg"
    files = [filename, filename2, filename3, filename4]
    mst_loader = MST_loader()
    if args.restart:
        file = open(mst_loader.label_path, 'w')
        file.close()

    dir_name = "Data_Images/ISIC_2019_Training_Input"
    dir_list = os.listdir(dir_name)
    dir_list = files

    for i, file in enumerate(dir_list):
        file_name = f"{dir_name}/{file}"
        if file[-3:] != 'jpg':
            continue
        if i % 50 == 0:
            print(f"{i+1} out of {len(dir_list)}")
            print(file_name)
        
        mst_loader.find_bucket(file_name)



    
