import numpy as np
from torchvision.transforms import v2
from cv2 import blur
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import sys

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class ContrastiveTransformations(object):

    def __init__(self, mode='train', n_views=2):
        if mode == 'train':
            base_transformations = v2.Compose([
                v2.PILToTensor(),
                v2.Resize(size=(250,250), antialias=True),
                v2.RandomApply([v2.ColorJitter(brightness=[.7, 1.3],contrast=[.9, 1.1],saturation=0.1,hue=(-0.1, 0.1))], p=.85),
                transforms.RandomGrayscale(p=0.1),
                v2.RandomResizedCrop(size=(224,224), antialias=True),
                v2.RandomHorizontalFlip(p=0.75),
                #v2.RandomRotation(359),
                v2.GaussianBlur(3, sigma=(0.1, 1.5))
            ])
        if mode == 'eval':
            base_transformations = v2.Compose([
                v2.PILToTensor(),
                v2.Resize(size=(224,224)),
            ])
            self.n_views = 1

        self.base_transforms = base_transformations
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class ISICDataSet(Dataset):
    def __init__(self, mode='train_100', eval=False):
        self.mode = mode
        self.eval = eval

        self.train_data_file = 'Data_Images/ISIC_2019_Training_Input'
        if mode == 'train_100':
            self.train_data_labels_file = 'Data_Images/training_labels_100.csv'
        if mode == 'train_5':
            self.train_data_labels_file = 'Data_Images/training_labels_5.csv'
        if mode == 'train_1':
            self.train_data_labels_file = 'Data_Images/training_labels_1.csv'
        if mode == 'validation':
            self.train_data_labels_file = 'Data_Images/validation_labels_100.csv'
        if mode == 'test':
            self.train_data_labels_file = 'Data_Images/test_labels_100.csv'

        self.training_data_labels_df = pd.read_csv(self.train_data_labels_file, sep=',', header=None, skiprows=1)
        self.training_data_labels_df.drop(self.training_data_labels_df.columns[len(self.training_data_labels_df.columns)-1], axis=1, inplace=True)
        self.size = (224,224)
        self.num_labels = self.training_data_labels_df.shape[1]
        if 'train' in mode:
            toggle = 'train'
        else:
            toggle = 'eval'
        self.pair_maker = ContrastiveTransformations(mode=toggle)
    def __len__(self):
        return len(self.training_data_labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.train_data_file,
                                self.training_data_labels_df.iloc[idx, 0] + ".jpg")
        X = Image.open(img_name)
        if not self.eval:
            X_1, X_2 = self.pair_maker(X)
            X_1 = X_1 / 255.0
            X_2 = X_2 / 255.0
            
            return X_1, X_2
        if self.eval:
            X = Image.open(img_name)
            X_1 = self.pair_maker(X)
            X = X_1[0] / 255.0
            label = self.training_data_labels_df.iloc[idx, 1:]
            img_name = self.training_data_labels_df.iloc[idx,0]
            y = torch.tensor(label.tolist())
            return X, y, img_name

def test_format():
    fn = lambda: datasets.STL10('Test_Data', split='unlabeled',transform=ContrastiveTransformations(), download=True)
    dl = DataLoader(dataset=fn(), batch_size=16, shuffle=True, drop_last=True)
    for x, _ in dl:
        print(type(x))
        print(len(x))
        print(x[0].shape)
        print(x[1].shape)
        break
    return dl



def create_train_test_val():

    def manual_stratify(df, desired_percent):
        desired = desired_percent / 100
        frac = desired / .70
        df_train_labels_5 = pd.DataFrame(columns=['image','MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC' ])
        for key in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']:
            df_sample = df[(df[key] == 1.0)].sample(frac=frac, replace=False, random_state=42)
            df_train_labels_5 = pd.concat([df_train_labels_5, df_sample])
        return df_train_labels_5


    training = 'Data_Images/training_labels_100.csv'
    validation = 'Data_Images/validation_labels_100.csv'
    test = 'Data_Images/test_labels_100.csv'

    training_1 = 'Data_Images/training_labels_1.csv'
    training_5 = 'Data_Images/training_labels_5.csv'

    df_all_labels = pd.read_csv('Data_Images/ISIC_2019_Training_GroundTruth.csv', sep=',')
    #print(df_all_labels.columns)
    #df_all_labels.drop(columns='UNK', inplace=True)

    df_train_labels, df_test_labels =  train_test_split(df_all_labels, stratify=df_all_labels[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']], test_size=0.30, random_state=42)
    df_val_labels, df_test_labels = train_test_split(df_test_labels, stratify=df_test_labels[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']], test_size=0.5, random_state=42)

    #print(df_train_labels.head(5))
    df_train_labels_5 = pd.DataFrame(columns=['image','MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC' ])
    df_train_labels_1 = pd.DataFrame(columns=['image','MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC' ])
    

    df_train_labels_5 = manual_stratify(df_all_labels, desired_percent=5)
    df_train_labels_1 = manual_stratify(df_all_labels, desired_percent=1)

    if not os.path.exists(training):
        df_train_labels.to_csv(training, index=False)
    
    if not os.path.exists(validation):
        df_val_labels.to_csv(validation, index=False)

    if not os.path.exists(test):
        df_test_labels.to_csv(test, index=False)

    if not os.path.exists(training_1):
        df_train_labels_1.to_csv(training_1, index=False)

    if not os.path.exists(training_5):
        df_train_labels_5.to_csv(training_5, index=False)
        


    

    
        

def visualize_transformations(data):
    NUM_IMAGES = 4
    data_img = []
    for i in range(NUM_IMAGES):
        print(data[0][i,:,:,:].shape)
        print(data[1][i, :, :, :].shape)
        data_img.append((data[0][i,:,:,:].view(1, 3, 224,224)*255.0).to(torch.uint8))
        data_img.append((data[1][i,:,:,:].view(1, 3, 224,224)*255.0).to(torch.uint8))
    imgs = torch.stack(data_img, dim=0).squeeze(1)
    print(imgs.shape)
    img_grid = torchvision.utils.make_grid(imgs, nrow=NUM_IMAGES, pad_value=0.9)
    print(img_grid.shape)
    img_grid = img_grid.permute(1, 2, 0)
    plt.figure(figsize=(10,5))
    plt.title('Augmented image examples of the ISIC dataset')
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()

def get_training_data(batch_size, mode, eval=False):
    test_data = ISICDataSet(mode=mode, eval=eval)
    train_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    return train_dataloader


if __name__ == '__main__':
    
    test_data = ISICDataSet(mode='train_1', eval=False)
    train_dataloader = DataLoader(dataset=test_data, batch_size=16, shuffle=True, drop_last=True)
    for data in train_dataloader:
        visualize_transformations(data)
        break
    '''
    validation_dataloader = get_training_data(batch_size=16, mode='train_1', eval=True)
    step= 0
    for  X, y, img_name in validation_dataloader:
        #print(X)
        print(type(y))
        print(y)
        print(img_name)
        step +=1
        if step == 2:
            break
    '''
    #df = pd.read_csv('Data_Images/ISIC_2019_Training_GroundTruth.csv', sep=',', header=None, skiprows=1)
    #print(df.head(5))
    #df = pd.read_csv('Data_Images/ISIC_2019_Training_GroundTruth.csv', sep=',', header=None)
    #print(df.head(5))
    #create_train_test_val()

