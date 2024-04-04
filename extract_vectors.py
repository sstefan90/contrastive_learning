import torch
import torch.nn as nn
from data_processing import get_training_data
from contrastive_learning import ResNetBaseModel
import sys
import pandas as pd

def load_model(path):
    model = ResNetBaseModel(model_type='resnet_18', emb_size=256)
    checkpoint = torch.load(f'model_checkpoints/{path}')
    model.load_state_dict(checkpoint['model'])
    model.fc = nn.Identity()
    #print(model)
    return model

def append_to_file(vector, img_name, filename):
    #print(vector)
    #print(img_name)
    df_vector = pd.DataFrame(vector.numpy())
    df_filenames = pd.DataFrame(data=img_name)
    df_merge = df_filenames.merge(df_vector, how='outer', left_index=True, right_index=True)
    #print(df_merge)
    assert not df_merge.isnull().values.any(), "something is wrong! There's nan"
    df_merge.to_csv(filename, mode='a', header=False, index=False)



def extract_vectors(model):
    filename = 'Data_Images/vectors.csv'
    file = open(filename, 'w')
    file.close()
    batch_size = 256
    dataloader = get_training_data(batch_size=256, mode='train_100', eval=True)
    with torch.inference_mode():
        step = 0
        for X, y, img_name in dataloader:
            vector = model(X)
            append_to_file(vector, img_name, filename)
            step +=1
            if (step % 10) == 0:
                print("on example", step*batch_size, 'out of 25k')
                sys.exit()

if __name__ == "__main__":
    model = load_model(path='model_typ:resnet_18_lr:0.001_batch_size:128_emb_size:256/epoch:3_step:550.pth')
    extract_vectors(model)