#fine tune densenet201 on skin dermatology examples
import torch.nn as nn
import numpy as np
import torch
from torch.utils import tensorboard
from torchvision.models import densenet201
from data_processing import get_training_data
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import utils
import argparse
import os
import copy
from extract_vectors import load_model
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
class DenseNet201Modified(nn.Module):
    def __init__(self, emb_size):
        super(DenseNet201Modified, self).__init__()
        self.model = densenet201(pretrained=True)
        
        
        out_features = self.model.classifier.out_features
        self.fine_tune_layer = nn.Sequential(nn.Linear(out_features + emb_size, 1000), nn.Linear(1000, 256), nn.Linear(256, 8))
        self.fine_tune_layer.apply(self.initialize_weights)
    def forward(self, img, vec):
        x = self.model(img)
        x1 = torch.cat([x, vec], dim=1)
        out = self.fine_tune_layer(x1)
        return out
    
    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class FineTuning:
    def __init__(self, args):
        self.lr = args.lr
        self.log_step = args.log_step
        self.epoch = args.epoch
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size
        self.device = args.device
        self.model_path = args.model_path
        self.learning_rate = args.lr
        self.training_path = args.training_path
        self.emb_size = self.emb_size
        self.weights = args.weights
        self.model = DenseNet201Modified(emb_size=self.emb_size)
        self.model_checkpoint_dir = args.model_checkpoint_dir
        logging_path = f'logging/model_type:DENSENET201_lr:{self.lr}_batch_size:{self.batch_size}_emb_size:{self.emb_size}'
        self.writer = tensorboard.SummaryWriter(log_dir=logging_path)
        if not os.path.exists(logging_path):
            os.mkdir(logging_path)


    def checkpoint_model(self, model, epoch, step, optimizer, log_dir):
        checkpoint = {
        'epoch': epoch,
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
        model_path = f'{log_dir}/model_type:DENSENET201_lr:{self.lr}_batch_size:{self.batch_size}_emb_size:{self.emb_size}'
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        torch.save(checkpoint, model_path + f'/epoch:{epoch}_step:{step}.pth')


    def train_model(self):
        writer = self.writer
        model = copy.deepcopy(self.model)
        model.to(self.device)
        feature_generator_model = load_model(self.model_path)
        feature_generator_model.to(self.device)
        weights = torch.Tensor([0.0567, 0.0336,  0.066, 0.129, 0.074, 0.247, 0.239,0.152])

        objective = torch.nn.CrossEntropyLoss().to(self.device)
        layers_to_train = []
        for name, param in model.named_parameters():
            if 'fine_tune_layer' in name:
                layers_to_train.append(param)
        optimizer = torch.optim.AdamW(layers_to_train, self.learning_rate, weight_decay=1e-3)

        training_generator = get_training_data(batch_size=self.batch_size, mode=self.training_path, eval=True)
        validation_generator = get_training_data(batch_size=self.batch_size, mode='validation', eval=True)

        for epoch in range(self.epoch):
            for step, data in enumerate(training_generator):
                #concatenate the augmented images oon to 0 axis
                X_train, y, _ = data
                #print('X_train.shape', X_train.shape)
                #print('y.shape', y.shape)
                X_train = X_train.to(self.device)
                y = y.to(self.device)
                with torch.inference_mode():
                    feature_vector = feature_generator_model(X_train)
                #print('feature_vector', feature_vector.shape)
                predictions = model(X_train, feature_vector)
                #print('predictions.shape', predictions.shape)
                if self.weights:
                    loss = objective()
                    loss = objective(predictions, y, reduce='none')
                    loss = loss * weights
                    loss = loss.mean()
                else:
                    loss = objective(predictions,y )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            
                iteration = (epoch)*len(training_generator)+(step)

                if iteration % self.log_step == 0:
                    loss_items = []

                    with torch.inference_mode():
                        num_correct = 0
                        num_examples = 0
                        for step, data in enumerate(validation_generator):
                            X_val, y, _ = data

                            X_val = X_val.to(self.device)
                            y = y.to(self.device)
                            feature_vector = feature_generator_model(X_val)
                            predictions = model(X_val, feature_vector)
                            loss = objective(predictions, y)
                            num_correct += utils.num_correct(prediction=predictions, labels=y)
                            num_examples+= X_val.shape[0]
                            loss_items.append(loss.item())
                    accuracy = num_correct/num_examples
                    writer.add_scalar('loss', sum(loss_items)/len(loss_items), global_step=iteration)
                    writer.add_scalar('top1 accuracy', accuracy, global_step=iteration)

                    print(f'epoch {epoch}, iteration {iteration}, accuracy {accuracy}, loss {loss}')
                    self.checkpoint_model(model, epoch, iteration, optimizer, log_dir=self.model_checkpoint_dir)
        self.checkpoint_model(model, epoch, iteration, optimizer, log_dir=self.model_checkpoint_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse user input arguments into function')
    parser.add_argument('--model_type', type=str, default='resnet_18', help='base model type')
    parser.add_argument('--batch_size',type=int, default=32, help='batch_size')
    parser.add_argument('--log_step', type=int, default=100, help='sample rate of recording')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=5, help='max number of epochs')
    parser.add_argument('--weights', type=bool, default=False, help='weight classes by frequency')
    parser.add_argument('--emb_size', type=int, default=256, help='size of resulting embedding')
    parser.add_argument('--device', type=str, default='cuda', help='device to train on, cpu or gpu')
    parser.add_argument('--training_path', type=str, default='train_100', help='train_100, train_1, train_5')
    parser.add_argument('--model_path', type=str, default='model_typ:resnet_18_lr:0.001_batch_size:128_emb_size:256/epoch:7_step:1000.pth', help='feature_generator_model')
    parser.add_argument('--model_checkpoint_dir', type=str, default='model_checkpoints', help='device to train on, cpu or gpu')
    args = parser.parse_args()
    fine_tune = FineTuning(args=args)
    fine_tune.train_model()
