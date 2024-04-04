import numpy as np
import torch.nn as nn
import torch
from torch.utils import tensorboard
from torchvision.models import resnet18, resnet34, resnet50
from data_processing import get_training_data
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import utils
import argparse
import os
import copy

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import warnings
warnings.filterwarnings("ignore")

class ResNetBaseModel(nn.Module):

    def __init__(self, model_type, emb_size):
        super(ResNetBaseModel, self).__init__()
        self.resnets = {"resnet_18": resnet18(pretrained=False, num_classes=3*emb_size),
                        "resnet_34": resnet34(pretrained=False, num_classes=3*emb_size),
                        "resnet_50": resnet50(pretrained=False, num_classes=3*emb_size)}
        try:
            self.model = self.resnets[model_type]
        except Exception as e:
            print(f"model type {model_type} not in resnet dict! {self.resnets.keys()}")
            raise
        #input_size = self.model.fc.in_features
        self.model.fc = nn.Sequential(self.model.fc, nn.ReLU(), nn.Linear(3*emb_size, emb_size) )

    def forward(self, x):
        return self.model(x)

class SimCLR:
    def __init__(self, args):
        
        #self.writer = tensorboard.SummaryWriter(log_dir='logging')
        self.batch_size = args.batch_size
        self.log_step = args.log_step
        self.n_views = 2
        self.emb_size = args.emb_size
        self.model_type = args.model_type
        self.temperature = 0.07
        self.model = ResNetBaseModel(model_type=args.model_type, emb_size=args.emb_size)
        self.lr = args.lr
        #TODO: add weight decay later
        #self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.epochs = args.epoch
        self.device = args.device
        self.model_checkpoint_dir = args.model_checkpoint_dir
        logging_path = f'logging/model_type:{self.model_type}_lr:{self.lr}_batch_size:{self.batch_size}_emb_size:{self.emb_size}'
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
        model_path = f'{log_dir}/model_type:{self.model_type}_lr:{self.lr}_batch_size:{self.batch_size}_emb_size:{self.emb_size}'
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        torch.save(checkpoint, model_path + f'/epoch:{epoch}_step:{step}.pth')

    def info_nce_loss(self, output_vec):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        output_vec = F.normalize(output_vec, dim=1).to(self.device)

        similarity_matrix = torch.matmul(output_vec, output_vec.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1).to(self.device)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
        

    def train(self):
        writer = self.writer
        model = copy.deepcopy(self.model)
        optimizer = torch.optim.AdamW(model.parameters(), self.lr, weight_decay=1e-3)
        objective = torch.nn.CrossEntropyLoss().to(self.device)
        model.to(self.device)

        training_generator = get_training_data(batch_size=self.batch_size, mode='train_100')
        validation_generator = get_training_data(batch_size=self.batch_size, mode='validation')


        for epoch in range(self.epochs):
            for step, X_train in enumerate(training_generator):
                #concatenate the augmented images oon to 0 axis
                
                X_train = torch.cat(X_train, dim=0)
                X_train = X_train.to(self.device)
                output_vector = model(X_train)
                predictions, labels = self.info_nce_loss(output_vector)
                loss = objective(predictions, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            
                iteration = (epoch)*len(training_generator)+(step)

                if iteration % self.log_step == 0:
                    loss_items = []
                    accuracy_items = []

                    with torch.inference_mode():
                        for step, X_val in enumerate(validation_generator):
                            X_val = torch.cat(X_val, dim=0).to(self.device)
                            output_vector = model(X_val)
                            predictions, labels = self.info_nce_loss(output_vector)
                            loss = objective(predictions, labels)
                            accuracy = utils.top1_accuracy(prediction=predictions, labels=labels)
                            loss_items.append(loss.item())
                            accuracy_items.append(accuracy.item())
                    writer.add_scalar('loss', sum(loss_items)/len(loss_items), global_step=iteration)
                    writer.add_scalar('top1 accuracy', sum(accuracy_items)/len(accuracy_items), global_step=iteration)

                    print(f'epoch {epoch}, iteration {iteration}, accuracy {accuracy}, loss {loss}')
            

                    self.checkpoint_model(model, epoch, iteration, optimizer, log_dir=self.model_checkpoint_dir)
        self.checkpoint_model(model, epoch, iteration, optimizer, log_dir=self.model_checkpoint_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse user input arguments into function')
    parser.add_argument('--model_type', type=str, default='resnet_18', help='base model type')
    parser.add_argument('--batch_size',type=int, default=128, help='batch_size')
    parser.add_argument('--log_step', type=int, default=100, help='sample rate of recording')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='max number of epochs')
    parser.add_argument('--emb_size', type=int, default=256, help='size of resulting embedding')
    parser.add_argument('--device', type=str, default='cuda', help='device to train on, cpu or gpu')
    parser.add_argument('--model_checkpoint_dir', type=str, default='model_checkpoints', help='device to train on, cpu or gpu')
    args = parser.parse_args()
    simclr = SimCLR(args=args)
    simclr.train()





        
