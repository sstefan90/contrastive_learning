#!/bin/bash
python contrastive_learning.py --lr=0.0001 --epoch=20 --log_step=100
python contrastive_learning.py --lr=0.001 --epoch=15 --log_step=100
python contrastive_learning.py --lr=0.01 --epoch=15 --log_step=100
python contrastive_learning.py --lr.001 --epoch=15 --log_step=100 --model_type=resnet_34
python model.py --lr=0.0001 --epoch=10 --log_step=150
python model.py --lr=0.001 --epoch=5 --log_step=150
python model.py --lr=0.001 --epoch=5 --weights=True --log_step=150
python model.py --lr=0.01 --epoch=5 --log_step=150