*** 


**Data Preprocessing and Skin Tone Classification**

Skin tone was classified by matching the training dataset to the closest Monk Skin Tone Scale bucket. 
Skin Legions were isolated using edge detection.



**Contrastive Learning Embedding Generation**

SimCLR was trained using augmented images from the ISIC dataset. Augmentations included:
* Random Crops
* Random Rotations
* Random Resizing
* Color Jitter
* Random GreyScale
* Gaussian Blur

SimCLR model implementation with Noise Contrastive Estimation Loss `constrastive_learning.py`. Vector embedding generation and storage `extract_vectors.py`

**DenseNet201 Fine-Tuning**

DenseNet201 Fine-Tuning with and without SimCLR embeddigns `model.py`

**Compute**

GPU-Enabled machines from Azure were rented for model training and evaluation

**Dataset Citations**
Tschandl P., Rosendahl C. Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi.10.1038/sdata.2018.161 (2018)
