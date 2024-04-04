***Contrastive Learning for Skin Tone Aware Cancer Classification***

Model performance was evaluated and stratified by skin tone of test examples. The International Skin Imaging Collaboration is an open-source dataset brought about with hopes of improving skin cancer classification through the joint effort of researchers in the community. While there are many high-quality microscope training data images, there are very few training data examples with darker skin tones. This project aims to see whether augmenting the data with meta-learning techniques, such as embeddings generated through contrastive learning, helps to improve classification precision and accuracy on darker skin tone examples. Meta-Learning techniques are known to improve model classification performance in low data regimes. 

**Data Preprocessing and Skin Tone Classification**

Skin tone was classified by matching the training dataset to the closest Monk Skin Tone Scale bucket. 
Skin Lesions were isolated using edge detection `mst_skin_tone.py`


<img width="200" alt="Screen Shot 2024-04-04 at 1 48 09 PM" src="https://github.com/sstefan90/contrastive_learning/assets/22806151/b7d1e21e-79f5-4fcc-9ec5-2c3f7ad98c0c">

**Contrastive Learning Embedding Generation**

SimCLR was trained using augmented images from the ISIC dataset. Augmentations included:
* Random Crops
* Random Rotations
* Random Resizing
* Color Jitter
* Random GreyScale
* Gaussian Blur

SimCLR model implementation with Noise Contrastive Estimation Loss `constrastive_learning.py`. Vector embedding generation and storage `extract_vectors.py`

<img width="275" alt="Screen Shot 2023-12-03 at 6 21 35 PM" src="https://github.com/sstefan90/contrastive_learning/assets/22806151/7fdbb854-9dc6-4cea-8635-58f9bc2e12fe">


**DenseNet201 Fine-Tuning**

DenseNet201 Fine-Tuning with and without SimCLR embeddings `model.py`

**Compute**

GPU-Enabled machines from Azure were rented for model training and evaluation

**Dataset Citations**
Tschandl P., Rosendahl C. Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi.10.1038/sdata.2018.161 (2018)
