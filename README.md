# DSTD-Net:A Dual-Stream Transformer with Diff-attention for Multisource Remote Sensing Classification
Code for "A Dual-Stream Transformer with Diff-attention for Multisource Remote Sensing Classification".<br>
Lin Xu, Hao Zhu, Licheng Jiao, Wenhao Zhao, Xiaotong Li, Biao Hou, Zhongle Ren, and Wenping Ma<br>
Xidian University<br>
>To minimize the feature redundancy of multisource and maximize the complementary advantages of multisource, a Dual-Stream Transformer with Diff-attention (DSTD)-Net is proposed for multisource remote sensing classification in this paper. Firstly, in terms of feature extraction, we use a Self-attention and Co-attention (SCA) block to extract both specific advantageous features and common essential features. Based on that, a self-attention module strengthened by diff-attention (SSDA) that pays attention to the difference between two specific advantageous features is designed to reduce the essential redundancy in specific features. It can take advantage of the difference between two specific features and reduce the essential redundancy of the specific advantageous features, making them purer and better for classification. Finally, since the specific features and common features of MS and PAN images make different contributions to classification, a Multi-stage Gated Fusion (MGF) strategy is used. The MGF strategy mainly uses Gated multisource units (GMU) to adapt the weight of different features and fuse them. So our MGF strategy can strengthen the specific advantageous features beneficial for classification and weaken the common features.Above all, the several experiment results verify our proposed networks’ effectiveness and robustness.<br>
<img width="960" alt="img" src="https://github.com/blackkiring/DSTD/assets/115865511/b586bbaa-852c-4d99-9f3f-b752065b5dba">   <br>
## Environment
Our experiments are all trained on a work station with RTX3090 24GB GPU. We provide the requirement.txt to help you to set the pytorch environment.
## Data
PAN and MS images (Inconvenient to provide)   <br>
groundtruth file <br>
## Preprocessing
We provide preprocess.py to process our input PAN and MS images. You can change the dataset and the size of images in the preprocess.py. Furthermore, you need to change the number of categries according to different dataset. The majority paragrams you can set them in preprocess.py.

## train
our model is an end-to-end network. To train our network, you only need to run the Demo.py. We set hyper-parameters according to the best performance in the experiments of our net work.
## visualize
Since different datasets have different classification, we provide different visualize code for different datasets. You can change them if you need. The output are the classification
maps, including the image with ground truth and the overall image.


