# MultimodalWordDiscovery
This repository contains the code for the submission ``A DNN-HMM-DNN Hybrid Model for Discovering Word-like Units from
Spoken Captions and Image Regions''.

### How to run it
Requirement: Pytorch 0.3 for pretraining the VGG 16/Res 34 net
  
1. Download the MSCOCO 2k, MSCOCO 20k image features from here and put them under the directory data/mscoco
2. Download the pretrained image classifier weights from here
3. Example: Run the linear softmax model with Res 34 image features on MSCOCO 2k: 
```
python run_image2phone.py --dataset mscoco2k --feat_type res34 --model_type linear --image_posterior_weights_file classifier_weights.npz --lr 0.01
```
4. Run the following for help on more customized experiments: 
```
python run_image2phone.py --help
```
