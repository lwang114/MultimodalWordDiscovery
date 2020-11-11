# MultimodalWordDiscovery
This repository contains the code for the paper [``A DNN-HMM-DNN Hybrid Model for Discovering Word-like Units from
Spoken Captions and Image Regions''](https://www.researchgate.net/publication/343218251_A_DNN-HMM-DNN_Hybrid_Model_for_Discovering_Word-like_Units_from_Spoken_Captions_and_Image_Regions).

### How to run it
Requirement: Pytorch 0.3 for pretraining the VGG 16/Res 34 net
  
1. Download the MSCOCO 2k, MSCOCO 20k image features from [here](https://drive.google.com/file/d/14iShQBAc_Y1-QnPfB3Sit2YC6ZnuMJIM/view?usp=sharing) and put them under the directory data/mscoco
2. Download the pretrained image classifier weights from [here](https://drive.google.com/file/d/1nHXpvYrOgjpu63B4Ylckrf_iMuSTDIix/view?usp=sharing)
3. Example: Run the linear softmax model with Res 34 image features on MSCOCO 2k: 
```
python run_image2phone.py --dataset mscoco2k --feat_type res34 --model_type linear --image_posterior_weights_file classifier_weights.npz --lr 0.01
```
4. Run the following for help on more customized experiments: 
```
python run_image2phone.py --help
```

### Cite
Please consider citing the following paper if you use the code:
```
@inproceedings{WH-interspeech2020,
    author = {Liming Wang and Mark Hasegawa-Johnson},
    title = {A {DNN-HMM-DNN} Hybrid Model for Discovering Word-like Units from Spoken Captions and Image Regions},
    booktitle = {Interspeech},
    year = {2020}
}
```
