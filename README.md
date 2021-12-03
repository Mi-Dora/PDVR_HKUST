# Partial Duplicate Video Retrieval based on Deep Metric Learning
This repository contains the implementation of the term project for 21 Fall-HKUST CSIT 5910.

We use one publicly available dataset namely [CC_WEB_VIDEO](http://vireo.cs.cityu.edu.hk/webvideo/) , to download this dataset, you can use our developed tools in [CC_WEB_VIDEO Downloader](https://github.com/Mi-Dora/CC_WEB_VIDEO_Downloader).

And we also use one self-made dataset, which is made up of videos from Bilibili and can be downloaded [here](https://drive.google.com/drive/folders/1XYat0tl2vFmquWZsOYW3w19reJMnkYOJ?usp=sharing).

## Getting started

### Installation

* Clone this repo:
```bash
git clone https://github.com/Mi-Dora/PDVR_HKUST
cd PDVR_HKUST
```
* You can install all the dependencies by
```bash
conda install --file requirements.txt
```

### Run PDVR Demo

```shell
python run_demo.py
```

### DML training

* Triplets from the CC_WEB_VIDEO can be injected if the global features and triplet of the evaluation set
 are provide.
```bash
python train_dml.py --evaluation_set output_data/cc_web_video_features.npy --evaluation_triplets output_data/cc_web_video_triplets.npy --train_set output_data/vcdb_features.npy --triplets output_data/vcdb_triplets.npy --model_path model/
```

### Evaluation

* Evaluate the performance of the system by providing the trained model path and the global features of the 
CC_WEB_VIDEO.
```bash
python evaluation.py --fusion Early --evaluation_set output_data/cc_vgg_features.npy --model_path model/
````
OR
```bash
python evaluation.py --fusion Late --evaluation_features cc_web_video_feature_files.txt --evaluation_set output_data/cc_vgg_features.npy --model_path model/
```

* The *mAP* and *PR-curve* are returned

## Related Project
**[CC_WEB_VIDEO Downloader](https://github.com/Mi-Dora/CC_WEB_VIDEO_Downloader)**

## Reference Projects

**[Near-Duplicate Video Retrieval  with Deep Metric Learning](https://github.com/MKLab-ITI/ndvr-dml)**

**[Intermediate CNN Features](https://github.com/MKLab-ITI/intermediate-cnn-features)**
