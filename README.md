# TSHVNet
![]( https://img.shields.io/badge/license-MIT-green.svg)  
This repo. is the official implementation of "TSHVNet:Simultaneous Nuclear Instance Segmentation and Classification in Histopathological Images Based on Multi-Attention Mechanisms" .  
Please see the [paper](https://www.hindawi.com/journals/bmri/2022/7921922/).  

## Overview    
<img width="333" alt="image" src="https://user-images.githubusercontent.com/59470630/222143496-76c5f33b-0edc-462e-b261-7f91a6f6a033.png">


##Visual demonstration of the comparative performance of different models on the Pannuke dataset.   
<img width="407" alt="image" src="https://user-images.githubusercontent.com/59470630/222143572-444c44e5-f016-4092-9997-7815cb1241e2.png">

##Visual demonstration of the comparative performance of different models on the Consep dataset.
<img width="416" alt="image" src="https://user-images.githubusercontent.com/59470630/222143606-77779280-8448-41e5-9d38-d98b45164ad1.png">


## Run  
1.Requirements:  
* python3  
* Pytorch 1.0.1
We have uploaded the corresponding environment package for your convenience. 

2.Training:  
* Prepare the required images and store them in new_data folder, the example format of the training, validation, and testing datasets is in the  "dataset" folder.
consep training dataset：hv2/dataset/training_data/consep/train/540×540_80×80
consep val dataset：hv2/dataset/training_data/consep/valid/540×540_80×80
consep test dataset：hv2/dataset/CoNSeP/Test/Images
* Run ``` Python run_train.py```  

3. get the best weight file 
* cd hv2
* Run ``` Python hv2/find_best.py```  

4.Testing:
Modify weight path：
* Run ``` Sh run_tile.sh```  
You'll get four files at "data4/jyh/hv2/dataset/sample_titles/pred"

5.Evaluation quantitative index calculation
* Run ```Python compute_stats.py```

UL-HVNet:
## Overview   
<img width="415" alt="image" src="https://user-images.githubusercontent.com/59470630/222142723-f9b1addf-a93d-4c47-a12a-f9c2077ff638.png">

##Visual demonstration of the comparative performance of different models on the Pannuke dataset.   
<img width="416" alt="image" src="https://user-images.githubusercontent.com/59470630/222144144-8e66ab13-d10f-4074-bc9d-da81ff8c97e0.png">

##Visual demonstration of the comparative performance of different models on the Consep dataset.
<img width="416" alt="image" src="https://user-images.githubusercontent.com/59470630/222144117-7b8b01b4-deaf-4fec-b908-c7f6eb726346.png">
The experimental procedure is the same as TSHVNet

## Citation  
If you find our paper/code is helpful, please consider citing:  
```Yuli Chen, Yuhang Jia, Xinxin Zhang, Xue Li, Miao Ma, Zengguo Sun, Zhao Pei, "TSHVNet: Simultaneous Nuclear Instance Segmentation and Classification in Histopathological Images Based on Multiattention Mechanisms", BioMed Research International, vol. 2022, Article ID  7921922, 17 pages, 2022. https://www.hindawi.com/journals/bmri/2022/7921922/```



