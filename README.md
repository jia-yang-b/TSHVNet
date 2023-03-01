# TSHVNet
![]( https://img.shields.io/badge/license-MIT-green.svg)  
This repo. is the official implementation of "Simultaneous Nuclear Instance Segmentation and Classification in Histopathological Images Based on Multi-Attention Mechanisms" .  
Please see the [paper](https://www.hindawi.com/journals/bmri/2022/7921922/).  

## Overview    
<img width="407" alt="image" src="https://user-images.githubusercontent.com/59470630/222140327-fc5ab726-7b18-42bc-8349-d37b7b5b3c6f.png">


##Visual demonstration of the comparative performance of different models on the Pannuke dataset.   
<img width="333" alt="image" src="https://user-images.githubusercontent.com/59470630/222140326-8f74911c-9772-42f3-ab0d-5a16b2856a1b.png">

##Visual demonstration of the comparative performance of different models on the Consep dataset.
## Run  
1.Requirements:  
* python3  
* Pytorch 1.0.1
* 
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
The experimental procedure is the same as TSHVNet
## Overview   

## Citation  
If you find our paper/code is helpful, please consider citing:  
```Yuli Chen, Yao Zhou, Guoping Chen, Yuchuan Guo, Yanquan Lv, Miao Ma, Zhao Pei, Zengguo Sun, "Segmentation of Breast Tubules in H&E Images Based on a DKS-DoubleU-Net Model", BioMed Research International, vol. 2022, Article ID 2961610, 12 pages, 2022. https://doi.org/10.1155/2022/2961610```



