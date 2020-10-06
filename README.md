# Interpretability-guided Content-based Medical Image Retrieval
Repository related to the paper: Interpretability-guided Content-based Medical Image Retrieval

## Abstract 
When encountering a dubious diagnostic case, radiologists typically search in public or internal databases for similar cases that would help them in their decision-making process. This search represents a massive burden to their workflow, as it considerably reduces their time to diagnose new cases. It is, therefore, of utter importance to replace this manual intensive search with an automatic content-based image retrieval system. However, general content-based image retrieval systems are often not helpful in the context of medical imaging since they do not consider the fact that relevant information in medical images is typically spatially constricted. In this work, we explore the use of interpretability methods to localize relevant regions of images, leading to more focused feature representations, and, therefore, to improved medical image retrieval. As a proof-of-concept, experiments were conducted using a publicly available Chest X-ray dataset, with results showing that the proposed interpretability-guided image retrieval translates better the similarity measure of an experienced radiologist than state-of-the-art image retrieval methods. Furthermore, it also improves the class-consistency of top retrieved results, and enhances the interpretability of the whole system, by accompanying the retrieval with visual explanations.

## Method
![Alt text](aux_images/Method.png?raw=true "Title")

## Results
TO DO

## Installation 
To run the scripts, you need to have installed: 
* Keras 2.2.4
* tensorflow-gpu	1.10.0
* scikit-learn	0.21.1
* numpy	1.16.3
* pandas	0.25.3
* scipy	1.1.0
* cv2 4.2.0

## Instructions 
* Run precision.py if you want to replicate precision results presented in the paper. 

## Citation
If you use this repository, please cite this paper:
> Wilson Silva, Alexander Poellinger, Jaime S. Cardoso, Mauricio Reyes. Interpretability-guided Content-based Medical Image Retrieval. In Proceedings of the 22nd International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI'2020)

## Questions: 
If you have any question regarding the information provided in this repository, please contact the first author of the paper. E-mail: wilson.j.silva@inesctec.pt
