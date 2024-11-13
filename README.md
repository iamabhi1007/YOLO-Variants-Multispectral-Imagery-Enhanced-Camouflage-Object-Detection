# YOLO-Variants-Multispectral-Imagery-Enhanced-Camouflage-Object-Detection
Enhanced Camouflage Object Detection in Multispectral Imagery using Advanced YOLO Variants. 

Abstract. This study investigates the efficacy of advanced YOLO (You Only Look Once) variants, specifically YOLOv5, v7, v8, and v9, in detecting con-cealed camouflaged objects within multispectral imagery. Utilizing the Altum-PT multispectral camera, a comprehensive dataset of multispectral images was cap-tured and preprocessed using techniques such as fusion, false coloring, and pansharpening to enhance image quality and extract relevant features. The per-formance of these YOLO models was evaluated through extensive experiments across diverse environmental contexts. The results demonstrate that YOLO vari-ants, especially when combined with Slicing Aided Hyper Inference (SAHI), ex-hibit robust detection capabilities, with precision ranging from 93.9% to 98.2% and recall varying from 80.2% to 98.1%. Despite variations in performance met-rics due to architectural nuances and training methodologies, all models show promise for applications in surveillance and security. The implementation of SAHI successfully rectified false detections, improving prediction accuracy. This study contributes to advancing concealed object detection techniques and under-scores the potential of deep learning in enhancing surveillance capabilities.
Keywords: Multispectral, Camouflaged objects, Altum-PT, SAHI, YOLO, False colouring, deep learning, Aerial Imagery, Surveillance, Reconnaissance.

Implementation

The implementation of this study was carried out using Google Colab, leveraging its GPU support (T4 GPU) for efficient model training, validation, and testing. Python served as the primary programming language, with the Ultralytics YOLO models (v5, v7, v8, and v9) and SAHI models utilized to facilitate accurate camouflaged object detection. Google Colab’s environment enabled the rapid processing and experimentation needed for this study, while its GPU resources optimized the training and testing phases, ensuring high-performance model evaluation and consistent results across varied environmental contexts.

The detailed description of models and processes refer the Ultralytics website. https://docs.ultralytics.com/models/

For creating your own custom model refer to videos featuring custom object detection using YOLO.

Installation

Include step-by-step installation instructions, using package managers like pip or conda.

Example:
pip install -r requirements.txt

1. Running the Code

Include example commands to run the main script or specific functions.

Example:
python main.py --input <input_file> --output <output_file>

2. Model Training, Validation, and Testing

Example command for training: (similarly for validation and testing)

python train.py --model yolov8 --data <data_path> --epochs 50

Methodology

![image](https://github.com/user-attachments/assets/4c849086-7740-4b59-981b-c101b6722210)

Fig. 1. Process flow of the proposed methodology for Camouflage Reconnaissance.

![image](https://github.com/user-attachments/assets/4a494529-3688-4030-b955-f75db8651bb9)

Fig. 2. Electromagnetic spectrum for understanding different wavelength bands [11].

![image](https://github.com/user-attachments/assets/2a4684b2-ec26-4358-8007-a42b1f070fbb)
![image](https://github.com/user-attachments/assets/bc3eef7b-d704-434e-ab95-d392fc4654e8)

Fig. 4. Dataset Collection procedure. (a) Camera setup, (b) Target Objects setup for camouflage, (c) Collection of Data, (d) Labels/Instances Graph represents the num-ber of collected sample target distributions in the dataset, (e) Sample Multispectral images captured in various bands and (f) Captured sample target camouflage ob-jects.

![image](https://github.com/user-attachments/assets/c530e410-dc16-4f60-bcdb-cb5d442dd21e)

Fig. 5. Results of Pre-Processing Method (a) Fused Multispectral images of 5 bands, (b) After Pansharpening, (c) Before False Colouring and (d) After False Colouring.

Results

![image](https://github.com/user-attachments/assets/daa3f04c-222f-45b8-b02d-c265aee99e12)

Fig. 6. YOLO v5 without and with SAHI (Zoomed for better view of the object detec-tion). (a) YOLO v5 without SAHI Output showing false detection, and (b) YOLO v5 prediction for the same image with SAHI correct prediction.

![image](https://github.com/user-attachments/assets/1dbd5ab9-abd8-4126-a413-e16941ad2ff3)

Fig. 7. YOLO v8 prediction without and with SAHI (Zoomed for better view of the object detection). (a) YOLO v8 without SAHI Output partial detection, and (b) YOLO v8 prediction for the same image with SAHI better detection.

![image](https://github.com/user-attachments/assets/e9810af6-8e8e-42fb-85f6-c69d6158ea62)

Fig. 8. YOLO v5 prediction on test images in various environments, fused and multi-spectral images (Highlighted for understanding). (a) Person, Soldier and Camou-flagetent prediction, (b) Detected camougflagetent, person and soldier in low contrast, shadow regions and, also with partial sunlight, (c) Person, Soldier and Camou-flagetent prediction in a different scenario, and (d) False coloured Image for Soldier, Person and Camouflagetent prediction.

![image](https://github.com/user-attachments/assets/871656be-d39e-44eb-b300-c88def1cec3d)

Fig. 9. YOLO v7 prediction on test images in various environments, fused and multi-spectral images (Highlighted for understanding). (a) Person, Soldier and Camou-flagetent prediction, (b) Detected camougflagetent, person and soldier in low contrast, shadow regions and, also with partial sunlight, (c) Person, Soldier and Camou-flagetent prediction in a different scenario, and (d) False coloured Image for Soldier, Person and Camouflagetent prediction.

![image](https://github.com/user-attachments/assets/7da37f11-2d6d-435f-9034-615d81b2240d)

Fig. 10. YOLO v8 prediction on test images in various environments, fused and mul-tispectral images (Highlighted for understanding). (a) Person, Soldier and Camou-flagetent prediction, (b) Detected camougflagetent, person and soldier in low contrast, shadow regions and, also with partial sunlight, (c) Person, Soldier and Camou-flagetent prediction in a different scenario, and (d) False coloured Image for Soldier, Person and Camouflagetent prediction.

![image](https://github.com/user-attachments/assets/b7ef8f81-e243-41d0-8d0e-31532c6828d8)

Fig. 11. YOLO v9 prediction on test images in various environments, fused and mul-tispectral images (Highlighted for understanding). (a) Person, Soldier and Camou-flagetent prediction, (b) Detected camougflagetent, person and soldier in low contrast, shadow regions and, also with partial sunlight, (c) Person, Soldier and Camou-flagetent prediction in a different scenario, and (d) False coloured Image for Soldier, Person and Camouflagetent prediction.

![Screenshot 2024-11-11 105552](https://github.com/user-attachments/assets/0d25c1bd-8b44-417d-9132-ec2d69c86ac0)

Discussion
Across all models, YOLO v5 and v9 demonstrated the most consistent high perfor-mance, with YOLO v5 excelling in recall and YOLO v9 providing strong localization at mAP@0.5:.95. YOLO v8 displayed commendable precision. However, recall in complex scenarios remains a limitation. YOLO v7 showed robust detection for well-defined objects but faced challenges with camouflaged targets. This comparative analysis reveals that while YOLO v9 offers superior adaptability across diverse envi-ronments, YOLO v5's high precision and recall make it a valuable choice for camou-flaged object detection tasks, particularly in surveillance and security applications.
However, the findings confirm that SAHI with YOLO models exhibit improved per-formance in detecting camouflaged objects, a notable advancement over traditional approaches. For example, YOLOv5 and v8 with SAHI mitigated false detections, a critical improvement over methods like RYOLO and S2ANet which, although effec-tive in spectral attention, lacked precision in fine-grained object differentiation in high-noise or low-contrast scenarios [5,12]. Refining YOLO’s detection capabilities with multispectral fusion and pansharpening aligns with recent advances while intro-ducing novel optimizations tailored to challenging camouflaged conditions.
The proposed approach differs from EAPT and other pyramid transformers that use hierarchical attention mechanisms, as we prioritize lightweight architectures for real-time processing, which is critical in surveillance applications. Compared to these more computationally intensive models, our implementation offers a practical balance between accuracy and resource efficiency, addressing operational demands in dy-namic and resource-limited environments [14].

Conclusions
The study highlights the effectiveness of YOLO variants from v5 to v9 for concealed object detection in multispectral imagery. Through experimentation, it is shown that these models accurately identify targets across diverse conditions. The detection im-ages affirm their reliability, with each variant displaying varying performance metrics due to architectural nuances and training methodologies. The novelty of the research lies in the implementation of SAHI with v5, v8 was done successfully in rectifying false detection and confidently predicting camouflage objects with an additional 10 % accuracy compared to without SAHI. The use of fusion, pansharpening, and false coloring in preprocessing uniquely strengthened the models’ ability to capture critical details, supporting applications in security and surveillance. Despite all the differ-ences, the chosen models exhibit promise for surveillance and security applications. This study contributes to advancing concealed object detection and underscores the potential of deep learning in security contexts. The outcomes of these studies give way for future research and should focus on optimizing hyperparameters, incorporat-ing more data, and exploring advanced architectures to enhance detection capabilities.

References
1.	 Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934. [DOI: N/A]
2.	Wang, X., Zhang, T., Xu, J., & Wang, X. (2021). D2Det: Towards High Quality Object De-tection and Instance Segmentation. arXiv preprint arXiv:2105.01677. [DOI: N/A]
3.	Giorgi, G. A., Macêdo, D., Gama, V., & Pereira, J. M. V. (2021). Image Fusion in Remote Sensing: A Review of Current State-of-the-Art and Possible Future Developments. Remote Sensing, 13(3), 393. [DOI: 10.3390/rs13030393]
4.	 Zhu, C., & Ma, Y. (2021). Info-guided AGN: Utilizing Information Flow to Improve At-tention Mechanism. arXiv preprint arXiv:2106.13009. [DOI: N/A]
5.	Wang, X., & Zhang, T. (2021). RYOLO: Multi-scale Feature Aggregation for Object Detec-tion in Remote Sensing Imagery. arXiv preprint arXiv:2110.00123. [DOI: N/A]
6.	Saini, S., Kumar, V., & Roy, P. P. (2021). Deep learning-based camouflage object detection: A review. Journal of Visual Communication and Image Representation, 79, 103047. [DOI: 10.1016/j.jvcir.2021.103047]
7.	Singh, G., Kaur, H., & Bansal, J. (2020). Camouflaged Object Detection and Recognition Using Deep Learning: A Review. In Proceedings of the International Conference on Ma-chine Learning, Big Data, Cloud and Parallel Computing (pp. 221-232). Springer, Singa-pore. [DOI: N/A]
8.	Echave, M. S., Garcia-Rodriguez, A., & Skarmeta, A. F. G. (2018). Review on Prepro-cessing Techniques for Hyperspectral Image Processing. Journal of Sensors, 2018, Article ID 3120812, 13 pages. https://doi.org/10.1155/2018/3120812
9.	Zhang, M., Song, Y., Wu, X., Qi, Y., & Sheppard, C. J. (2020). A Survey of Data Fusion Methods and Applications with Hyperspectral Data. Information Fusion, 57, 47–67. [DOI: 10.1016/j.inffus.2019.07.010]
10.	Tilton, N., Baru, C., & Karpatne, M. (2020). Deep Learning for Large-Scale Satellite Image Analysis: A Review. IEEE Geoscience and Remote Sensing Magazine, 8(3), 8–23. [DOI: 10.1109/MGRS.2020.2973197]
11.	Infiniti Optics, "Electromagnetic Spectrum," [Online]. Available: https://www.infinitioptics.com/glossary/electromagnetic-spectrum. [Accessed: July 31, 2024.
12.	Yujie LIU, Xiaorui SUN, Wenbin SHAO, Yafu YUAN. (2024). S2ANet: Combining local spectral and spatial point grouping for point cloud processing. Virtual Reality & Intelligent Hardware 6.4 (2024): 267-279. [DOI: 10.1016/j.vrih.2023.06.005]
13.	Zhiyong Xiao, Yukun Chen, Xinlei Zhou, Mingwei He, Li Liu, Feng Yu, Minghua Jiang. (2024). Human action recognition in immersive virtual reality based on multi-scale spatio-temporal attention network (Computer Animation and Virtual Worlds 35.5 (2024): e2293). [DOI: 10.1002/cav.2293]
14.	Lin, X., Sun, S., Huang, W., Sheng, B., Li, P., & Feng, D.D. (2021). EAPT: Efficient At-tention Pyramid Transformer for Image Processing. IEEE Transactions on Multimedia, 25, 50-61. [DOI: 10.1109/TMM.2021.3120873]

Authors

Abhilash Hegde 1*[0000-0002-0734-9171], 
Siddalingesh S Navalgund 2[0000-0001-6857-915X], 
Archana Nandibewoor 1#$[0000-0002-3698-6057], 
Aachan B Kulkarni 3, 
Anurag G Deshpande 3, 
Abushekh 3, 
Bhairavi M Anantpur 3

*JRF, Research Scholar, ARDB-DRDO Research Lab, Department of Electronics and Commu-nication Engineering, SDM College of Engineering and Technology, Dharwad-580002, Affiliat-ed to Visvesvaraya Technological University, Belagavi-590018, Karnataka, India
2Assistant Professor, Department of Electronics and Communication Engineering, SDM College of Engineering and Technology, Dharwad-580002, Affiliated to Visvesvaraya Technological University, Belagavi-590018, Karnataka, India
#Assistant Professor, ARDB-DRDO Research Lab, Department of CSE, SDM College of Engi-neering and Technology, Dharwad-580002, Affiliated to Visvesvaraya Technological University, Belagavi-590018, Karnataka, India
$Associate Professor, Department of AI/ML, Mangalore Institute of Technology and Engineer-ing, Moodabidri, Affiliated to Visvesvaraya Technological University, Belagavi-590018, Karna-taka, India
3Student, Department of CSE, SDM College of Engineering and Technology, Dharwad-580002, Affiliated to Visvesvaraya Technological University, Belagavi, Karnataka, India

1#narchana2006@gmail.com 1*abhilash.hegde1007@gmail.com 









