Spectral modeling analysis, including data preprocessing, wavelength selection, dataset splitting, regression, classification, clustering, and related process visualization.
The overall algorithm flow is as follows:
![图片1](https://github.com/user-attachments/assets/60bf3221-b126-4dd3-b9c0-a64aae484bbc)
The spectral preprocessing offers 11 methods, including MMS, SS, CT, SNV, MA, SG, MSC, FD1, FD2, DT, and WVAE. 
Wavelength selection includes six dimensionality reduction methods: CARS, SPA, LARS, UVE, GA, and PCA.
Dataset splitting methods include Random, SPXY, and KS.
For regression models, classic chemometric quantitative analysis methods are provided, such as PLSR, RF, SVR, ELM, and ANN, as well as advanced deep learning methods like CNN and Transformer.
Classification models include classic chemometric methods like PLS-DA, ANN, SVM, and RF, along with advanced deep learning methods like CNN, SAE, and Transformer.
Clustering methods include KMeans and FCM.
The model evaluation module provides common evaluation metrics for model assessment.

Main function: main.py
Spectral data loading: DataLoad.py

Spectral visualization:![图片2](https://github.com/user-attachments/assets/4f5a00c9-ebde-45a5-b433-26b457c2bc0b)

Preprocessing visualization:![图片3](https://github.com/user-attachments/assets/1fadcbaa-3d9f-4b46-a0ac-cc43e1d0e47b)

Wavelength selection:![图片5](https://github.com/user-attachments/assets/e6b238cd-69e9-420d-ae38-050875b37e45)

Regression fitting:![1730689652815](https://github.com/user-attachments/assets/7404ed79-4adc-4101-8784-173e016d6516)
