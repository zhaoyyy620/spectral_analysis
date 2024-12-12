Total workflow of spectral modeling analysis, including data preprocessing, wavelength selection, dataset splitting, regression, classification, clustering, and related process visualization.
The related algorithm flow is as follows:

![图片1](https://github.com/user-attachments/assets/60bf3221-b126-4dd3-b9c0-a64aae484bbc)

Spectral preprocessing includes 11 methods: MMS, SS, CT, SNV, MA, SG, MSC, FD1, FD2, DT, and WVAE. 
Wavelength selection utilizes six dimensionality reduction techniques: CARS, SPA, LARS, UVE, GA, and PCA.
Dataset splitting methods include Random, SPXY, and KS.
For regression modeling, both traditional chemometric quantitative analysis methods, such as PLSR, RF, SVR, ELM, and ANN, and advanced deep learning approaches, including CNN and Transformer, are provided.
Classification models comprise traditional chemometric methods, such as PLS-DA, ANN, SVM, and RF, as well as advanced deep learning models like CNN, SAE, and Transformer.
Clustering methods include KMeans and FCM.
Finally, the model evaluation module offers commonly used metrics for assessing model performance, ensuring comprehensive evaluation of results.

Main function: main.py
Spectral data loading: DataLoad.py

Spectral visualization:![图片2](https://github.com/user-attachments/assets/4f5a00c9-ebde-45a5-b433-26b457c2bc0b)

Preprocessing visualization:![图片3](https://github.com/user-attachments/assets/1fadcbaa-3d9f-4b46-a0ac-cc43e1d0e47b)

Wavelength selection:![图片5](https://github.com/user-attachments/assets/e6b238cd-69e9-420d-ae38-050875b37e45)

Regression fitting:![1733983795126](https://github.com/user-attachments/assets/3c36ef3c-dae6-4263-8342-29ddc9ab7a7c)

Pixel-level predicted value mapping:![1733984207040](https://github.com/user-attachments/assets/ca3346be-a21f-4294-b649-4f2c2c58da6b)

We have provided a comprehensive and scalable spectral analysis framework designed to bridge traditional methods with cutting-edge deep learning technologies, opening up new possibilities for spectral analysis. We look forward to increased exchange and collaboration between academia and industry, inspiring more innovation and practice.   By working together, the potential of spectroscopic analysis can be fully realized, elevating its impact in both fundamental research and real-world applications, and driving the field to unprecedented heights.
