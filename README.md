# Urban-invasive-species
Code for "Detecting Plant Invasion in Urban Parks with Aerial Image Time Series and Residual Neural Network". The paper has been accepted to remote sensing journal on Oct 22 2020.


Background:

Invasive plants are a major agent threatening biodiversity conservation and directly affecting our living environment. This study aims to evaluate the potential of deep learning, one of 22 the fastest-growing trends in machine learning, to detect plant invasion in urban parks using high resolution (0.1 m) aerial image time series. Capitalizing on a state-of-the-art, popular architecture Residual Neural Network (ResNet), we examined key challenges applying deep learning to detect plant invasion: relatively limited training sample size (invasion often confirmed in the field) and 26 high forest contextual variation in space (from one invaded park to another) and over time (caused 27 by varying stages of invasion and the difference in illumination condition). To do so, our evaluations focused on a widespread exotic plant, autumn olive (Elaeagnus umbellate) which), that has invaded 20 urban parks across Mecklenburg County (1,410 km2) in North Carolina, USA. 

The results demonstrate a promising spatial and temporal generalization capacity of deep learning to detect urban invasive plants. In particular, the performance of ResNet was consistently over 96.2% using training samples from 8 (out of 20) or more parks. The model trained by samples from only 4 parks still achieved an accuracy of 77.4%. ResNet was further found tolerant 34 of high contextual variation caused by autumn olive’s progressive invasion and the difference in illumination condition over the years. Our findings shed light on prioritized mitigation actions for effectively managing urban invasive plants.

Implementation:

To deal with those challenges, this repo provides three data loaders for each scenario presented in the paper. Then data is prepared and sent to pytorch deep learning model. Results are reported after training phase completes.

Please feel free to leave any comments and feedback to cli33@uncc.edu. Thank you for your interest.
