# Portfolio
---

<h2 id="Deep1">Deep-Learning and Computer Vision</h2>

### Separating Latent Overlapped Fingerprints

<center><img src="images/LOG1.png" width="50%" height="50%" alt="A 100x100 image"/></center>

<u>Oct 2021 - Now</u>

<div style="text-align: justify">Overlapped fingerprints are frequently encountered in latent fingerprints lifted from crime scenes. It is necessary to separate such overlapped fingerprints into component fingerprints so that existing fingerprint matchers can recognize them. In this project, we are trying to separate these fingerprints to provide two reconstructed fingerprints completely. We are currently basing our work on U-Net architecture, image augmentations, and some non-trivial masks (geometrics, ...). We generated the database with GAN to improve the learning of supervised algorithms.</div>
<br>
<center><img src="images/finger1.png"/></center>

**Bibliography:**

[1] Jianjiang Feng, Yuan Shi, Jie Zhou, "Robust and Efficient Algorithms for Separating Latent Overlapped Fingerprints", IEEE Transactions on Information Forensics and Security, vol. 7, no. 5, pp. 1498-1510, 2012. 
<br>
[2] Fanglin Chen, Jianjiang Feng, Anil K. Jain, Jie Zhou, and Jin Zhang, "Separating Overlapped Fingerprints", IEEE Transactions on Information Forensics and Security, vol. 6, no. 2, pp. 346-359, 2011. 

---
### Accident Anticipation and Risky Region Localization

<u>Oct 2021 - Now</u>

<div style="text-align: justify">A critical risk in autonomous vehicle is accident caused by an the external environment. Thus, we are investigating methods to anticipate risk and define a risky region localization. We are currently working on [1] to implement their work and adapt it to autonomous vehicles. We use a dashcam database to train and test the performance of the algorithm. </div>
<br>
<center><img src="images/DASHCAM.png"/></center>

**Bibliography:**

[1] Zeng, K.-H. et al. (2017) ‘Agent-Centric Risk Assessment: Accident Anticipation and Risky Region Localization’, arXiv:1705.06560 [cs] [Preprint]. Available at: http://arxiv.org/abs/1705.06560 (Accessed: 3 November 2021).
<br>
[2] Jain, A. et al. (2016) ‘Brain4Cars: Car That Knows Before You Do via Sensory-Fusion Deep Learning Architecture’, arXiv:1601.00740 [cs] [Preprint]. Available at: http://arxiv.org/abs/1601.00740 (Accessed: 3 November 2021).


---
### Semi-supervised algorithms and image augmentations for the segmentation of satellite images

<center><img src="images/ONERA.png" width="50%" height="50%" alt="A 100x100 image"/></center>

<u>May 2021 - Aug 2021</u>

<I> Private Research paper</I>

<div style="text-align: justify">This research work was completed through an internship at ONERA and CNES (Aerospace Lab). My role was to complete research on semi-supervised learning algorithms to increase the performance of identifying the buildings with a minimal database (Mean Teacher and Naive Student algorithms [1]). </div>
 <br> 

<center><img src="images/satelite1.png"/></center>

**Bibliography:**

[1] Tarvainen, A. and Valpola, H. (2018) ‘Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results’, arXiv:1703.01780 [cs, stat]. Available at: http://arxiv.org/abs/1703.01780 (Accessed: 2 August 2021)

[2] French, G. et al. (2020) ‘Semi-supervised semantic segmentation needs strong, varied perturbations’, arXiv:1906.01916 [cs]. Available at: http://arxiv.org/abs/1906.01916 (Accessed: 2 August 2021)

---
### Autonomous UAV: Dassault UAV Challenge


[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Yannis-98/YOLOv4-Cloud-Tutorial)

<center><img src="images/LOG2.png" width="50%" height="50%" alt="A 100x100 image"/></center>

<u>Sept 2020 - May 2021</u>


<div style="text-align: justify">Competition gathering the best students in Europe. The UAV must carry out several missions. The first one is to automatically follow ground marks to reach a target. The second is that the UAV has to identify different targets according to the orders given beforehand (different forms and colors) and carefully deposit packages. As I lead the Artificial intelligence development of this project, I must coordinate the team. 
 
<br>
 
I developed forecasting and identifying methods to complete different tasks automatically, such as ground markers identification and parcel drop-off (YOLO method, non-linear optimization, gradient boosting, clustering...). To create the database, I generated artificial images with simulation software and python scripts. I enhanced the database by varying different external parameters (brightness to simulate an image taken at the end of the day...) of images to improve the diversity within the database.</div>
<br>
<center><img src="images/uav_test_t.png"/></center>

**Bibliography:**

[1] Bochkovskiy, A., Wang, C.-Y. and Liao, H.-Y.M. (2020) ‘YOLOv4: Optimal Speed and Accuracy of Object Detection’, arXiv:2004.10934 [cs, eess] [Preprint]. Available at: http://arxiv.org/abs/2004.10934 (Accessed: 3 November 2021).

---
### Colorization Transformer by Google

<u>Feb 2021 - July 2021</u>

<div style="text-align: justify">The objective was to understand and implement the research paper [1]. Image colorization is a challenging, inherently stochastic task that requires a semantic understanding of the scene and knowledge of the world. Core immediate applications of the technique include producing organic new colorizations of existing image and video content as well as giving life to originally grayscale media, such as old archival images and videos.</div>
<br>
<center><img src="images/google.png"/></center>

**Bibliography:**

[1] Kumar, M., Weissenborn, D. and Kalchbrenner, N. (2021) ‘Colorization Transformer’, arXiv:2102.04432 [cs] [Preprint]. Available at: http://arxiv.org/abs/2102.04432 (Accessed: 1 September 2021).

---
### Image Inpaiting

<u>Jan 2020 - July 2020</u>

<div style="text-align: justify">mage inpainting focuses on the reconstruction of deteriorated images or when fillingmissing parts. This technique is used extremely often in images reconstitution because itallows to reconstitute historical images which have deteriorated. I principally studied the LASSO algorithm which is based on contraction of regression coefficients. and the ridgeregression. The linear regression consists in resolving: </div>
<br>
<center><a href="https://www.codecogs.com/eqnedit.php?latex=\omega&space;^*&space;=&space;argmin(X\omega&space;-&space;y)^T(X\omega-y)&plus;\alpha\Omega(w)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega&space;^*&space;=&space;argmin(X\omega&space;-&space;y)^T(X\omega-y)&plus;\alpha\Omega(w)" title="\omega ^* = argmin(X\omega - y)^T(X\omega-y)+\alpha\Omega(w)" /></a></center>
<br>
<center><img src="images/inpaint.PNG"/></center>

**Bibliography:**

[1] Bin Shen and Wei Hu and Zhang, Yimin and Zhang, Yu-Jin: Image Inpainting via Sparse Representation Proceedings of the 2009 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP ’09)

---
### Particle Tracking 

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Yannis-98/Particle-detection-and-tracking---Velocimetry-)

<u>Jan 2019 - July 2019</u>

<div style="text-align: justify">The project's objective was to study and numerically represent a flow around a cylinder using the principle of PTV. A camera with a laser was placed in front of the cylinder; the flow contains particles that do not influence the characteristics of the flow. Our role was to use images from the camera. Using the particle's flow around the cylinder, we can use our mathematical and computing knowledge. The first step was to identify particles on each image from the film. The background being black, we can identify white particles. We used a correlation method based on neural networks. Through weights, the method identifies the particles that resemble each other between two images. Then we infer vector fields</div>
<br>
<center><img src="images/veloci.png"/></center>
<br>

---
## Optimization
### Propeller optimization, airship UAV

<u>Sept 2020 - Jan 2021</u>

<div style="text-align: justify">The aim is to dimension a propeller from a dirigible-type drone intended for indoor aerial evolution in the large ENSAM lecture hall. Initially, the drone will be a communication tool enabling images to be transmitted or carried autonomously and securely. Then, it can be modified to allow inspection or control of structures such as the vault of the large lecture hall. The airship must move extremely slowly to respect these specifications because it will only be used as a communication tool. Due to the type of UAV (dirigible), the UAV has to move thanks to the gas in the airship essentially. Propellers are only here to begin the movement.
The aim is then to maximize the mechanical efficiency. I decided to develop a method to generate the most effective helix. By using using characteristic points from a blade I developed a parametric method by randomly modifying in a value interval the data of the helix.</div>
<br>
<center><a href="https://www.codecogs.com/eqnedit.php?latex=A_{j}^{i}=A_{0}^{i}&plus;F_{j}^{i}(X_{j}^{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A_{j}^{i}=A_{0}^{i}&plus;F_{j}^{i}(X_{j}^{i})" title="A_{j}^{i}=A_{0}^{i}+F_{j}^{i}(X_{j}^{i})" /></a></center>
<br>
<center><img src="images/uav_airship.PNG"/></center>
<br>

---
### Heat Exchanger Optimization 

<u>Jan 2019 - July 2019</u>

<div style="text-align: justify">The objective of this project is to find a solution to recuperate the warmth coming out of a system that produces fumes at high temperatures. The solution I found was a creation of an exchanger to recuperate fumes to warm the air, which entered another system. The other system required air at the temperature of T=800K. I had to dimension my exchanger to respect size specifications.
<br>
The objective is then to achieve the most efficient exchanger possible. In the first step, I calculated the different coefficients governing the thermal transfer (DTLM method, NUT...) to complete a pre-dimensioning. I created an exchanger geometry with the highest exchange surface between air and fumes to obtain an efficient thermal exchange between air and fumes. I developed my solution in CATIA. I finally simulated my model and analyzed our results under StarCCM+ (CFD software). According to the results, we had to complete the dimensioning by determining the perfect dimensions, materials, and architecture to obtain the most optimized heat exchanger.</div>
<br>
<center><img src="images/echan.png"/></center>
<br>

---
## Reinforcement Learning
### Lunar Landing 

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1KDPy4kTRqeoygDsbhbr3Dr3lcRgYk99c)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Yannis-98/Reinforcement-Learning-LunarLander)

<u>Oct 2021</u>


<div style="text-align: justify">This  project concerns the landing of a shuttle on the moon. The objective is to land the shuttle using reinforcement learning methods. We must then determine the most efficient policy possible in this environment and optimize the method. Reinforcement learning is based on reward and episode; the shuttle gains rewards when approaching the landing zone. On the contrary, it loses its reward when it moves away from the zone. The episode ends when the shuttle crashes. I study the performance of policies (2C, PPO, DQN, and QR-DQN) and optimize them with Optuna. I found that the PPO provides the best performance.</div>
<br>
<center><img src="images/PPO.gif"/></center>

**Bibliography:**

[1] https://gym.openai.com/envs/LunarLander-v2/

---
### Genetic Algorithms

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://drive.google.com/drive/u/0/folders/11AsHT-SIPapUlx2fFjee14zFXpgJwj94)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Yannis-98/Genetic-Algorithms)

<u>Sept 2021</u>


<div style="text-align: justify">The objective of this project is to understand genetic algorithms concept. We studied NSGA-II, DEAP, and multi-objective algorithms. I will implement evolutionary algorithms, especially multi-objective ones to complete learning. To do this, different functions are provided to plot the individual generated at each generation or to compare the results between them. </div>
<br>
<center><img src="images/gymcart.gif"/></center>

---
<center>© 2021 Yannis Montreuil Powered by Jekyll and the Minimal Theme.</center>
