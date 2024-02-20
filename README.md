# Physics-based Deep Learning for Imaging Neuronal Activity via Two-photon and Light Field Microscopy

Welcome to the code repository for our paper, "Physics-based Deep Learning for Imaging Neuronal Activity via Two-photon and Light Field Microscopy", available at https://ieeexplore.ieee.org/document/10141580 and https://www.biorxiv.org/content/10.1101/2022.10.11.511633v1.full.pdf. Our work presents a deep learning framework for neuronal activity imaging, integrating LFM's rapid imaging with the high-resolution capabilities of 2P microscopy.
To apply the framework to your datasets, initiate the process with the following scripts in order:

initFrwd.py - Initializes the forward model.  
initG.py - Sets up the generative network.  
mainLF2P.py - Commences the main adversarial training loop.


We drew inspiration for part of the adversarial component from the work of G. Kwon, C. Han, and D.-s. Kim in their paper, "Generation of 3D Brain MRI Using Auto-Encoding Generative Adversarial Networks". A special shoutout to the authors for sharing their code at https://github.com/cyclomon/3dbraingen !
