This repository hosts the code for our publication "Physics-based Deep Learning for Imaging Neuronal Activity via Two-photon and Light Field Microscopy".
As mntiond in our paper https://ieeexplore.ieee.org/document/10141580, setting up the forward model for rapid computation and initializing the reconstruction network is essential before proceeding to adversarial training. Please execute the code in the following sequence:

initFrwd.py - Initializes the forward model.
initG.py - Sets up the generative network.
mainLF2P.py - Commences the main adversarial training loop.


We drew inspiration for part of the adversarial component from the work of G. Kwon, C. Han, and D.-s. Kim in their paper, "Generation of 3D Brain MRI Using Auto-Encoding Generative Adversarial Networks," presented at the International Conference on Medical Image Computing. A special shoutout to the authors for sharing their code at https://github.com/cyclomon/3dbraingen !
