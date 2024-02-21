# Physics-based Deep Learning for Imaging Neuronal Activity via Two-photon and Light Field Microscopy

Welcome to the code repository for our paper, "Physics-based Deep Learning for Imaging Neuronal Activity via Two-photon and Light Field Microscopy", available at [IEEE Xplore](https://ieeexplore.ieee.org/document/10141580) and [BioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.11.511633v1.full.pdf). Our work introduces a deep learning framework designed for imaging neuronal activity, combining the fast 3D imaging capabilities of Light Field Microscopy (LFM) with the high-resolution strengths of Two-photon (2P) microscopy.

To implement this framework in your research, start with the following scripts:

- `initFrwd.py` - Prepares the forward model.
- `initG.py` - Establishes the generative network.
- `mainLF2P.py` - Begins the adversarial training loop.
  
If you use this code in your research, please cite our paper [IEEE Xplore](https://ieeexplore.ieee.org/document/10141580)

Part of our adversarial approach was inspired by "Generation of 3D Brain MRI Using Auto-Encoding Generative Adversarial Networks" by G. Kwon, C. Han, and D.-s. Kim, whose code is available [here](https://github.com/cyclomon/3dbraingen).
