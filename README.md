# Fluid Graph Network
Code repo for FGN

### Requirement
* PyTorch 1.7.1, DGL 0.7.0, PyTorch Geometric 1.7.0, CuPy (latest version), Numpy, Scipy, Scikit-learn & Numba
* Pandas, Partio (https://github.com/wdas/partio) for I/O
* FRNN from: https://github.com/lxxue/FRNN
* The code is tested under Linux Ubuntu 18.04 with CUDA 10.2.

### Structure
SPH-based data structure and MPS solving scheme are included in Particles.py, cuHelper.py, run_physics_model.py </br>
The pretrained weights are available at "./training" folder. </br>
Grid used to benchmark and generate data are available at "./grid" folder. </br>

### Usage
* The train and test data can be generated using generate_data.py in "./training" folder: 
 ```
 cd training
 python generate_data.py
 ``` 
* To run the simulation using pretrained GNN:
The example simulation script is in the "./running_script" folder. </br>
```
cd running_script
sh gnn_simulate.sh
``` 

### Related resources
* Conjugate gradient solver in Pytorch: https://github.com/sbarratt/torch_cg
* Continuous convolution for end-to-end position based fluid simulation: https://github.com/isl-org/DeepLagrangianFluids
* SPH library: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH


