# Fluid Graph Network
Code repo for FGN

### Requirement
* PyTorch 1.7.1, DGL 0.7.0, PyTorch Geometric 1.7.0, CuPy (latest version), Numpy, Scipy, Scikit-learn & Numba
* Pandas, Partio (https://github.com/wdas/partio) for I/O
* FRNN from: https://github.com/lxxue/FRNN

### Structure
TSPH's data structure and solving scheme is included in Particles.py, cuHelper.py </br>
The pretrained weights are available at "./training" folder. </br>
Grid used to benchmark and generate data are available at "./grid" folder. </br>

### Usage
* The train and test data can be generated using generate_data.py in "./training" folder: 
 ```
 cd training
 python generate_data.py
 ``` </br>
* To run the simulation using pretrained GNN:
The example simulation script is in the "./running_script" folder. </br>
```
cd running_script
sh gnn_simulate.sh
``` </br>

