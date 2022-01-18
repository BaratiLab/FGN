# FGN-code
Code repo for reproduce FGN

### Requirement
* PyTorch 1.7.1, DGL 0.7.0, PyTorch Geometric 1.7.0, CuPy (latest version), Numpy, Scipy, Scikit-learn & Numba
* FRNN from: https://github.com/lxxue/FRNN

### Structure
The implementation of MPS is included in Particles.py, cuHelper.py </br>
The pretrained weights are available at "./training" folder. </br>
Grid used to benchmark and generate data are available at "./grid" folder. </br>
The example simulation script is in the "./running_script" folder: for example: sh gnn_simulate.sh
