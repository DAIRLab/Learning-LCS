# Learning Linear Complementarity Systems

The code is an implementation of the paper 
[Learning Linear Complementarity Systems](https://arxiv.org/abs/2112.13284),
coauthored by Wanxin Jin, Alp Aydinoglu, Mathew Halm, and Michael Posa. 
We have developed a new learning formulation to learn a linear complementarity system (LCS), which 
consists of a linear dynamics model and complementarity constraint. 


## Overview
Folder *lcs* is a python package, which includes 
* `lcs_learning.py`, which packs different  utility classes/functions necessary 
for learning and testing a LCS, 
particularly including a class `LCS_VN` for 
learning a LCS by the proposed violation-based method and a class `LCS_PN` for 
learning a LCS by the prediction-based method. For technical details of those two different
learning techniques, please refer to our paper. 

* `optim.py` is an implementation of different gradient-based techniques. 

Folder *evaluations*  contains different evaluations 
of the proposed violation-based learning method (`LCS_VN`)
in comparison with the prediction-based method (`LCS_PN`).

**RUN:** the quick way to start the code is to run each script in the _evaluations_ folder. 
All notations/variable used in the code follow the paper or conventions.


## Dependency
* CasADi:  for solving QP and autodiff. (>= 3.5.1. Info: https://web.casadi.org/), be sure to install `qpoases` plugin.
* Numpy:  for matrix computation. (>= 1.18.1. Info: https://numpy.org/)


## Citation
If you find this project/paper helpful in your research, please consider citing our paper.

    @misc{jin2021learning,
          title={Learning Linear Complementarity Systems}, 
          author={Wanxin Jin and Alp Aydinoglu and Mathew Halm and Michael Posa},
          year={2021},
          eprint={2112.13284},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

