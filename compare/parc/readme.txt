--------------------------------------------------
PARC - Piecewise Affine Regression and Classification.

(C) 2021 by A. Bemporad
--------------------------------------------------

This package implements an algorithm in PYTHON to solve multivariate
regression and classification problems using piecewise linear predictors 
over a polyhedral partition of the feature space. 

The algorithm is based on the following paper:

[1] A. Bemporad, “Piecewise Linear Regression and Classification,”
https://arxiv.org/abs/2103.06189, 2021.

This software is distributed without any warranty. Please cite the above paper 
if you use this software.


Version tracking:
v1.1     (March 15, 2021) Added feature_selection option in fit method, to create
                          partitions in reduced feature space.
v1.0     (March 10, 2021) First public release.
