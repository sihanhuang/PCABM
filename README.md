# PCABM

This is the sample code for conducting pairwise covariates-adjusted stochastic block model (PCABM), which is a generalization of SBM that incorporates pairwise covariate information.

Simulation Example.ipynb is a a toy example illustrating how to use pcabm package, with data generating from a pcabm. Political Blog Example.ipynb is applying mle of pcabm to a famous real world data set. 

Followings are introductions to pcabm's files:

- commFunc.py : some common functions that will be used often.
- dcbm.py     : calculating mle for degree corrected block model
- pcabm.py    : calculating mle for pairwise covariates-adjusted stochastic block model
- sc.py       : different spectral clustering methods

In PCABM/plem_and_ecv_matlab_code/cpl_m_code/, Matlab codes for pseudo likelihood EM (PLEM) and spectral clustering with adjustment are in the files CA_plEM.m and CA_SCWA.m; 
In PCABM/plem_and_ecv_matlab_code/, ecv_chooseK.m and ecv_variableselect6.m are the codes for selecting number of communities and feature selection.
