# PCABM

This is the sample code for conducting pairwise covariates-adjusted stochastic block model (PCABM), which is a generalization of SBM that incorporates pairwise covariate information.

Simulation Example.ipynb is a a toy example illustrating how to use pcabm package, with data generating from a pcabm. Political Blog Example.ipynb is applying mle of pcabm to a famous real world data set. 

Followings are introductions to pcabm's files:

- commFunc.py : some common functions that will be used often.
- dcbm.py     : degree corrected block model
- pcabm.py    : pairwise covariates-adjusted stochastic block model using tabu search
- sc.py       : different spectral clustering methods
- ecv.py      : using edge cross validation to choose K and covariates
- plem.py     : pairwise covariates-adjusted stochastic block model using pseudo likelihood

To replicate simulation results, people could run script in 'simulations' folder. The name is the figure number in the paper. To run a single simulation, use
<pre><code>python filename.py</code></pre> 
To run multiple simulations, use
<pre><code>sbatch filename.sh</code></pre> 

To save the simulation results, you need to make a folder './output/filename', or anywhere else you'd like. To aggregate simulation results, use corresponding commented command in filename.sh. 