# Atari
OpenES and CEM code for Atari using Nature network, and analysis tools.

Based on OpenAI ES code from estool, CEM code from CEM-RL, Nature network from Canonical_ES_Atari.  
Tools vignette.py and gradient_study.py have been adapted from https://github.com/DevMaelFranceschetti/PAnd_Swimmer .  

## Requirements : 
python3.6  
gym 0.17.3  
gym['atari']  
atari-py 0.2.6  
cma  
numpy  
tensorflow 1.14.0  
matplotlib (gradient.py and vignette.py visualizations)  
PIL (gradient.py and vignette.py visualizations)  
ffmpeg (video output from monitor in viz.py)  

## Launch algorithms (examples) : 

	python3.6 main.py --algo CEM --env Qbert --nb_eval 10 --pop_size 15  

	python3.6 main.py --algo CEM --env Kangaroo --nb_eval 10 --pop_size 15  

	python3.6 main.py --algo OpenES --env Qbert --nb_eval 10 --pop_size 15  

## Use Vignette.py

You can check the comments in the code to understand all the parameters used.
Suppose you have a policy parameters file in a directory "my_parameters" beside the python code, and the policy parameters filename is "params1" (a pkl file), just run :

	python3.6 vignette.py --env Qbert --filename my_parameters --basename params --min_iter 1 --max_iter 1 --step_iter 1 

If you want to run vignette for files "params1" and "params5" for example, you can run :  

	python3.6 vignette.py --env Qbert --filename my_parameters --basename params --min_iter 1 --max_iter 5 --step_iter 4 
  
If you want to compute less directions around the parameters, you can tune the nb_lines parameter. Default is 50. You can also decrease the precision and the number of parameters tested around by increasing the stepalpha parameter. See an example :  

	python3.6 vignette.py --env Qbert --filename my_parameters --basename params --min_iter 1 --max_iter 1 --step_iter 1 --nb_lines 30 --stepalpha 2.5  

## Use Gradient_study.py

Parameters are quite similar for gradient_study.py :
Suppose you have 5 policy parameters files in a directory "my_parameters" beside the python code, and each pkl policy parameters filename is "params" followed by the iteration number (ex : params10, params20 ... params50).  
You can run :  

	python3.6 gradient_study.py --env Qbert --filename my_parameters --basename params --min_iter 10 --max_iter 50 --step_iter 10 --nb_lines



