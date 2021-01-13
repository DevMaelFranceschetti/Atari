# Atari
OpenES and CEM code for Atari using Nature network, and analyse tools

Based on OpenAI ES code from estool, CEM code from CEM-RL and Nature network from Canonical_ES_Atari

## Requirements : 
python3.6
gym 0.17.3
gym['atari']
atari-py 0.2.6
cma
numpy
tensorflow 1.14.0
matplotlib (gradient.py and vignette.py visualizations)
PIL (gradient.py and vignette.pu visualizations)
ffmpeg (video output from monitor in viz.py)

## Launch algorithms (examples) : 

python3.6 main.py --algo CEM --env Qbert --nb_eval 10 --pop_size 15

python3.6 main.py --algo OpenES --env Qbert --nb_eval 10 --pop_size 15



