import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse
import gym
import pickle
import json
import time
from es import SimpleGA, CMAES, PEPG, OpenES
from policy import Policy # lots of warning ^^

def eval_actor(policy, nb_eval=1):
    fit = []
    for i in range(nb_eval):
        rew, steps = policy.rollout(False) # evaluate this policy parameters
        fit.append(rew)
    if np.mean(fit) > 2000 : 
        params = policy.get_parameters()
        save_params(params, "saved_params_"+str(np.mean(fit))+"_"+str(time.time()))
        save_vb(policy.vb, "saved_vb_"+str(np.mean(fit))+"_"+str(time.time()))
    return np.mean(fit)

def save_params(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump({"parameters": params}, f)

def save_vb(vb, filename):
    np.save(filename+".npy", vb)

def load_params(filename):
	with open(filename, 'rb') as f:
		params = pickle.load(f)['parameters']
	return params

def getPointsChoice(init_params,num_params, minalpha, maxaplha, stepalpha, prob):
	"""
	# Params :
	init_params : actor parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	prob : the probability to choose each parameter dimension (float)
	
	# Function:
	Returns parameters around base_params on direction choosen by random choice of proba 'prob' on param dimensions.
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives a good but very noisy visualisation and not easy to interpret.
	"""
	#init_params = np.copy(base_params)
	d = np.random.choice([1, 0], size=(num_params,), p=[prob, 1-prob]) #select random dimensions with proba 
	print("d: "+str(d))
	print("proportion: "+str(np.count_nonzero(d==1))+"/"+str(num_params))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsUniform(init_params,num_params, minalpha, maxaplha,stepalpha):
	"""
	# Params :
	init_params : actor parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	
	# Function:
	Returns parameters around base_params on direction choosen by uniform random draw on param dimensions in [0,1).
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives the best visualisation.
	"""
	#init_params = np.copy(base_params)
	d = np.random.uniform(0, 1, num_params) #select uniformly dimensions [0,1)
	print("d: "+str(d))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsDirection(init_params,num_params, minalpha, maxaplha,stepalpha, d):
	"""
	# Params :
	init_params : actor parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	d : pre-choosend direction
	
	# Function:
	Returns parameters around base_params on direction given in parameters.
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha.
	This method gives an output that is comparable with other results if directions are the same.
	"""
	#init_params = np.copy(base_params)
	print("d: "+str(d))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getPointsUniformCentered(init_params,num_params, minalpha, maxaplha,stepalpha):
	"""
	# Params :
	init_params : actor parameters to study around (array)
	num_params : the length of the parameters array (int)
	minalpha : the start value for alpha parameter (float)
	maxalpha : the end/highest value for alpha parameter (float)
	stepalpha : the step for alpha value in the loop (float)
	
	# Function:
	Returns parameters around base_params on direction choosen by uniform random draw on param dimensions in [-1,1).
	Parameters starts from base_params to base_params+maxalpha on one side of the direction and
	from base_params to base_params-maxaplha on the other side. The step of alpha is stepalpha. 
	This method gives bad visualisation.
	"""
	#init_params = np.copy(base_params)
	d = np.random.uniform(-1, 1, num_params) #select uniformly dimensions in [-1,1)
	print("d: "+str(d))
	theta_plus = []
	theta_minus = []
	for alpha in np.arange(minalpha, maxaplha, stepalpha):
		theta_plus.append(init_params + alpha * d)
		theta_minus.append(init_params - alpha * d)
	return theta_plus, theta_minus #return separaterly points generated around init_params on each side (+/-)

def getDirectionsMuller(nb_directions,num_params):
    """
    # Params :
    nb_directions : number of directions to generate randomly in unit ball
    num_params : dimensions of the vectors to generate (int value, only 1D vectors)
	
    # Function:
    Returns a list of vectors generated in the uni ball of 'num_params' dimensions, using Muller
    """
    D = []
    for _ in range(nb_directions):
        u = np.random.normal(0,1,num_params)
        norm = np.sum(u**2)**(0.5)
        r = np.random.random()**(1.0/num_params)
        x = r*u/norm
        print("vect muller:"+str(x))
        print("euclidian dist:"+str(euclidienne(x, np.zeros(len(x)))))
        D.append(x)
    return D

def euclidienne(x,y):
    """
    # Params :
    x,y : vectors of the same size
	
    # Function:
    Returns a simple euclidian distance between x and y.
    """
    return np.linalg.norm(np.array(x)-np.array(y))

def order_all_by_proximity(vectors):
    """
    # Params :
    vectors : a list of vectors
	
    # Function:
    Returns the list of vectors ordered by inserting the vectors between their nearest neighbors
    """
    ordered = []
    for vect in vectors :
        if(len(ordered)==0):
            ordered.append(vect)
        else:
            ind = compute_best_insert_place(vect, ordered)
            ordered.insert(ind,vect)
    return ordered

def compute_best_insert_place(vect, ordered_vectors):
    """
    # Params :
    ordered_vectors : a list of vectors ordered by inserting the vectors between their nearest neighbors
    vect : a vector to insert at the best place in the ordered list of vectors
	
    # Function:
    Returns the index where 'vect' should be inserted to be between the two nearest neighbors using euclidien distance
    """
    # Compute the index where the vector will be at the best place :
    value_dist = euclidienne(vect, ordered_vectors[0])
    dist_place = [value_dist]
    for ind in range(len(ordered_vectors)-1):
        value_dist = np.mean([euclidienne(vect, ordered_vectors[ind]),euclidienne(vect, ordered_vectors[ind+1])])
        dist_place.append(value_dist)
    value_dist = euclidienne(vect, ordered_vectors[len(ordered_vectors)-1])
    dist_place.append(value_dist)
    ind = np.argmin(dist_place)
    return ind
	
if __name__ == "__main__":

	print("Parsing arguments")

	parser = argparse.ArgumentParser()
	parser.add_argument('--env', default='Qbert', type=str)
	parser.add_argument('--nb_lines', default=50, type=int)# number of directions generated,good value : precise 100, fast 60, ultrafast 50
	parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
	parser.add_argument('--maxalpha', default=150, type=float)# end value for alpha, good value : large 100, around actor 10
	parser.add_argument('--stepalpha', default=5, type=float)# step for alpha in the loop, good value : precise 0.5 or 1, less precise 2 or 3
	parser.add_argument('--eval_maxiter', default=25000, type=float)# number of steps for the evaluation. Depends on environment.
	parser.add_argument('--nb_eval', default=2, type=int)# number of evaluation to compute fitness (mean)
	parser.add_argument('--min_colormap', default=0, type=int)# min score value for colormap used (depend of benchmark used)
	parser.add_argument('--max_colormap', default=1500, type=int)# max score value for colormap used (depend of benchmark used)
	parser.add_argument('--proba', default=0.1, type=float)# proba of choosing an element of the actor parameters for the direction, if using the choice method.
	parser.add_argument('--basename', default="params", type=str)# base (files prefix) name of the actor pkl files to load
	parser.add_argument('--min_iter', default=150, type=int)# iteration (file suffix) of the first actor pkl files to load
	parser.add_argument('--max_iter', default=150, type=int)# iteration (file suffix) of the last actor pkl files to load
	parser.add_argument('--step_iter', default=1, type=int)# iteration step between two consecutive actor pkl files to load
	parser.add_argument('--base_output_filename', default="vignette_output_CEM_new", type=str)# name of the output file to create
	parser.add_argument('--filename', default="CEM_1610033651.994504", type=str)# name of the directory containing the actors pkl files to load

	args = parser.parse_args()

	# Creating environment and initialising actor and parameters
	print("Creating environment")
	game = args.env #"Qbert"
	env_name = env_name = '%sNoFrameskip-v4' % game # use NoFrameskip game, as in CES
	env = gym.make(env_name)

	# neural network
	print("creating neural network")
	with open("sample_configuration.json",'r') as f:
		configuration = json.loads(f.read())
	actor = Policy(env, network=configuration['network'], nonlin_name=configuration['nonlin_name'])
	actor.max_episode_len = args.eval_maxiter	
	theta0 = actor.get_parameters()
	num_params = len(theta0)
	v_min_fit = args.min_colormap
	v_max_fit = args.max_colormap
	print("VMAX :"+str(v_max_fit))

	# Choosing directions
	#D = np.random.rand(args.nb_lines,num_params)
	D = getDirectionsMuller(args.nb_lines,num_params)

	# Ordering the directions :
	D = order_all_by_proximity(D)

	# Name of the actor files to analyse consecutively with the same set of directions: 
	filename_list = [args.basename+str(i) for i in range(args.min_iter,args.max_iter+args.step_iter,args.step_iter)]# generate actor files list to load
	vb_list = [str(i)+"_vb.npy" for i in range(args.min_iter,args.max_iter+args.step_iter,args.step_iter)]# generate virtual batch files list to load
	# Compute fitness over these directions :
	last_actor_params = [] #save the last parameters, to compute direction followed from the previous actor
	for indice_file in range(len(filename_list)):
		filename = filename_list[indice_file]
		vb_file = vb_list[indice_file]
		vb = np.load(args.filename+"/"+vb_file)
		# Loading actor params
		print("STARTING : "+str(filename))
		params = load_params(args.filename+"/"+filename)
		actor.set_parameters(params)
		actor.set_vb(vb)
		theta0 = params #actor.get_parameters()
		if(len(last_actor_params)>0):
			previous = last_actor_params
			base_vect = theta0 - previous #compute direction followed from the previous actor
			last_actor_params = theta0 #update last_actor_params
		else:
			base_vect = theta0 #if first actor (no previous), considering null vector is the previous
			last_actor_params = theta0 #update last_actor_params
		print("params : "+str(theta0))
		# evaluate the actor
		init_score = eval_actor(actor, nb_eval=args.nb_eval) # evaluate the initial actor
		print("Actor initial fitness : "+str(init_score))
		# Running geometry study around the actor
		print("Starting study around...")
		theta_plus_scores = []
		theta_minus_scores = []
		image = []
		base_image = []
		
		### Direction followed from precedent actor :
		length_dist = euclidienne(base_vect, np.zeros(len(base_vect)))
		d= base_vect / length_dist #reduce to unit vector
		theta_plus, theta_minus = getPointsDirection(theta0,num_params, args.minalpha, args.maxalpha, args.stepalpha, d)
		temp_scores_theta_plus = []
		temp_scores_theta_minus = []
		start_time = time.time()
		for param_i in range(len(theta_plus)):
			# we evaluate the actor (theta_plus) :
			actor.set_parameters(theta_plus[param_i])
			score_plus = eval_actor(actor,nb_eval=args.nb_eval)
			temp_scores_theta_plus.append(score_plus)
			# we evaluate the actor (theta_minus) :
			actor.set_parameters(theta_minus[param_i])
			score_minus = eval_actor(actor,nb_eval=args.nb_eval)
			temp_scores_theta_minus.append(score_minus)
		print("duration : "+str(time.time()-start_time))
		#we invert scores on theta_minus list to display symetricaly the image with init params at center,
		# theta_minus side on the left and to theta_plus side on the right
		buff_inverted = np.flip(temp_scores_theta_minus)
		plot_pixels = np.concatenate((buff_inverted,[init_score],temp_scores_theta_plus))
		base_image.append(plot_pixels)#adding these results as a line in the output image
		#saving the score values
		theta_plus_scores.append(temp_scores_theta_plus)
		theta_minus_scores.append(temp_scores_theta_minus)
		### Directions chosen
		for step in range(len(D)) :
			start_time = time.time()
			print("step "+str(step)+"/"+str(len(D)))
			#computing actor parameters
			d = D[step]
			theta_plus, theta_minus = getPointsDirection(theta0,num_params, args.minalpha, args.maxalpha, args.stepalpha, d)
			temp_scores_theta_plus = []
			temp_scores_theta_minus = []
			for param_i in range(len(theta_plus)):
				# we evaluate the actor (theta_plus) :
				actor.set_parameters(theta_plus[param_i])
				score_plus = eval_actor(actor,nb_eval=args.nb_eval)
				print("score plus : "+str(score_plus))
				temp_scores_theta_plus.append(score_plus)
				# we evaluate the actor (theta_minus) :
				actor.set_parameters(theta_minus[param_i])
				score_minus = eval_actor(actor,nb_eval=args.nb_eval)
				print("score minus : "+str(score_minus))
				temp_scores_theta_minus.append(score_minus)
			#we invert scores on theta_minus list to display symetricaly the image with init params at center,
			# theta_minus side on the left and to theta_plus side on the right
			buff_inverted = np.flip(temp_scores_theta_minus)
			plot_pixels = np.concatenate((buff_inverted,[init_score],temp_scores_theta_plus))
			image.append(plot_pixels)#adding these results as a line in the output image
			#saving the score values
			theta_plus_scores.append(temp_scores_theta_plus)
			theta_minus_scores.append(temp_scores_theta_minus)
			print("duration : "+str(time.time()-start_time)+" s")
		#assemble picture from different parts (choosen directions, dark line for separating, and followed direction)
		separating_line = np.zeros(len(base_image[0]))
		last_params_marker = int(length_dist/args.stepalpha)
		marker_pixel = min(len(separating_line), max(0,int((len(base_image[0])-1)/2-last_params_marker)))
		separating_line[marker_pixel] = v_max_fit
		final_image = np.concatenate((image, [separating_line], base_image), axis=0)
		#showing final result
		final_image = np.repeat(final_image,10,axis=0)#repeating each line 10 times to be visible
		final_image = np.repeat(final_image,10,axis=1)#repeating each line 10 times to be visible
		plt.imsave(args.base_output_filename+"_"+str(filename)+".png",final_image, vmin=v_min_fit, vmax=v_max_fit, format='png')

	env.close()
