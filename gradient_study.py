import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse
import gym
import pickle
import json
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from es import SimpleGA, CMAES, PEPG, OpenES
from policy import Policy # lots of warning ^^

def eval_actor(policy, nb_eval=1, save_if_greater=False, save_greater_than=2000):
    fit = []
    for i in range(nb_eval):
        rew, steps = policy.rollout(False) # evaluate this policy parameters
        fit.append(rew)
    if save_if_greater and np.mean(fit) > save_greater_than : 
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

	# About loading the actors files :
	#   The actor filenames we use have a common prefix (actor_td3_1_step_1 for example) followed by the number of iterations.
	#   example : actor_td3_1_step_1_1000, actor_td3_1_step_1_25000, or myactor_200000. 
	#   Check parameters 'basename', 'min_iter, 'max_iter' and step_iter.
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', default='Qbert', type=str)
	parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
	parser.add_argument('--maxalpha', default=110, type=float)# end value for alpha, good value : 100
	parser.add_argument('--stepalpha', default=2, type=float)# step for alpha in the loop, good value : 0.5
	parser.add_argument('--eval_maxiter', default=25000, type=float)# number of steps for the evaluation.
	parser.add_argument('--min_colormap', default=0, type=int)# min score value for colormap used (depend of benchmark used)
	parser.add_argument('--nb_eval', default=3, type=int)# number of evaluations for each parameters
	parser.add_argument('--max_colormap', default=1500, type=int)# max score value for colormap used (depend of benchmark used)
	parser.add_argument('--filename', default="CEM_1610033651.994504", type=str)# name of the directory containing the actors pkl files to load
	parser.add_argument('--basename', default="params", type=str)# base (files prefix) name of the actor pkl files to load
	parser.add_argument('--min_iter', default=10, type=int)# iteration (file suffix) of the first actor pkl files to load
	parser.add_argument('--max_iter', default=630, type=int)# iteration (file suffix) of the last actor pkl files to load
	parser.add_argument('--step_iter', default=1, type=int)# iteration step between two consecutive actor pkl files to load (increment the file suffix)
	parser.add_argument('--output_filename', default="gradient_output_CEM.png", type=str)# name of the output file to create
	parser.add_argument('--save_greater', default=False, type=bool)# save parameters found greater than greater_than param
	parser.add_argument('--greater_than', default=5000, type=float)# name of the output file to create
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

	# Name of the actor files to analyse consecutively with the same set of directions: 
	filename_list = [args.basename+str(i) for i in range(args.min_iter,args.max_iter+args.step_iter,args.step_iter)]# generate actor files list to load
	vb_list = [str(i)"_vb.npy" for i in range(args.min_iter,args.max_iter+args.step_iter,args.step_iter)]# generate virtual batch files list to load
	# modif id
	image_filename = args.output_filename #output picture  
	# Compute fitness over these directions :
	last_actor_params = [] #s ave the last parameters, to compute direction followed from the precedent actor
	result = []
	directions = []
	dot_values = []
	yellow_markers = []
	red_markers = []
	for indice_file in range(len(filename_list)): # modif id
		file_start_time = time.time()
		filename = filename_list[indice_file]
		vb_filename = vb_list[indice_file]
		vb = np.load(args.filename+"/"+vb_filename)
		# Loading actor params
		print("FILE : "+str(filename)+" ("+str(indice_file+1)+"/"+str(len(filename_list))+")")
		params = load_params(args.filename+"/"+filename)
		actor.set_parameters(params)
		actor.set_vb(vb)
		theta0 = params
		if(len(last_actor_params)>0):
			previous = last_actor_params
			base_vect = theta0 - previous # compute direction followed from the precedent actor
			last_actor_params = theta0 # update last_actor_params
		else:
			base_vect = theta0 # if first actor (no precedent), considering null vector is the precedent
			last_actor_params = theta0 # update last_actor_params

		print(" - params : "+str(theta0))
		# evaluate the actor
		init_score = eval_actor(actor, nb_eval=args.nb_eval)
		print(" - Actor initial fitness : "+str(init_score))
		# Running geometry study around the actor
		theta_plus_scores = []
		theta_minus_scores = []
		base_image = []
		
		### Direction followed from precedent actor :
		length_dist = euclidienne(base_vect, np.zeros(len(base_vect)))
		print(" - length_dist : "+str(length_dist))
		if length_dist != 0 :
			d= base_vect / abs(length_dist) # reduce to unit vector
		else:
			d = np.zeros(len(base_vect))
		directions.append(d*5)# save unity vector of estimated gradient direction
		theta_plus, theta_minus = getPointsDirection(theta0,num_params, args.minalpha, args.maxalpha, args.stepalpha, d)
		temp_scores_theta_plus = []
		temp_scores_theta_minus = []
		for param_i in range(len(theta_plus)):
			# we evaluate the actor (theta_plus) :
			actor.set_parameters(theta_plus[param_i])
			score_plus = eval_actor(actor, nb_eval=args.nb_eval, save_if_greater=args.save_greater, save_greater_than = args.greater_than)
			temp_scores_theta_plus.append(score_plus)
			# we evaluate the actor (theta_minus) :
			actor.set_parameters(theta_minus[param_i])
			score_minus = eval_actor(actor, nb_eval=args.nb_eval, save_if_greater=args.save_greater, save_greater_than = args.greater_than)
			temp_scores_theta_minus.append(score_minus)
		# we invert scores on theta_minus list to display symetricaly the image with init params at center,
		# theta_minus side on the left and to theta_plus side on the right
		buff_inverted = np.flip(temp_scores_theta_minus)
		plot_pixels = np.concatenate((buff_inverted,[init_score],temp_scores_theta_plus))
		# saving the score values
		theta_plus_scores.append(temp_scores_theta_plus)
		theta_minus_scores.append(temp_scores_theta_minus)

		
		# assemble picture from different parts (choosen directions, dark line for separating, and followed direction)
		mean_value = (v_max_fit-v_min_fit)/2+v_min_fit
		separating_line = np.array([v_min_fit]*len(plot_pixels))
		last_params_marker = int(length_dist/args.stepalpha)
		if last_params_marker < 0 :
			marker_last = min( int((len(plot_pixels)-1)/2+last_params_marker) , len(plot_pixels)-1)
		else:
			marker_last = max( int((len(plot_pixels)-1)/2-last_params_marker), 0)
		marker_actor = int((len(plot_pixels)-1)/2)
		yellow_markers.append(marker_actor)
		red_markers.append(marker_last)
		separating_line[marker_last] = mean_value # previous actor in blue (original version, modified by red markers below)
		separating_line[marker_actor] = v_max_fit # current actor in yellow
		result.append(separating_line)# separating line
		result.append(plot_pixels) # adding multiple time the same entry, to better see
		result.append(plot_pixels)
		result.append(plot_pixels)
		print("FILE DURATION : "+str(time.time()-file_start_time)+" s")
	
	# preparing final result
	final_image = np.repeat(result,10,axis=0)# repeating each 10 times to be visible
	final_image = np.repeat(final_image,20,axis=1)# repeating each 20 times to be visible

	plt.imsave(image_filename,final_image, vmin=v_min_fit, vmax=v_max_fit, format='png')

	env.close()

	# adding dot product & markers visual infos : 
	im = Image.open(image_filename)
	width, heigh = im.size
	output = Image.new("RGB",(width+170, heigh))
	for y in range(heigh):
		for x in range(width):
			output.putpixel((x,y),im.getpixel((x,y)))
	for nb_m in range(len(red_markers)):# red markers
		for y in range(10):
			for x in range(20):
				output.putpixel((x+20*red_markers[nb_m],y+nb_m*40),(255,0,0))

	for i in range(0,len(filename_list)-1):# dot product values # modif id
		scalar_product = np.dot(directions[i], directions[i+1])
		print("dot prod : "+str(scalar_product))
		color = (0,255,0)
		if(scalar_product < -0.2):
			color = (255,0,0)
		else :
			if (scalar_product >-0.2 and scalar_product < 0.2):
				color = (255,140,0)
		for j in range(min(int(10*abs(scalar_product))+10,150)):
			for k in range(1,20):
				output.putpixel((width+j+10,i*40+30+k),color)

	# saving result
	output.save(image_filename,"PNG")

