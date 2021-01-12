import numpy as np
import matplotlib.pyplot as plt
import argparse
import cma
from es import OpenES, sepCEM
import gym
from policy import Policy # lots of warnings
import json
import time
import os
import pickle

def save_params(params, filename):
    """
        Save the given parameters in a pkl file
    """
    with open(filename, 'wb') as f:
        pickle.dump({"parameters": params}, f)

def save_vb(filename):
    """
        Save the currend virtual batch in a npy file
    """
    global policy
    vb = policy.vb
    np.save(filename+"_vb.npy", vb)

def run_solver(solver, max_iter):
    """
        Run the given solver, print and save logs
    """
    dir_name = "OpenES_"+str(time.time())
    os.mkdir(dir_name)
    history = []
    start_time = time.time()
    best_score = 0
    for j in range(max_iter):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            fitness_list[i] = fit_func(solutions[i])
        solver.tell(fitness_list)
        result = solver.result() # first element is the best solution, second element is the best fitness
        history.append(result[1])
        duration = time.time() - start_time
        f = open(dir_name+"/log.txt","a") # write log file (stats)
        f.write(str(j)+"\t"+str(duration)+"\t"+str(np.mean(fitness_list))+"\t"+str(np.min(fitness_list))+"\t"+str(np.max(fitness_list))+"\t"+str(result[1])+"\n")
        f.close()
        if (j+1) % 10 == 0 or result[1] > best_score:
            print("fitness at iteration", (j+1), result[1])
            print("time from start "+str(duration))
            print("mean fit : "+str(np.mean(fitness_list)))
            print("min fit : "+str(np.min(fitness_list)))
            print("max fit : "+str(np.max(fitness_list)))
            save_params(np.asarray(result[0]), dir_name+"/params"+(str(j+1))) # save solution
            save_vb(dir_name+"/"+(str(j+1))) # save corresponding virtual batch
            best_score = result[1]
    print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])
    return history

def eval_actor(parameters, nb_eval = 10):
    """
        Evaluate the given parameters (solution) and return the mean reward
    """
    global policy
    fit = []
    policy.set_parameters(parameters) # set the parameters to the policy
    for i in range(nb_eval):
        rew, steps = policy.rollout(False) # evaluate this policy parameters
        fit.append(rew)
    return np.mean(fit) # return the mean fitness

if __name__ == "__main__":  # lots of warning at start due to imports
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', default='Qbert', type=str)    # gym env
	parser.add_argument('--nb_eval', default=10, type=int)     # number of episodes for evaluation
	parser.add_argument('--max_iter', default=10000, type=int) # number of iterations of the algorithm
	parser.add_argument('--pop_size', default=16, type=int)    # number of iterations of the algorithm
	parser.add_argument('--sigma', default=0.6, type=float)    # sigma value for the optimizer
	parser.add_argument('--algo', default="OpenES", type=str)  # optimizer to run, choices : OpenES, CEM
	parser.add_argument('--elitism', default=True, type=bool)  # use elitism ?
	args = parser.parse_args()
	print("################################")
	print("Launched : "+str(args.algo)+", env "+args.env+", nb_eval "+str(args.nb_eval)+", sigma "+str(args.sigma))

	# create atari env :
	env_name = env_name = '%sNoFrameskip-v4' % args.env # use NoFrameskip game
	env = gym.make(env_name)
	fit_func = eval_actor  # defines fintess function (actor evaluation function)
	# create evaluation policy for parameters, based on the Nature network of CES (needs CES configuration file)
	with open("sample_configuration.json",'r') as f:
		configuration = json.loads(f.read())
	policy = Policy(env, network=configuration['network'], nonlin_name=configuration['nonlin_name'])
	vb = policy.get_vb() # init virtual batch
	nb_params = len(policy.get_parameters())

	if args.algo == "OpenES":
		# defines OpenAI's ES algorithm solver. Note that we needed to anneal the sigma parameter # pop 10
		optimizer = OpenES(nb_params,              # number of model parameters
			    sigma_init=0.6,                # initial standard deviation
			    sigma_decay=1,                 # annealing coefficient for standard deviation
			    learning_rate=0.1,             # learning rate for standard deviation
			    learning_rate_decay = 1,       # annealing coefficient for learning rate 
			    popsize=args.pop_size,         # population size
			    antithetic=False,              # whether to use antithetic sampling
			    weight_decay=0,                # weight decay coefficient
			    rank_fitness=False,            # use rank rather than fitness numbers
			    forget_best= not args.elitism) # cancel elitism ?
	elif args.algo == "CEM":
		optimizer = sepCEM(nb_params,
			    mu_init=None,
			    sigma_init=1e-3,
			    pop_size=args.pop_size,
			    damp=1e-3,
			    damp_limit=1e-5,
			    parents=None,
			    elitism=args.elitism,
			    antithetic=False)
	else:
		print("Optimizer is not available. Available : OpenES, CEM.")

	history = run_solver(optimizer, args.max_iter) # launching the algorithm
