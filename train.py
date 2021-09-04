import logging
import numpy as np
import torch
import random
from torch import optim
from torch.autograd import Variable
from args import read_options
from trainer import *

def main(params):
	trainer = Trainer(params)
	if params.test:
		trainer.test_() # test
	else:
		trainer.train() # train


if __name__ == '__main__':
	args = read_options()

	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)

	formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

	log_path = './logs_/log-{}.txt'.format(args.prefix)

	if not os.path.exists('./logs_'):
		os.mkdir('./logs_')

	fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix)) 
	fh.setLevel(logging.DEBUG) 
	fh.setFormatter(formatter) 
	
	ch = logging.StreamHandler() 
	ch.setLevel(logging.INFO) 
	ch.setFormatter(formatter)

	logger.addHandler(ch)
	logger.addHandler(fh)

	# setup random seeds
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed) 
	torch.cuda.manual_seed_all(args.seed) 

	main(args)

	