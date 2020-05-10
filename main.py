def main(config):
	from COBRA import Solver

	solver = Solver(config)
	cudnn.benchmark = True
	return solver.train()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--compute_all', type=bool, default=False)
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--just_valid', type=bool, default=False)

	parser.add_argument('--lr', type=list, default=[1e-4, 2e-4, 2e-4, 2e-4, 2e-4])
	parser.add_argument('--beta1', type=float, default=0.5)
	parser.add_argument('--beta2', type=float, default=0.999)
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--output_shape', type=int, default=512)
	parser.add_argument('--alpha', type=float, default=0.5)
	parser.add_argument('--beta', type=float, default=0.166)
	parser.add_argument('--gamma', type=float, default=0.166)
	parser.add_argument('--delta', type=float, default=0.166)
	parser.add_argument('--datasets', type=str, default='wiki_doc2vec') # xmedia, wiki_doc2vec, mscoco, gossip, politifact, nus_tfidf, metoo, crisis, crisis_damage
	parser.add_argument('--view_id', type=int, default=-1)
	parser.add_argument('--sample_interval', type=int, default=1)
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--num_negative_samples', type=int, default=25)
	parser.add_argument('--num_anchors', type=int, default=5)
	parser.add_argument('--use_nce', type=bool, default=False)

	config = parser.parse_args()

	seed = 123
	print('seed: ' + str(seed))
	import numpy as np
	np.random.seed(seed)
	import random as rn
	rn.seed(seed)
	import os
	os.environ['PYTHONHASHSEED'] = str(seed)
	import torch
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	from torch.backends import cudnn
	cudnn.enabled = False

	results = main(config)
