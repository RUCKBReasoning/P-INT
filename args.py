import argparse
import os

def read_options():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="NELL-One", type=str) # FB15k237-One || NELL-One
	parser.add_argument("--embed_dim", default=100, type=int)
	parser.add_argument("--hidden_dim", default=100, type=int)
	parser.add_argument("--mode",default='test',type=str)
	parser.add_argument("--few", default=1, type=int)
	parser.add_argument("--batch_size", default=32, type=int) 
	parser.add_argument("--random_embed", action='store_true') 
	parser.add_argument("--lr", default=0.0005, type=float) 
	parser.add_argument("--margin", default=64.0, type=float)
	parser.add_argument("--max_batches", default=200000, type=int)
	parser.add_argument("--dropout", default=0.2, type=float)
	parser.add_argument("--log_every", default=100, type=int)
	parser.add_argument("--test", action='store_true')
	parser.add_argument("--grad_clip", default=5.0, type=float)
	parser.add_argument("--weight_decay", default=0.0, type=float)
	parser.add_argument("--embed_model", default='TransE', type=str)
	parser.add_argument("--prefix", default='intial', type=str)
	parser.add_argument("--seed", default='88', type=int)
	parser.add_argument("--epoch", default=3, type=int)

	parser.add_argument("--hop", default=3, type=int)
	parser.add_argument("--max_neighbor", default=200, type=int)
	parser.add_argument("--eval_every", default=100, type=int) #100

	parser.add_argument("--neighbor_limit",default=300,type=int)
	parser.add_argument("--topk",default=100,type=int)
	parser.add_argument("--kernel_num",default=21,type=int)


	args = parser.parse_args()

	if not os.path.exists('models/'):
		os.mkdir('models/')
	args.save_path = 'models/' + args.dataset + str(args.few) + args.prefix

	print("------HYPERPARAMETERS-------")
	for k, v in vars(args).items():
		print(k + ': ' + str(v))
	print("----------------------------")

	return args

