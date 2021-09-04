import torch
import numpy as np
import json
import random
import scipy.sparse as ssp
import logging

def reltri2tri(trip_tasks, rel2id, ent2id):
	trip_data = []
	for key in trip_tasks.keys():
		rel_trip = trip_tasks[key]
		for j in range(len(rel_trip)):
			trip_data.append([ent2id[rel_trip[j][0]], rel2id[rel_trip[j][1]], ent2id[rel_trip[j][2]]])
	return trip_data

def h2hrt(train_id):
	h_hrt = {}
	for i in range(len(train_id)):
		if train_id[i][0] not in h_hrt.keys():
			h_hrt[train_id[i][0]] = []
		h_hrt[train_id[i][0]].append(train_id[i])
	return h_hrt


def trip2rel2tripid(trip_id_all,rel_all):
	rel2trip = {}
	for i in range(len(rel_all)):
		rel2trip[rel_all[i]] = []
	for j in range(len(trip_id_all)):
		trip_one = trip_id_all[j]
		rel2trip[trip_one[1]].append(trip_one)
	return rel2trip

def item2id(item, rel2id):
	item_id = []
	for i in range(len(item)):
		one = []
		for j in range(len(item[i])):
			one.append(rel2id[item[i][j]])
		item_id.append(one)
	return item_id


def path_read(path_dict_str, rel2id, ent2id):
	path_str = {}
	path_id = {}
	for key,item in path_dict_str.items():
		h,t = key.split('&')[0], key.split('&')[1]
		key_update = (h,t)
		key_update_id = (ent2id[h],ent2id[t])
		if len(item)!=0:
			item_id = item2id(item, rel2id)
		else:
			item_id = []
		path_str[key_update] = item
		path_id[key_update_id] = item_id
	return path_str,path_id

def test_relkind(rel_test_trip, train_test_path_id):
	test2relkind_dict = {}
	test2relkind = {}
	for key, item in rel_test_trip.items():
		test2relkind_dict[key] = []
		test2relkind[key] = []
		for i in range(len(item)):
			trip = item[i]
			pair = (trip[0],trip[2])
			path = train_test_path_id[pair]
			path_set = set()
			for m in range(len(path)):
				for n in range(len(path[m])):
					path_set.add(path[m][n])
			test2relkind_dict[key].append(pair)
			test2relkind[key].append(list(path_set))
	return test2relkind_dict, test2relkind


def set_rel_sim_count(num_rel_id): 
	sim_set = {}
	for key, item1 in num_rel_id.items(): 
		list_rel_all = item1
		set_rel_all = []
		for i in range(len(list_rel_all)):
			set_rel_all.append(set(list_rel_all[i])) 
		set_onerel_sim = []
		set_avg = []
		for setm in set_rel_all:
			set_one_sim = []
			for setn in set_rel_all:
				bing = list(set(setm) | set(setn))
				jiao = list(set(setm)&set(setn))
				if len(bing)!=0:
					set_one_sim.append(len(jiao)/len(bing))
				else:
					set_one_sim.append(0)
			set_onerel_sim.append(set_one_sim)
			sum_one = (sum(set_one_sim) - 1)/(len(set_one_sim)-1)
			set_avg.append(sum_one)
		sim_set[key] = set_avg
	return sim_set

def train_generate(sp_num, dataset_path, batch_size, train_tasks, ent2id, rel2id, id2ent, id2rel, e1rel_e2, rel2candidates):
	task_pool = list(train_tasks.keys()) 
	num_tasks = len(task_pool)
	rel_idx = 0
	while True:
		if rel_idx % num_tasks == 0:
			random.shuffle(task_pool)
		query = task_pool[rel_idx % num_tasks] 
		
		rel_idx += 1
		candidates = rel2candidates[query] 
		candidates_id = []
		for i in range(len(candidates)):
			candidates_id.append(ent2id[candidates[i]])

		if len(candidates) <= 20:
			continue

		rel_tt = train_tasks[query] 

		random.shuffle(rel_tt) 

		train_tri_id = [[ent2id[triple[0]], rel2id[triple[1]], ent2id[triple[2]]] for triple in rel_tt]

		train_tri_id_fil = []
		for trip in train_tri_id:
			if trip[0] != trip[2]:
				train_tri_id_fil.append(trip)
		train_tri_id = train_tri_id_fil

		support_pair = train_tri_id[:sp_num]
		query_pair = train_tri_id[sp_num:]
		
		if len(support_pair) == 0 or len(query_pair)==0:
			continue
		if len(query_pair) < batch_size:
			query_pair_pos = [random.choice(query_pair) for _ in range(batch_size)]
		else:
			query_pair_pos = random.sample(query_pair, batch_size)

		support_pair = [[pair[0],pair[2]] for pair in support_pair]
		query_pair_pos = [[pair[0],pair[2]] for pair in query_pair_pos]
		
		one_tomany_train = []
		for i in range(len(query_pair_pos)):
			one2many = e1rel_e2[id2ent[int(query_pair_pos[i][0])]+query]
			one2many2id = [ent2id[_] for _ in one2many]
			one_tomany_train.append(one2many2id)

		yield support_pair, query_pair_pos, one_tomany_train,candidates_id
		

def rel_submit(pair,train_test_path_id):
	rel_all = []
	for i in range(len(pair)):
		pair_one = (pair[i][0], pair[i][1])
		rel_list = train_test_path_id[pair_one]
		rel_set = set()
		for m in range(len(rel_list)):
			for n in range(len(rel_list[m])):
				rel_set.add(rel_list[m][n])
		rel_all.append(list(rel_set))
	return rel_all


def path_submit(pair, train_test_path_id):
	path_all = []
	for i in range(len(pair)):
		pair_one = (pair[i][0], pair[i][1])
		path_list = train_test_path_id[pair_one]
		path_all.append(path_list)
	return path_all

def pad_tensor(tensor: torch.Tensor, length, value=0, dim=0) -> torch.Tensor:
	return torch.cat(
		(tensor, tensor.new_full((*tensor.size()[:dim], length - tensor.size(dim), *tensor.size()[dim + 1:]), value)),
		dim=dim)

def list2tensor(data_list: list, padding_idx,  dtype=torch.long,  device=torch.device("cpu")):
	max_len = max(map(len, data_list))
	max_len = max(max_len, 1)
	data_tensor = torch.stack(
		tuple(pad_tensor(torch.tensor(data, dtype=dtype), max_len, padding_idx, 0) for data in data_list)).to(device)
	return data_tensor

