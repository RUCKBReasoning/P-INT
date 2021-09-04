"""
build background knowledge graph
define some methods over KB:
# extract paths between the head and tail
# extract enclosing subgraph of the head and tail
"""

import os
import dgl
import json
import torch
import numpy as np
import networkx as nx
from itertools import product
from collections import defaultdict, Counter

#from data_utils import id2name

def id2name(name2id):
	
	return dict(zip(name2id.values(), name2id.keys()))


class BGGraph(object):

	def __init__(self, args):
		
		self.data_dir        = os.path.join(args["data_dir"], args["dataset_name"])

		self.ent2id          = json.load(open(os.path.join(self.data_dir, 'entity2id.json')))
		self.id2ent          = id2name(self.ent2id)
		self.rel2id          = json.load(open(os.path.join(self.data_dir, 'relation2id.json')))
		self.id2rel          = id2name(self.rel2id)

		self.out             = json.load(open(os.path.join(self.data_dir, 'path_graph_out.json')))


	def two_wise_bfs(self, pair):
		'''
		param: pair: the head entity and tail entity must be string (str_h, str_t)
		'''
		start,end = pair

		# first-order neighbor
		start_FN, end_FN = set(), set()
		s_f_path_tracker, e_f_path_tracker = defaultdict(list), defaultdict(list)

		if start in self.out.keys():
			for edge in self.out[start]:
				start_FN.add(edge[1])
				path = []
				path.append(edge)
				s_f_path_tracker[edge[1]].append(path)

		if end in self.out.keys():
			for edge in self.out[end]:
				end_FN.add(edge[1])
				path = []
				path.append(edge)
				e_f_path_tracker[edge[1]].append(path)

		# second-order neighbor
		start_SN = set()
		s_s_path_tracker = defaultdict(list)

		for node in start_FN:
			paths = s_f_path_tracker[node]
			if str(node) in self.out.keys():
				for edge in self.out[str(node)]:
					start_SN.add(edge[1])
					for path in paths:
						#path.append(edge) !!! wrong! it will change s_f_path_tracker
						s_s_path_tracker[edge[1]].append(path+[edge])
			else:
				continue
		
		return start,end,start_FN,start_SN,end_FN,s_f_path_tracker,s_s_path_tracker,e_f_path_tracker


	def get_inverse_relation(self, r_id):
		"""
		when combining the paths from both sides,
		the path starting from the end needs to be inversed,
		not only the order of the entities and relations
		but also the relation name, i.e., the 'rel' needs to be 'rel_inv', the 'rel_inv' needs to be 'rel'
		"""
		
		rel = self.id2rel[r_id]
		if '_inv' == rel[-4:]:
			rel = rel[:-4]
		else:
			rel = rel + '_inv'

		return self.rel2id[rel]


	def combine_two_path(self,start,end,intermediate,left_path_tracker,right_path_tracker):
		ans_path = []

		for mid in intermediate:
			two_sides = product(left_path_tracker[mid], right_path_tracker[mid])
			for left_side, right_side in two_sides:
				left_path = []
				for r,e in left_side:
					left_path.append(str(r))
					left_path.append(str(e))
				left_path = start + ' - ' + ' - '.join(left_path)

				if len(right_side)==0:
					one_path = left_path
				else:
					right_path = []
					for r,e in right_side:
						right_path.append(str(self.get_inverse_relation(r)))
						right_path.append(str(e))
					right_path = right_path[::-1]
					right_path = ' - '.join(right_path[1:]) + ' - ' + end

					one_path = left_path + ' - ' + right_path
				ans_path.append(one_path)

		return ans_path


	def rstrip_path_loop(self, paths):
		# to rstrip the loop, caused by added inverse relationin, in the path
		cleaned_path = []

		for path in paths:
			entities = [path.split(' - ')[i] for i in range(len(path.split(' - '))) if i%2 == 0]
			head = entities[0]
			tail = entities[-1]
			mids = entities[1:-1]
			if head in mids or tail in mids:
				continue
			cleaned_path.append(path)
		
		return cleaned_path


	def find_all_paths_one_pair(self, pair):	
		start,end,start_FN,start_SN,end_FN,s_f_path_tracker,s_s_path_tracker,e_f_path_tracker = self.two_wise_bfs(pair)

		intermediate = []
		left_path_tracker, right_path_tracker = defaultdict(list), defaultdict(list)

		path1, path2, path3 = [], [], []

		if int(end) in start_FN: # 1-hop
			intermediate = start_FN & set([int(end)])
			left_path_tracker = s_f_path_tracker
			right_path_tracker[int(end)] = ['']

			path1 = self.combine_two_path(start,end,intermediate,left_path_tracker,right_path_tracker)
			
		if start_FN & end_FN:   # 2-hop
			intermediate = start_FN & end_FN
			left_path_tracker = s_f_path_tracker
			right_path_tracker = e_f_path_tracker

			path2 = self.combine_two_path(start,end,intermediate,left_path_tracker,right_path_tracker)
			path2 = self.rstrip_path_loop(path2)
			
		if start_SN & end_FN:  # 3-hop
			intermediate = start_SN & end_FN
			left_path_tracker = s_s_path_tracker
			right_path_tracker = e_f_path_tracker

			path3 = self.combine_two_path(start,end,intermediate,left_path_tracker,right_path_tracker)
			#there can be loop in 3-hop paths
			path3 = self.rstrip_path_loop(path3)
			
		ans_path = list(set(path1 + path2 + path3))

		return ans_path
	
	
