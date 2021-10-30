import torch
import torch.nn as nn
from utils import *
from torch.autograd import Variable
import torch.nn.functional as F
import networkx

class T_GRUA(nn.Module):
	def __init__(self,  kernel_num, embed_dim, hidden_dim, h_hrt_bg, ent2id, id2ent, id2rel, batch_size, edge_matrix, edge_nums, topk, rel_emb, ent_emb,  device):
		nn.Module.__init__(self)
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.h_hrt_bg = h_hrt_bg
		self.rel_emb = rel_emb
		self.device = device
		self.ent2id = ent2id
		self.id2ent = id2ent
		self.id2rel = id2rel
		self.batch_size = batch_size
		self.edge_matrix = edge_matrix
		self.edge_nums = edge_nums
		self.topk = topk
		self.kernel_num = kernel_num
		self.ent_emb = ent_emb

		self.GRUc = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_dim)

		self.T_GRUS = T_GRUS(self.GRUc, self.embed_dim, self.hidden_dim, self.rel_emb, self.ent_emb, self.device)
		self.T_GRUQ = T_GRUQ(self.GRUc, self.embed_dim, self.hidden_dim, self.rel_emb, self.edge_matrix, self.topk, self.device)
		self.loss = nn.MarginRankingLoss(margin=2.0)

		self.n_bins = 21
		self.dense = nn.Linear(self.n_bins, 200)
		self.dense1 = nn.Linear(self.n_bins, 1)
		self.dense_add = nn.Linear(200, 1)

		self.s_th = nn.Linear(2*self.embed_dim,self.embed_dim)
		self.su = nn.Linear(self.embed_dim,self.embed_dim)
		self.th = nn.Linear(self.embed_dim,self.embed_dim)
		self.att = nn.Linear(self.embed_dim,1)
		mu = self.kernel_mus(self.n_bins) 
		sigma = self.kernel_sigmas(self.n_bins)
		self.mu = Variable(torch.FloatTensor(mu), requires_grad = False).view( 1, 1, 1, self.n_bins).to(self.device)
		self.sigma = Variable(torch.FloatTensor(sigma), requires_grad = False).view(1, 1, 1,self.n_bins).to(self.device)

	def label_pro(self,right_tail_list, parent_index, tree_q):

		mask_f = torch.ones(tree_q.size()[0],tree_q.size()[1])
		mask_b = torch.ones(tree_q.size()[0],tree_q.size()[1])
		label = torch.zeros(tree_q.size()[0],tree_q.size()[1])

		for i in range(len(right_tail_list)):
			label = torch.where(tree_q==right_tail_list[i],torch.ones(tree_q.size()[0],tree_q.size()[1]),label)

		index_label = label[0][parent_index[0].long()] 
		mask_f[1] = mask_f[1].int()&(~index_label.int()) 
		index_label = label[1][parent_index[1].long()] 
		mask_f1 = mask_f[1][parent_index[1].long()] 
		mask_f[2] = mask_f[2].int() & mask_f1.int() & (~(index_label.int())) 

		for i in range(tree_q.size()[1]): 
			mask_b[1][parent_index[1][i].long()] = mask_b[1][parent_index[1][i].long()].int() & (~(label[2][i].int()))
		for i in range(tree_q.size()[1]):
			mask_b[0][parent_index[0][i].long()] = mask_b[0][parent_index[0][i].long()].int() & (~(label[1][i].int())) & mask_b[1][i].int()
		mask = mask_f*mask_b
		
		mask = torch.where(tree_q==len(self.ent2id),torch.zeros(tree_q.size()[0],tree_q.size()[1]),mask)

		return mask,label

	def count_one(self, t_h, t_h_q, idx, B, tree_all_emb, support_tree_emb, support_tree_emb_list, node, Train=True, parent_all=None, path_all=None):

		index_num = B.tolist()
		index_num_c = index_num.copy()
		max_len = max(index_num)
		emb_all = torch.zeros(len(index_num), max_len, tree_all_emb.size()[-1]).to(self.device)
		mask_all = torch.zeros(len(index_num), max_len).to(self.device)
		for i in range(len(index_num)):
			if i==0:
				continue
			index_num[i] = index_num[i]+index_num[i-1]
		score = []
		for i in range(len(index_num)):
			if i==0:
				idx_one = idx[:index_num[i]]
			else:
				idx_one = idx[index_num[i-1]:index_num[i]]

			'''
			for q in idx_one.tolist():
				print("rel:",path_all[q],[self.id2rel[a] for a in path_all[q]])
				print(path_all[q])
				print("node:",parent_all[q],[self.id2ent[a] for a in parent_all[q]])
				print(path_all[q])
			'''
			true_node = [self.id2ent[i] for i in node[idx_one].tolist()]
			emb_one = tree_all_emb[idx_one]

			emb_all[i][:index_num_c[i]] = emb_one
			mask_all[i][:index_num_c[i]] = 1

		for i in range(len(support_tree_emb_list)):
			
			similarity = torch.matmul( F.normalize(emb_all,2,-1),  F.normalize(support_tree_emb_list[i],2,-1).transpose(0,1))
			beta = torch.matmul(support_tree_emb_list[i], self.th(t_h[i]).unsqueeze(-1)).squeeze(-1)
			alpha = torch.bmm(emb_all, self.th(t_h_q).unsqueeze(-1)).squeeze(-1)
			alpha = alpha * mask_all	
			#alpha = alpha / torch.sum(alpha,dim=-1).unsqueeze(-1)  # softmax
			similarity = similarity * beta 
			similarity,_ = similarity.topk(k=1,dim=2)
			pooling_value = torch.exp((- ((similarity.unsqueeze(-1) - self.mu) ** 2) / (self.sigma ** 2) / 2)).squeeze(2)
			log_pooling_sum = torch.log(torch.clamp(pooling_value, min=1e-30)) * mask_all.unsqueeze(-1) * 0.01 
			log_pooling_sum = torch.sum(log_pooling_sum, 1)

			if i==0:
				log_s = log_pooling_sum.unsqueeze(1)
			else:
				log_s = torch.cat([log_s, log_pooling_sum.unsqueeze(1)], dim=1)
			
			output = self.dense_add(self.dense(log_pooling_sum)) 

			if i==0:
				output_all = output
			else:
				output_all = torch.cat([output_all, output],dim=-1)	

		output_all = torch.sigmoid(output_all)
		output,_ = torch.max(output_all,dim=-1)
		output = output.unsqueeze(-1)

		if Train==True:
			right_score = output[0]
			wrong_score = output[1:]
			right_score_a = right_score.repeat(wrong_score.size()[0],1)
			label = torch.ones(right_score_a.size()[0])
			loss = self.loss(right_score_a.squeeze(1),wrong_score.squeeze(1),label)
			return loss
		else:
			return output

 
	def forward(self, support_pair, support_rel, support_path, query_head, query_tail, one_tomany, cos_rel_all, Train = True, rel_candidates=None):
		num_T = max(map(len, support_rel))

		flag_S = True 
		if num_T == 0:
			flag_S = False
		else:
			support_tree_emb, support_tree_emb_list, t_h = self.T_GRUS(support_path, support_pair)  
		
		if flag_S: 
			if Train==True:
				tree_q, tree_all_emb, parent_index, parent_node, aim_rel_all = self.T_GRUQ(support_tree_emb, support_rel, query_head, cos_rel_all, t_h, True)
			else:
				tree_q, tree_all_emb, parent_index, parent_node, aim_rel_all = self.T_GRUQ(support_tree_emb, support_rel, query_head, cos_rel_all, t_h, False)

			tree_all_emb = tree_all_emb.reshape(tree_all_emb.size()[0],-1,tree_all_emb.size()[-1])

		if Train:
			loss = 0
			if flag_S: 
				for i in range(tree_q.size()[0]):
					aim_rel_one = aim_rel_all[i]
					path_all = []
					path_a1 = []
					path_a1_one = aim_rel_one[0].tolist()
					for m in range(len(path_a1_one)):
						path_a1.append([path_a1_one[m]])
					path_a2 = []
					path_a2_one = aim_rel_one[1].tolist()
					for m in range(len(path_a2_one)):
						path_a2.append([path_a1_one[m],path_a2_one[m]])
					path_a3 = []
					path_a3_one = aim_rel_one[2].tolist()
					for m in range(len(path_a3_one)):
						path_a3.append([path_a1_one[m],path_a2_one[m],path_a3_one[m]])
					path_all=path_a1+path_a2+path_a3
					# parent
					parent_node_one = parent_node[i]
					parent_all = []
					parent_a1 = []
					parent_a1_one = parent_node_one[0].tolist()
					for m in range(len(parent_a1_one)):
						parent_a1.append([parent_a1_one[m]])
					parent_a2 = []
					parent_a2_one = parent_node_one[1].tolist()
					for m in range(len(parent_a2_one)):
						parent_a2.append([parent_a1[m][0],parent_a2_one[m]])
					parent_a3 = []
					parent_a3_one = parent_node_one[2].tolist()
					for m in range(len(parent_a3_one)):
						parent_a3.append([parent_a1[m][0],parent_a2_one[m],parent_a3_one[m]])
					parent_all=parent_a1+parent_a2+parent_a3

					right_tail_list = one_tomany[i] 
					mask_one,label_one = self.label_pro(right_tail_list, parent_index[i], tree_q[i])
					filt_node = mask_one*tree_q[i]
	
					filt_node = filt_node.reshape(-1,1).squeeze(1).tolist()
					right_in_tree_node = list(set(filt_node).intersection(set(right_tail_list))) # positive
					wrong_in_tree_node = list(set(filt_node)-set(right_in_tree_node)-set([0,len(self.ent2id)]))  #negative
					if len(right_in_tree_node)==0:
						continue
					if query_tail[i] not in right_in_tree_node:
						continue
					pos_node = query_tail[i]
					neg_node = random.sample(wrong_in_tree_node, min(3,len(wrong_in_tree_node)))  

					pos_neg_node = [int(pos_node)]+neg_node
					if len(pos_neg_node)<2:
						continue

					t_h_q = self.ent_emb(torch.tensor(pos_neg_node).long().to(self.device)) - self.ent_emb(query_head[i])			
					pos_neg_node_tensor = torch.tensor(pos_neg_node).unsqueeze(-1)
					A = tree_q[i].reshape(-1,1).squeeze(-1)==pos_neg_node_tensor
					idx = A.nonzero().select(-1,1)
					B = torch.sum(A,dim=1)
					loss = loss + self.count_one(t_h, t_h_q, idx, B, tree_all_emb[i], support_tree_emb, support_tree_emb_list, tree_q[i].reshape(-1,1).squeeze(-1),True, parent_all, path_all)
			return loss
		
		else:
			hit1 = []
			hit5 = []
			hit10 = []
			mrr = []
			if flag_S:
				for i in range(tree_q.size()[0]):
					tree_node_one = tree_q[i].reshape(-1,1).squeeze(1) 
					filt_node = list(set(tree_node_one.tolist()).intersection(set(rel_candidates)))
					filt_node = list(set(filt_node)-set(one_tomany[i]))

					filt_name = [self.id2ent[i] for i in filt_node]
					rel_candidates_name = [self.id2ent[i] for i in rel_candidates]
					if query_tail[i] not in filt_node:
						hit1.append(0)
						hit5.append(0)
						hit10.append(0)
						mrr.append(0)
						continue
					t_h_q = self.ent_emb(torch.tensor(filt_node).long().to(self.device)) - self.ent_emb(query_head[i])
					A = tree_q[i].reshape(-1,1).squeeze(-1)==torch.tensor(filt_node).unsqueeze(-1)
					idx = A.nonzero().select(-1,1)
					B = torch.sum(A,dim=1)
					output = self.count_one(t_h, t_h_q, idx, B, tree_all_emb[i], support_tree_emb, support_tree_emb_list, tree_q[i].reshape(-1,1).squeeze(-1), False).squeeze(-1)

					sort_score, sort_index = torch.sort(output, descending=True) # filter nodes' scores
					node_index = torch.tensor(filt_node)[sort_index.long()]	 # sort filter nodes according to scores
					r_max_score = sort_score.tolist()[node_index.tolist().index(query_tail[i])] 	# find the score for the right answer	
					index_r = sort_score.tolist().index(r_max_score) + 1 # the index for the right answer
					 
					
					if index_r==1:
						hit1.append(1)
					else:
						hit1.append(0)
					if index_r<=5:
						hit5.append(1)
					else:
						hit5.append(0)
					if index_r<=10:
						hit10.append(1)
					else:
						hit10.append(0)
					mrr.append(1/index_r)


			return hit1,hit5,hit10,mrr


	def cos(self,tree_all_emb, support_tree_emb):
		tree_emb_bro = tree_all_emb.unsqueeze(-2).repeat(1,1,support_tree_emb.size()[0],1)
		sim = torch.sigmoid(torch.cosine_similarity(tree_emb_bro, support_tree_emb,dim=-1))
		sim_max = torch.max(sim,dim=-1)[0]
		return sim_max

	def kernel_mus(self, n_kernels): 
		l_mu = [1]
		if n_kernels == 1:
			return l_mu

		bin_size = 2.0 / (n_kernels - 1)  
		l_mu.append(1 - bin_size / 2)  
		for i in range(1, n_kernels - 1):
			l_mu.append(l_mu[i] - bin_size)
		return l_mu


	def kernel_sigmas(self, n_kernels):
		bin_size = 2.0 / (n_kernels - 1)
		l_sigma = [0.001]  
		if n_kernels == 1:
			return l_sigma

		l_sigma += [0.1] * (n_kernels - 1)
		return l_sigma

	def get_intersect_matrix(self, support_tree_emb, node_emb, mask_d):
		node_emb_bro = node_emb.unsqueeze(-2).repeat(1,1,support_tree_emb.size()[0],1)
		sim = torch.cosine_similarity(node_emb_bro, support_tree_emb,dim=-1)
		sim_max = torch.max(sim,dim=-1)[0].unsqueeze(-1)
		pooling_value = torch.exp((- ((sim_max - self.mu) ** 2) / (self.sigma ** 2) / 2)) * mask_d
		pooling_value = torch.log(torch.clamp(pooling_value, min=1e-10))*0.01
		log_pooling_sum = torch.sum(pooling_value, 2)
		return log_pooling_sum

	
	def knrm(self, node, node_emb, support_tree_emb):
		mask_d = torch.where(node==68544,torch.full_like(node, 0),torch.full_like(node, 1)) 
		mask_d = mask_d.view(mask_d.size()[0], mask_d.size()[1], 1)
		log_pooling_sum = torch.sigmoid(self.get_intersect_matrix(support_tree_emb, node_emb, mask_d))
		return log_pooling_sum


class T_GRUQ(nn.Module):
	def __init__(self, GRUc, embed_dim, hidden_dim, rel_emb, edge_matrix, topk, device):
		nn.Module.__init__(self)
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.rel_emb = rel_emb
		self.device = device
		self.hiddenRNN = GRUc
		self.edge_matrix = edge_matrix
		self.topk = topk
	
	def forward(self, support_tree_emb, support_rel,query_head,cos_rel_all,t_h, Train=True):
		support_rel_all = []
		for i in range(len(support_rel)):
			for j in range(len(support_rel[i])):
				if support_rel[i][j] not in support_rel_all:
					support_rel_all.append(support_rel[i][j])
		support_rel_all = torch.tensor(support_rel_all)  
		start_emb = torch.zeros(query_head.size()[0],self.embed_dim)
		tree_node_list = []
		tree_node_emb = []
		parent_index_list = [] 
		parent_node_list = [] 
		aim_rel_all_list = [] 
		aim_rel_all_list_b = []
		step = 0
		max_step = 3
		while True:
			step = step + 1
			if step==max_step+1:
				break
			if step==1:
				current_entity = query_head.unsqueeze(1)
			else:
				current_entity = aim_ent
			candidates = self.edge_matrix[current_entity.long()]
			candidate_ent, candidate_relations = candidates.select(-1, 0), candidates.select(-1, 1)
			batch_size, num_count, nei_num = candidate_ent.size()[0],candidate_ent.size()[1],candidate_ent.size()[2]
			batch=torch.arange(batch_size)
			candidate_relations_new = candidate_relations.unsqueeze(-1).repeat(1,1,1,support_rel_all.size()[0]) 
			support_rel_all_new = support_rel_all.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(candidate_relations.size()[0],candidate_relations.size()[1],candidate_relations.size()[2],1)

			score_rs = cos_rel_all[support_rel_all_new.reshape(-1,1).squeeze(1).long(),candidate_relations_new.reshape(-1,1).squeeze(1).long()]
			score_rs = score_rs.reshape(candidate_relations.size()[0],candidate_relations.size()[1],candidate_relations.size()[2],-1)
			score_rs = torch.max(score_rs,dim=-1)[0]
			score_max = score_rs
			'''
			#t-h extend
			candidate_relations_new1 = candidate_relations.unsqueeze(-1).repeat(1,1,1,t_h.size()[0])
			score_th = torch.cosine_similarity(self.rel_emb(candidate_relations_new1.long()),t_h,dim=-1)
			score_th = torch.max(score_th,dim=-1)[0]
			score_max = score_th
			'''
			'''
			#support_emb, extend
			candidate_relations_new1 = candidate_relations.unsqueeze(-1).repeat(1,1,1,support_tree_emb.size()[0])
			score_s = torch.cosine_similarity(self.rel_emb(candidate_relations_new1.long()), support_tree_emb,dim=-1)
			score_s = torch.max(score_s,dim=-1)[0]
			score_max = score_s
			'''
			score_max = score_max.reshape(score_max.size()[0],-1) 

			if Train==True:
				if np.random.rand()<0.8:
					score_max = score_max * torch.randn(score_max.size()[0],score_max.size()[1])
				else:
					m = torch.distributions.multinomial.Multinomial(total_count=self.topk, logits=score_max)
					score_max = m.sample()
				
			next_scores, next_act = score_max.topk(k=min(self.topk, score_max.size(-1)),dim=-1)
			parent = ((next_act+1)/float(nei_num)).ceil() - 1  

			batch1 = batch.unsqueeze(1).repeat(1,next_scores.size()[1]).reshape(-1,1)
			aim_ent = candidate_ent.reshape(batch_size,-1)[batch1.squeeze(1).long(),next_act.reshape(-1,1).squeeze(1).long()].reshape(batch_size,-1)
			aim_rel = candidate_relations.reshape(batch_size,-1)[batch1.squeeze(1).long(),next_act.reshape(-1,1).squeeze(1).long()].reshape(batch_size,-1)

			aim_rel_all_list_b.append(aim_rel)
			if step==1:
				aim_emb = self.hiddenRNN(self.rel_emb(aim_rel.long()).reshape(-1,self.embed_dim)).reshape(batch_size,-1,self.embed_dim)
				tree_node_emb.append(aim_emb)
				tree_node_list.append(aim_ent)
				parent_node_list.append(query_head.unsqueeze(1).repeat(1,aim_ent.size()[1])) 
			else:
				parent_node = tree_node_list[step-2][batch.unsqueeze(1).repeat(1,num_count).reshape(-1,1).squeeze(1).long(),parent.reshape(-1,1).squeeze(1).long()].reshape(batch_size,-1)
				parent_emb = tree_node_emb[step-2][batch.unsqueeze(1).repeat(1,num_count).reshape(-1,1).squeeze(1).long(), parent.reshape(-1,1).squeeze(1).long()]
				aim_emb = self.hiddenRNN(self.rel_emb(aim_rel.long()).reshape(-1,self.embed_dim),parent_emb).reshape(batch_size,-1,self.embed_dim)
				tree_node_emb.append(aim_emb)
				tree_node_list.append(aim_ent)
				parent_index_list.append(parent)
				parent_node_list.append(parent_node)
				aim_rel_all_list.append(aim_rel_all_list_b[step-2][batch.unsqueeze(1).repeat(1,num_count).reshape(-1,1).squeeze(1).long(), parent.reshape(-1,1).squeeze(1).long()].reshape(batch_size,-1))
				if step==max_step:
					parent_index_list.append(torch.arange(next_scores.size()[1]).unsqueeze(0).repeat(batch_size,1).float())
					aim_rel_all_list.append(aim_rel)

		for i in range(len(tree_node_list)):
			if i==0:
				tree_node = tree_node_list[i].unsqueeze(1)
				tree_emb_all = tree_node_emb[i].unsqueeze(1)
				parent_index = parent_index_list[i].unsqueeze(1)
				parent_node = parent_node_list[i].unsqueeze(1)
				aim_rel_all = aim_rel_all_list[i].unsqueeze(1)
			else:
				tree_node = torch.cat([tree_node,tree_node_list[i].unsqueeze(1)],dim=1)
				tree_emb_all = torch.cat([tree_emb_all,tree_node_emb[i].unsqueeze(1)],dim=1)
				parent_index = torch.cat([parent_index,parent_index_list[i].unsqueeze(1)],dim=1)
				parent_node = torch.cat([parent_node.long(),parent_node_list[i].long().unsqueeze(1)],dim=1)
				aim_rel_all = torch.cat([aim_rel_all, aim_rel_all_list[i].unsqueeze(1)],dim=1)

		return tree_node.to(self.device), tree_emb_all.to(self.device), parent_index.to(self.device), parent_node.to(self.device), aim_rel_all.to(self.device)



class T_GRUS(nn.Module):
	def __init__(self, GRUc, embed_dim, hidden_dim, rel_emb, ent_emb, device):
		nn.Module.__init__(self)
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.rel_emb = rel_emb
		self.ent_emb = ent_emb
		self.device = device
		self.hiddenRNN = GRUc
	
	def forward(self, support_path, support_pair):
		support_pair_t = torch.tensor(support_pair)
		head_t = support_pair_t.select(-1,0)
		tail_t = support_pair_t.select(-1,1)
		t_h = self.ent_emb(tail_t) - self.ent_emb(head_t)

		support_path_filter = []
		flag = 0
		for i in range(len(support_path)):
			if i==0 and len(support_path[i])!=0:
				t_h_new = t_h[0].unsqueeze(0)
				support_path_filter.append(support_path[i])
				flag = 1
				
			if len(support_path[i])!=0 and i!=0:
				support_path_filter.append(support_path[i])
				if flag == 1:
					t_h_new = torch.cat([t_h_new,t_h[i].unsqueeze(0)],dim=0)
				else:
					t_h_new = t_h[i].unsqueeze(0)
					flag = 1

		tail_emb = []
		rel_pad = self.rel_emb.weight.size()[0] - 1
		for i in range(len(support_path_filter)):
			tail_loc = torch.tensor(list(map(len, support_path_filter[i])), dtype=torch.long)
			support_path_one = support_path_filter[i]
			support_path_one_pad = list2tensor(support_path_one, padding_idx=rel_pad, dtype=torch.int, device=self.device)
			if support_path_one_pad.size()[1]==1:
				one_step_rel = support_path_one_pad.select(-1,0)
				lay = 1
			if support_path_one_pad.size()[1]==2:
				one_step_rel = support_path_one_pad.select(-1,0)
				two_step_rel = support_path_one_pad.select(-1,1)
				lay = 2
			if support_path_one_pad.size()[1]==3:
				one_step_rel = support_path_one_pad.select(-1,0)
				two_step_rel = support_path_one_pad.select(-1,1)
				three_step_rel = support_path_one_pad.select(-1,2)
				lay = 3

			if lay==1:
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()))
			elif lay==2:
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()))
				update_emb2 = self.hiddenRNN(self.rel_emb(two_step_rel.long()), update_emb1)
			elif lay==3:
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()))
				update_emb2 = self.hiddenRNN(self.rel_emb(two_step_rel.long()), update_emb1)
				update_emb3 = self.hiddenRNN(self.rel_emb(three_step_rel.long()), update_emb2)
			batch = torch.arange(tail_loc.size()[0])
			tail_loc = tail_loc-1

			if lay==1:
				tail_emb_one = update_emb1[tail_loc]
			elif lay==2:
				cat_emb = torch.cat([update_emb1.unsqueeze(1),update_emb2.unsqueeze(1)], dim=1)
				tail_emb_one = cat_emb[batch,tail_loc]
			elif lay==3:
				cat_emb = torch.cat([update_emb1.unsqueeze(1),update_emb2.unsqueeze(1)], dim=1)
				cat_emb = torch.cat([cat_emb,update_emb3.unsqueeze(1)], dim=1)
				tail_emb_one = cat_emb[batch,tail_loc] 
			tail_emb.append(tail_emb_one)
		
		for i in range(len(tail_emb)):
			if i==0:
				tail_emb_all = tail_emb[0]
			else:
				tail_emb_all = torch.cat([tail_emb_all,tail_emb[i]], dim=0)

		return tail_emb_all,tail_emb,t_h_new

class KG:
    def __init__(self, facts: list, entity_num: int, relation_num: int, node_scores: list = None,
                 train_width=None, device=torch.device('cpu'), add_self_loop=False, self_loop_id=None):
        self.dataset = facts
        self.device = device
        self.train_width = train_width
        self.entity_num = entity_num
        self.relation_num = relation_num
        if entity_num is None:
            entity_num = max(map(lambda x: max(x[0], x[2]), facts)) + 1
        self.edge_data = [[] for _ in range(entity_num + 1)]
        for e1, rel, e2 in facts:
            self.edge_data[e1].append((e1, e2, rel))
        if node_scores is not None and (train_width is not None):
            neighbor_limit = train_width
            self.node_scores = node_scores
            for head in range(len(self.edge_data)):
                self.edge_data[head].sort(key=lambda x: self.node_scores[x[1]], reverse=True)
                self.edge_data[head] = self.edge_data[head][:neighbor_limit]
        
        self.ignore_relations = None
        self.ignore_edges = None
        self.ignore_relation_vectors = None
        self.ignore_edge_vectors = None
        self.ignore_pairs = None
        self.ignore_pair_vectors = None

    def to_networkx(self, multi=True, neighbor_limit=None):
        if neighbor_limit is None:
            neighbor_limit = max(map(len, self.edge_data))
        if multi:
            graph = networkx.MultiDiGraph()
        else:
            graph = networkx.DiGraph()
        for edges in self.edge_data:
            for head, tail, relation in edges[:neighbor_limit]:
                if multi:
                    graph.add_edge(head, tail, relation)
                else:
                    graph.add_edge(head, tail)
        return graph
