import json
from grapher import BGGraph
import os
from collections import defaultdict

args = {}
'''
# /home/jwx/data_create/NELL
args["data_dir"] = "/home/jwx/data_create"
args["dataset_name"] = "NELL"
'''
#'''
# /home/jwx/data_create/FB15k-237/fb_path_use
args["data_dir"] = "/home/jwx/data_create/FB15k-237"
args["dataset_name"] = "fb_path_use"
#'''

bg_graph = BGGraph(args)

def get_test_task_paths(args):
    """
    store
    1. all the patterns  (i.e. relation in the path at most 3-hop)
    2. all the subgraphs (i.e. nodes and their labels in the paths)
    between the head and candidate tails in test_tasks
    """
    #bg_graph        = BGGraph(args)
    dataset         = os.path.join(args["data_dir"], args["dataset_name"])   
    ent2id          = json.load(open(os.path.join(dataset, 'entity2id.json')))
    rel2id          = json.load(open(os.path.join(dataset, 'relation2id.json')))
    test_tasks      = json.load(open(os.path.join(dataset, 'test_tasks.json')))
    pair_to_paths   = defaultdict(list)
    
    for r, _ in test_tasks.items():
        
        for p in _:
            h = ent2id[p[0]]
            t = ent2id[p[2]]

            if h == t:
                continue
            k = str(h) + '_' + str(t)
            
            cleaned_paths = bg_graph.find_all_paths_one_pair([str(h),str(t)])
            
            pair_to_paths[k] = cleaned_paths

    # initialize path directory
    path_dir = os.path.join(dataset, 'path')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        print('Path directory created: {}'.format(path_dir))
    else:
        print('Path directory exists: {}'.format(path_dir))

    # save
    json.dump(pair_to_paths, open(os.path.join(path_dir, 'test_pair2paths.json'), 'w'))
    
    print("Successfully extract paths in test task (between query heads and candidates)")



def get_train_task_paths(args):
    """
    store
    1. all the patterns  (i.e. relation in the path at most 3-hop)
    2. all the subgraphs (i.e. nodes and their labels in the paths)
    between the head and candidate tails in test_tasks
    """
    #bg_graph        = BGGraph(args)
    dataset         = os.path.join(args["data_dir"], args["dataset_name"])   
    ent2id          = json.load(open(os.path.join(dataset, 'entity2id.json')))
    rel2id          = json.load(open(os.path.join(dataset, 'relation2id.json')))
    train_tasks      = json.load(open(os.path.join(dataset, 'train_tasks.json')))
    pair_to_paths   = defaultdict(list)
    
    for r, _ in train_tasks.items():
        
        for p in _:
            h = ent2id[p[0]]
            t = ent2id[p[2]]

            if h == t:
                continue
            k = str(h) + '_' + str(t)
            
            cleaned_paths = bg_graph.find_all_paths_one_pair([str(h),str(t)])
            
            pair_to_paths[k] = cleaned_paths

    # initialize path directory
    path_dir = os.path.join(dataset, 'path')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        print('Path directory created: {}'.format(path_dir))
    else:
        print('Path directory exists: {}'.format(path_dir))

    # save
    json.dump(pair_to_paths, open(os.path.join(path_dir, 'train_pair2paths.json'), 'w'))
    
    print("Successfully extract paths in train task (between query heads and candidates)")

def get_dev_task_paths(args):
    """
    store
    1. all the patterns  (i.e. relation in the path at most 3-hop)
    2. all the subgraphs (i.e. nodes and their labels in the paths)
    between the head and candidate tails in test_tasks
    """
    #bg_graph        = BGGraph(args)
    dataset         = os.path.join(args["data_dir"], args["dataset_name"])   
    ent2id          = json.load(open(os.path.join(dataset, 'entity2id.json')))
    rel2id          = json.load(open(os.path.join(dataset, 'relation2id.json')))
    dev_tasks      = json.load(open(os.path.join(dataset, 'dev_tasks.json')))
    pair_to_paths   = defaultdict(list)
    
    for r, _ in dev_tasks.items():
        
        for p in _:
            h = ent2id[p[0]]
            t = ent2id[p[2]]

            if h == t:
                continue
            k = str(h) + '_' + str(t)
            
            cleaned_paths = bg_graph.find_all_paths_one_pair([str(h),str(t)])
            
            pair_to_paths[k] = cleaned_paths

    # initialize path directory
    path_dir = os.path.join(dataset, 'path')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        print('Path directory created: {}'.format(path_dir))
    else:
        print('Path directory exists: {}'.format(path_dir))

    # save
    json.dump(pair_to_paths, open(os.path.join(path_dir, 'dev_pair2paths.json'), 'w'))
    
    print("Successfully extract paths in dev task (between query heads and candidates)")

if __name__ == "__main__":
    get_test_task_paths(args)
    get_train_task_paths(args)
    get_dev_task_paths(args)
