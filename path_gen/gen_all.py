import json
import os

args = {}
'''
# /home/jwx/data_create/NELL
args["data_dir"] = "/home/jwx/data_create/NELL"
args["dataset_name"] = "path"
'''
#'''
# /home/jwx/data_create/FB15k-237/fb_path_use
args["data_dir"] = "/home/jwx/data_create/FB15k-237/fb_path_use"
args["dataset_name"] = "path"
#'''

path_dir = os.path.join(args["data_dir"], args["dataset_name"])

with open(os.path.join(path_dir, "train_pair2paths.json"),'r') as f1,  open(os.path.join(path_dir, "test_pair2paths.json"),'r') as f2, open(os.path.join(path_dir, 'dev_pair2paths.json'),'r') as f3:
    train_pair = json.load(f1)
    test_pair = json.load(f2)
    dev_pair = json.load(f3)


with open(os.path.join(args["data_dir"],"entity2id.json"),'r') as f1, open(os.path.join(args["data_dir"],"relation2id.json"),'r') as f2:
    ent2ids = json.load(f1)
    rel2ids = json.load(f2)

id2ent = {str(value):key for key,value in ent2ids.items()}
id2rel = {str(value):key for key,value in rel2ids.items()}

path_dict_all = {}
for pairs_all in [train_pair, test_pair, dev_pair]:
    for key,items in pairs_all.items():
        path_dict_all[id2ent[key.split('_')[0]]+'&'+id2ent[key.split('_')[1]]] = []
        for one_item in items:
            one_item_list = one_item.split(' - ')
            if len(one_item_list) == 7:
                path_dict_all[id2ent[key.split('_')[0]]+'&'+id2ent[key.split('_')[1]]].append([id2rel[one_item_list[1]],id2rel[one_item_list[3]],id2rel[one_item_list[5]]])
            if len(one_item_list) == 5:
                path_dict_all[id2ent[key.split('_')[0]]+'&'+id2ent[key.split('_')[1]]].append([id2rel[one_item_list[1]],id2rel[one_item_list[3]]])
            if len(one_item_list) == 3:
                path_dict_all[id2ent[key.split('_')[0]]+'&'+id2ent[key.split('_')[1]]].append([id2rel[one_item_list[1]]])


with open(os.path.join(path_dir,"train_valid_test_pair2paths_name.json"),'w') as f1:
    f1.write(json.dumps(path_dict_all))



