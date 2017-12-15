import random
import pickle as pkl
import argparse
import csv
import numpy as np
import ide

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

# loads Train and Test data
def load_data(ftrain, ftest):
	Xtrain, Ytrain, Xtest = [],[],[]
	with open(ftrain, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtrain.append(rw)

	with open(ftest, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtest.append(rw)

	ftrain_label = ftrain.split('.')[0] + '_label.csv'
	with open(ftrain_label, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = int(row[0])
	        Ytrain.append(rw)

	print('Data Loading: done')
	return Xtrain, Ytrain, Xtest


num_feats = 274

#A random tree construction for illustration, do not use this in your code!
def create_random_tree(depth):
    if(depth >= 7):
        if(random.randint(0,1)==0):
            return TreeNode('T',[])
        else:
            return TreeNode('F',[])

    feat = random.randint(0,273)
    root = TreeNode(data=str(feat))

    for i in range(5):
        root.nodes[i] = create_random_tree(depth+1)

    return root
    

def iterate_tree(tree, feature):
    if tree.data == 'T':
        return 1
    elif tree.data == 'F':
        return 0
    elif tree.data not in feature.keys():
        return 2
    else:
        feature_val = feature[tree.data]
        del feature[tree.data]
        return iterate_tree(tree.children[feature_val], feature)


def predict(test, tree):
    return_list = []
    for test_row in test:
        feature_dict = dict()
        feature_no = 0
        for feature_val in test_row:
            feature_dict[feature_no] = feature_val
            feature_no += 1

        return_list.append(iterate_tree(tree, feature_dict))

    return return_list


parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']



Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

print("Training...")
table_rows = [ i for i in range(0, len(Xtrain))]
table_cols = [ i for i in range(0, num_feats)]
train = Xtrain
labels = Ytrain

# Create a decision tree
s = create_random_tree(0)
s.save_tree(tree_name)
print("Testing...")
Ypredict = predict(Xtest, s)

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")
