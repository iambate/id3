import random
import pickle as pkl
import argparse
import copy
import csv
import numpy as np
import scipy.stats as stats
import copy
import math
import sys

nodes_exp = 0

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
class FeatureDetail:
    def __init__(self, no_p, no_n, ig):
        self.no_p = no_p
        self.no_n = no_n
        self.ig = ig

    def __str__(self):
        return "(" + str(self.no_p) + ", " + str(self.no_n) + ", " + str(self.ig) + ")"

def get_feature_details(train, table_rows, feature_no, labels):
    return_dict = dict()
    val_pos = dict()
    val_neg = dict()
    for row_no in table_rows:
        feature_val = train[row_no][feature_no - 1]
        if(labels[row_no]==1):
            no_pos = val_pos.get(feature_val, 0)
            val_pos[feature_val] = no_pos + 1
        else:
            no_neg = val_neg.get(feature_val, 0)
            val_neg[feature_val] = no_neg + 1

    for feature_val in range(1, 6):
        no_p = val_pos.get(feature_val, 0)
        no_n = val_neg.get(feature_val, 0)
        if(no_p==0 or no_n==0):
            ig = 0
        elif(no_p == no_n):
            ig = 1
        else:
            total = (no_p + no_n)* 1.0
            ig = - no_p/total*math.log(no_p/total, 2) - no_n/total*math.log(no_n/total, 2)
        return_dict[feature_val] = FeatureDetail(no_p, no_n, ig)

    return return_dict

def get_feature_entropy(feature_details, table_rows):
    entropy = 0
    for feature_val in feature_details.keys():
        entropy+= (feature_details[feature_val].no_p + feature_details[feature_val].no_n
                  ) * feature_details[feature_val].ig
    return entropy/len(table_rows)

def get_class_entropy(table_rows, labels):
    no_p = 0
    no_n = 0
    for row_no in table_rows:
        if labels[row_no] == 1:
            no_p += 1
        else:
            no_n += 1
    if no_p == 0 or no_n == 0:
        return 0
    elif no_p == no_n:
        return 0
    else:
        total = (no_p + no_n) * 1.0
        return - no_p/total*math.log(no_p/total, 2) - no_n/total*math.log(no_n/total, 2)

def get_best_feature(train, table_rows, table_cols, labels):
    class_entropy = get_class_entropy(table_rows, labels)
    max_gain = -sys.maxint - 1
    best_feature = -1
    for feature_no in table_cols:
        feature_details = get_feature_details(train, table_rows, feature_no, labels)
        gain = class_entropy - get_feature_entropy(feature_details, table_rows)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature_no

    return best_feature

def get_higher_count(table_rows,labels):
    zero_count = 0
    one_count = 0
    for row in table_rows:
        if labels[row] == 0:
            zero_count+=1
        else:
            one_count+=1
    if zero_count == 0:
        return (1,True)
    if one_count == 0:
        return (0,True)
    if zero_count > one_count:
        return (0,False)
    else:
        return (1,False)

def get_tables_for_feature(train, table_rows, feature_no):
    return_dict = {1:[], 2:[], 3:[], 4: [], 5:[]}
    for row_no in table_rows:
        feature_val = train[row_no][feature_no - 1]
        li = return_dict.get(feature_val)
        li.append(row_no)
        return_dict[feature_val] =  li

    return return_dict


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

def gen_decision_tree(train, table_rows, table_cols, labels,pval):
    global nodes_exp
    label_value, either_zero = get_higher_count(table_rows,labels)
    if len(table_cols) == 0 or either_zero:
        if label_value == 1:
            nodes_exp+=1
            return TreeNode('T',[])
        else:
            nodes_exp+=1
            return TreeNode('F',[])
    feat_num = get_best_feature(train, table_rows, table_cols, labels)
    root = TreeNode(data = str(feat_num))
    featureSplit = get_tables_for_feature(train, table_rows, feat_num)
    table_cols.remove(feat_num)
    positive_count = 0
    negative_count = 0
    for row in table_rows:
        if labels[row] == 0:
            negative_count += 1
        else:
            positive_count += 1

    chi_stat = 0.0
    for k,v in featureSplit.iteritems():
        expected_pos = positive_count * len(v)/float(len(table_rows))
        expected_neg = negative_count * len(v)/float(len(table_rows))
        observed_pos = 0
        observed_neg = 0
        for row in v:
            if labels[row] == 0:
                observed_neg += 1
            else:
                observed_pos += 1
        dummy = 0.0
        if observed_pos != 0:
            dummy += pow(observed_pos - expected_pos, 2)/observed_pos

        if observed_neg != 0:
            dummy += pow(observed_neg - expected_neg, 2)/observed_neg

        chi_stat += dummy

    p_value = 1 - stats.chi2.cdf(chi_stat, len(featureSplit))

    if p_value < pval:
        for k,v in featureSplit.iteritems():
            if len(v) == 0:
                if label_value == 1:
                    nodes_exp+=1
                    return TreeNode('T',[])
                else:
                    nodes_exp+=1
                    return TreeNode('F',[])
            copy_table_cols = copy.deepcopy(table_cols)
            root.nodes[k-1] = gen_decision_tree(train, v, copy_table_cols,labels, pval)
    else:
        if label_value == 1:
            nodes_exp+=1
            return TreeNode('T',[])
        else:
            nodes_exp+=1
            return TreeNode('F',[])
    nodes_exp+=1
    return root
    

def iterate_tree(tree, feature):
    if tree.data == 'T':
        return 1
    elif tree.data == 'F':
        return 0
    elif int(tree.data) not in feature.keys():
        return 2
    else:
        data = int(tree.data)
        feature_val = feature[data]
        del feature[data]
        return iterate_tree(tree.nodes[feature_val - 1], feature)


def predict(test, tree):
    return_list = []
    for test_row in test:
        feature_dict = dict()
        feature_no = 1
        for feature_val in test_row:
            feature_dict[feature_no] = feature_val
            feature_no += 1

        return_list.append([iterate_tree(tree, feature_dict)])

    return return_list


parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = float(args['p'])
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']



Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

print("Training...")
table_rows = [ i for i in range(0, len(Xtrain))]
table_cols = [ i for i in range(1, num_feats + 1)]
train = Xtrain
labels = Ytrain

# Create a decision tree
s = gen_decision_tree(train, table_rows, table_cols, labels, pval)
s.save_tree(tree_name)
print("Testing...")
Ypredict = predict(Xtest, s)

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")

print "nodes exp" + str(nodes_exp)
