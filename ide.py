import math
import sys

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
        print "Gain:", gain
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
