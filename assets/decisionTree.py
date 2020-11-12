import os
import sys
import numpy as np
from collections import Counter
from dataLoader import DataLoader
from evaluator import Evaluator
import json


class DecisionTree():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.dataset = self.data_loader.get_dataset()
        self.head = self.data_loader.get_head()
        self.set_labels = sorted(self.data_loader.get_set_labels())
        self.all_labels = self.data_loader.get_all_labels()
        self.tree = None

        counter = Counter(self.all_labels).most_common()
        print(f"[{counter[0][1]} {counter[0][0]}/{counter[1][1]} {counter[1][0]}]")

    def majority_vote(self,dataset):
        list_of_labels = dataset[1:,-1]
        counter = Counter(list_of_labels).most_common()
        if len(counter) > 1:
            if counter[0][1] != counter[1][1]:
                return counter[0][0]
            else:
                return sorted(counter,key=lambda x:x[0])[1][0]
        else:
            return counter[0][0]

    def get_entropy(self,dataset):
        list_of_labels = dataset[1:,-1]

        num_of_instances = len(list_of_labels)
        counter = Counter(list_of_labels)
        label_count_pairs = counter.most_common()

        entropy = 0
        for pair in label_count_pairs:
            entropy += -pair[1]/num_of_instances*np.log2(pair[1]/num_of_instances)
        return entropy

    def get_information_gain(self,dataset):
        original_entropy = self.get_entropy(dataset)
        conditional_entropy = 0

        feature_values = dataset[1:,0]
        feature_count = Counter(feature_values).most_common()
        probability = []
        for idx,pair in enumerate(feature_count):
            probability.append(pair[1]/(len(dataset)-1))
            rows = [['feature','class']]
            for instance in dataset:
                if pair[0] == instance[0]:
                    rows.append([pair[0],instance[-1]])
            conditional_entropy += probability[idx]*self.get_entropy(np.array(rows))
        return original_entropy-conditional_entropy
    
    def find_best_feature(self,dataset):
        information_gain = []
        rows = len(dataset)
        columns = len(self.head)

        for column in range(columns-1):
            new_dataset = dataset[:,[column,-1]]
            information_gain.append(self.get_information_gain(new_dataset))
        
        highest_idx = np.argmax(information_gain)

        if information_gain[highest_idx] > 0:
            return highest_idx
        else:
            return None
    
    def split_dataset(self,dataset,split_idx):
        set_feature_values = set(dataset[1:,split_idx])

        splitted_dataset_dict = {}
        for value in set_feature_values:
            l = [list(self.head)]

            for row in dataset:
                if row[split_idx] == value:
                    l.append(row)
            
            splitted_dataset_dict[value] = np.array(l)
        return splitted_dataset_dict

    def train(self,max_depth=0,depth=0,dataset=None):
        if dataset is None:
            dataset = self.dataset

        all_labels = dataset[1:,-1]
        model = {}

        if max_depth == 0:
            self.tree = self.majority_vote(dataset)
            return
        elif len(set(all_labels)) == 1:
            self.tree = all_labels[0]
            return all_labels[0]

        ticker = False
        for i in range(len(dataset[0])-1):
            column = dataset[1:,i]
            if len(set(column)) != 1:
                ticker += 1
        if not ticker:
            return self.majority_vote(dataset) 
        
        best_feature_idx = self.find_best_feature(dataset)

        if best_feature_idx is None:
            return self.majority_vote(dataset)

        best_feature = self.head[best_feature_idx]
        model[best_feature] = {}

        splitted_dataset_dict = self.split_dataset(dataset,best_feature_idx)
        depth += 1

        for idx, (feature_value, sub_dataset) in enumerate(splitted_dataset_dict.items()):
            if depth <= max_depth-1:
                model[best_feature][feature_value] = self.train(max_depth,depth,sub_dataset)
            else:
                model[best_feature][feature_value] = self.majority_vote(sub_dataset)
            if idx == len(splitted_dataset_dict.items()) - 1:
                self.tree = model
                return model

        self.tree = model

    def unit_classify(self,row,tree):
        if type(tree) is not dict:
            return tree
        best_feature = list(tree.keys())[0]
        best_feature_idx = np.argwhere(self.head == best_feature)[0][0]
        best_feature_value = row[best_feature_idx]

        sub_tree = tree[best_feature][best_feature_value]
        return self.unit_classify(row,sub_tree)
        
    def classify(self,output_file,tree=None,dataset=None):
        if dataset is None:
            dataset = self.dataset
        if tree is None:
            tree = self.tree

        if not isinstance(tree,dict):
            with open(output_file,'w') as f:
                for row in dataset[1:]:
                    f.write(str(tree)+'\n')
        else:
            with open(output_file,'w') as f:
                for row in dataset[1:]:
                    f.write(str(self.unit_classify(row,tree))+'\n')
    
    def count_labels(self,dataset,feature,value):
        feature_idx = np.argwhere(self.head==feature)
        labels = []
        for row in dataset[1:]:
            if row[feature_idx] == value:
                labels.append(row[-1])
        counter = Counter(labels).most_common()
        counter = sorted(counter,key=lambda x:x[0])
        if len(counter)==2:
            return f"[{counter[0][1]} {counter[0][0]}/{counter[1][1]} {counter[1][0]}]"
        else:
            idx = self.set_labels.index(counter[0][0])
            if idx == 0:
                return f"[{counter[0][1]} {counter[0][0]}/0 {self.set_labels[1]}]"
            else:
                return f"[0 {self.set_labels[0]}/{counter[0][1]} {counter[0][0]}]"
    
    def get_sub_dataset(self,dataset,feature,value):
        feature_idx = np.argwhere(self.head==feature)
        sub_dataset = [list(self.head)]
        for row in dataset:
            if row[feature_idx] == value:
                sub_dataset.append(row)
        return np.array(sub_dataset)
    
    def visualize_tree(self,depth=1,dataset=None,tree=None):
        if dataset is None:
            dataset = self.dataset
        if tree is None:
            tree = self.tree

        if isinstance(tree,np.str_):
            print(f'Algorithm is majority vote: {str(tree)}')

        if isinstance(tree,dict):
            for name, content in tree.items():
                for k, v in content.items():
                    print("{}{} = {}: ".format('| ' * depth, name, k), end=self.count_labels(dataset,name,k))
                    if isinstance(v, str):
                        print()
                    else:
                        print()
                        sub_dataset = self.get_sub_dataset(dataset,name,k)
                        self.visualize_tree(dataset=sub_dataset,tree=v, depth=depth+1)
            return

if __name__ == "__main__":

    # Ensure exactly 7 arguments
    if len(sys.argv) != 7:
        print('USAGE: python decisionTree.py TRAIN_INPUT_FILE TEST_INPUT_FILE MAX_DEPTH TRAIN_OUT_FILE TEST_OUT_FILE METRICS_OUT_FILE')
        sys.exit(1)
    
    TRAIN_INPUT_FILE = sys.argv[1]
    TEST_INPUT_FILE = sys.argv[2]
    MAX_DEPTH = sys.argv[3]
    TRAIN_OUT_FILE = sys.argv[4]
    TEST_OUT_FILE = sys.argv[5]
    METRICS_OUT_FILE = sys.argv[6]

    # Check the input file type
    if not (TRAIN_INPUT_FILE.endswith('.tsv') and TEST_INPUT_FILE.endswith('.tsv')):
        print('Error: TRAIN_INPUT_FILE and TEST_INPUT_FILE must be .tsv files')
        sys.exit(1)
    
    # Check the split index
    if not MAX_DEPTH.isdigit():
        print('Error: MAX_DEPTH must be a number')

    # Check the output file type
    if not (TRAIN_OUT_FILE.endswith('.labels') and TEST_OUT_FILE.endswith('.labels')):
        print('Error: TRAIN_OUT_FILE and TEST_OUT_FILE must be .labels files')
        sys.exit(1)
    
    if not METRICS_OUT_FILE.endswith('.txt'):
        print('Error: METRICS_OUT_FILE must be .txt file')
        sys.exit(1)

    train_data_loader = DataLoader()
    train_data_loader.load_data(TRAIN_INPUT_FILE)
    train_dataset = train_data_loader.get_dataset()

    test_data_loader = DataLoader()
    test_data_loader.load_data(TEST_INPUT_FILE)
    test_dataset = test_data_loader.get_dataset()

    decision_tree = DecisionTree(train_data_loader)
    decision_tree.train(max_depth=int(MAX_DEPTH),dataset=train_dataset)
    
    # beautiful_format = json.dumps(decision_tree.tree,indent=4)
    # print(beautiful_format)

    decision_tree.visualize_tree()

    decision_tree.classify(output_file=TRAIN_OUT_FILE,dataset=train_dataset)
    decision_tree.classify(output_file=TEST_OUT_FILE,dataset=test_dataset)

    with open(TRAIN_OUT_FILE,'r' ) as f:
        prediction_col = f.read().splitlines()
    true_col = train_data_loader.get_all_labels()
    train_evaluator = Evaluator(true_col,prediction_col)
    train_error_rate = train_evaluator.get_error_rate()

    with open(TEST_OUT_FILE,'r') as f:
        prediction_col = f.read().splitlines()
    true_col = test_data_loader.get_all_labels()
    test_evaluator = Evaluator(true_col,prediction_col)
    test_error_rate = test_evaluator.get_error_rate()

    with open(METRICS_OUT_FILE,'w') as f:
        f.write(f"error(train): {train_error_rate}\n")
        f.write(f"error(test): {test_error_rate}\n")


