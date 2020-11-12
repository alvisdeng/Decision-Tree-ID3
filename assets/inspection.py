import os
import sys
from collections import Counter
from dataLoader import DataLoader
import numpy as np

class Inspection():
    def __init__(self,data_loader):
        self.data_loader = data_loader
        self.dataset = data_loader.get_dataset()
        self.labels = self.data_loader.get_all_labels()
        self.vote_result = None
        self.entropy = 0
        self.error_rate = 0

    def majority_vote(self):
        counter = Counter(self.labels)
        self.vote_result = counter.most_common()[0][0]

    def get_entropy(self):
        num_of_instances = len(self.labels)
        counter = Counter(self.labels)
        label_count_pairs = counter.most_common()

        result = 0
        for pair in label_count_pairs:
            result += -pair[1]/num_of_instances*np.log2(pair[1]/num_of_instances)
        self.entropy = result
    
    def get_error_rate(self):
        num_of_instances = len(self.labels)
        wrong = 0
        for label in self.labels:
            if label != self.vote_result:
                wrong += 1
        self.error_rate = wrong/num_of_instances
    
    def evaluate(self):
        self.majority_vote()
        self.get_entropy()
        self.get_error_rate()

        return self.entropy, self.error_rate, self.vote_result

if __name__ == "__main__":
    # Ensure exactly 3 arguments
    if len(sys.argv) != 3:
        print('USAGE: python inspection.py TRAIN_INPUT_FILE INSPECTION_OUT_FILE')
        sys.exit(1)
    
    TRAIN_INPUT_FILE = sys.argv[1]
    INSPECTION_OUT_FILE = sys.argv[2]

    # Check the input file type
    if not TRAIN_INPUT_FILE.endswith('.tsv'):
        print('Error: TRAIN_INPUT_FILE must be .tsv files')
        sys.exit(1)
    
    if not INSPECTION_OUT_FILE.endswith('.txt'):
        print('Error: INSPECTION_OUT_FILE must be .txt file')
        sys.exit(1)
    
    # Load the input file
    data_loader = DataLoader()
    data_loader.load_data(TRAIN_INPUT_FILE)

    inspection = Inspection(data_loader)
    entropy, error_rate, _ = inspection.evaluate()

    # Output the result
    with open(INSPECTION_OUT_FILE,mode='w+') as f:
        f.write('entropy: ' + str(entropy) + '\n')
        f.write('error: ' + str(error_rate))
