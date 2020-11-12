import numpy as np

class DataLoader():
    def __init__(self):
        self.dataset = None
        self.head = None
        self.label1 = None
        self.label2 = None
        self.all_labels = None

    def load_data(self,input_file):
        self.dataset = np.loadtxt(fname=input_file,dtype=np.unicode_,delimiter='\t')
        self.head = self.dataset[0,:]
        self.all_labels = self.dataset[1:,-1]

        labels_set = set(self.all_labels)

        if len(labels_set) == 0:
            print("Sorry, there's no label in the dataset")
        elif len(labels_set) == 1:
            self.label1 = list(labels_set)[0] 
        elif len(labels_set) == 2:
            self.label1, self.label2 = labels_set
        else:
            print("Sorry, this is binary classifer")
    
    def get_dataset(self):
        return self.dataset
    
    def get_head(self):
        return self.head
    
    def get_column(self,col_idx):
        return self.dataset[:,col_idx]
    
    def get_set_labels(self):
        return self.label1, self.label2
    
    def get_all_labels(self):
        return self.all_labels

if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.load_data('small_test.tsv')

    dataset = data_loader.get_dataset()
    head = data_loader.get_head()
    set_labels = data_loader.get_set_labels()
    all_labels = data_loader.get_all_labels()
    first_column = data_loader.get_column(0)

    print(f"The dataset's shape is(including head): {dataset.shape}")
    print("The dataset's head is:")
    print(head)
    print("The dataset has following labels:")
    print(set_labels)
    print("The dataset's first column is(including head):")
    print(first_column)