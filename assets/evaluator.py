class Evaluator():
    def __init__(self,true_col,prediction_col):
        self.true_col = true_col
        self.prediction_col = prediction_col
    
    def get_error_rate(self):
        total = len(self.true_col)
        wrong = 0
        for i in range(total):
            if self.true_col[i] != self.prediction_col[i]:
                wrong += 1
        
        return wrong/total

