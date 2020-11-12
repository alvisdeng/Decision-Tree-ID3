import matplotlib.pyplot as plt

depth = [0,1,2,3,4,5,6,7]
train_error = [0.4430,0.2013,0.1342,0.1141,0.1074,0.0872,0.0738,0.0671]
test_error = [0.5060,0.2169,0.1566,0.1687,0.2048,0.1687,0.1928,0.2048]

plt.plot(depth,train_error, ls='--', marker='o', label='Train Error Rate')
plt.plot(depth,test_error,  ls='-', marker='v', label='Test Error Rate')
plt.xlabel('Depth')
plt.ylabel('Error Rate')
plt.title('Politicians Dataset Train/Test Error Rate')
plt.legend()
plt.show()