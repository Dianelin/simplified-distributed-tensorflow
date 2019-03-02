import random

import simple_tensorflow as stf
import numpy as np
import csv


def load_data(filename):
    with open(filename, 'r')as f:
        reader = csv.reader(f)
        header_row = next(reader)
        get_labels = []
        get_inputs = []
        for row in reader:
            label = int(row[-1])
            get_labels.append([int(label == 0),int(label == 1),int(label == 2)])
            input_data = [(float(row[0])-4)/4, (float(row[1])-2)/2.5, (float(row[2])-1)/6.0, float(row[3])/2.5]
            get_inputs.append(input_data)
        cc = list(zip(get_inputs, get_labels))
        random.shuffle(cc)
        get_inputs[:], get_labels[:] = zip(*cc)
    return get_inputs, get_labels


cluster = {"master": "localhost:2222",
           "workers": ["localhost:2223","localhost:2225"],
           "ps": "localhost:2224"}

inputs, labels = load_data("data/iris.csv")

train_inputs = inputs[0:120]
train_labels = labels[0:120]
test_inputs = inputs[120:]
test_labels = labels[120:]


w = np.zeros(([4, 3]))
b = np.zeros([3])
c = 0.05

client = stf.Client()
client.register_cluster(cluster)
client.register_graph(w, b, c, active_fuc="softmax", loss_fuc="cross")
batch_size = 120
for j in range(500):
    count = 0
    for i in range(int(len(train_inputs)/batch_size)):
        x_batch = train_inputs[i*batch_size:batch_size*(i+1)]
        y_batch =  train_labels[i*batch_size:batch_size*(i+1)]
        y_ = (client.train(x_batch, y_batch))
        for i in range(len(y_)):
            if y_batch[i] == y_[i]:
                count += 1
    print("cycle {} : accuracy {}".format(j, count/len(train_inputs)))

predict = client.train(test_inputs, test_labels)
count = 0.0
for i in range(len(predict)):
    if test_labels[i] == predict[i]:
        count += 1
print(predict)
print("predict accuracy: {}".format(count/len(test_labels)))
