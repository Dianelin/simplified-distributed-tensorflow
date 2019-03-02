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
            label = int(row[0])
            get_labels.append([int(label == 0),int(label == 1)])
            input_data = [float(row[1])/800, float(row[2])/4.0, float(row[3])/4.0]
            get_inputs.append(input_data)
    return get_inputs, get_labels


cluster = {"master": "localhost:2222",
           "workers": ["localhost:2223","localhost:2225"],
           "ps": "localhost:2224"}

inputs, labels = load_data("data/student_data.csv")

train_inputs = inputs[0:300]
train_labels = labels[0:300]
test_inputs = inputs[300:]
test_labels = labels[300:]


w = np.zeros(([3, 2]))
b = np.zeros([2])
c = 0.005

client = stf.Client()
client.register_cluster(cluster)
client.register_graph(w, b, c, active_fuc="softmax", loss_fuc="cross")
for j in range(10):
    count = 0
    for i in range(10):
        x_batch = train_inputs[i*30:30*i+30]
        y_batch =  train_labels[i*30:30*i+30]
        y_ = (client.train(x_batch, y_batch))
        for i in range(len(y_)):
            if y_batch[i] == y_[i]:
                count += 1
    print("cycle {} : accuracy {}".format(j, count/len(train_inputs)))

predict = client.train(test_inputs, test_labels)
count = 0
for i in range(len(predict)):
    if test_labels[i] == predict[i]:
        count += 1
#print(predict)
print("predict accuracy: {}".format(count/len(test_labels)))
