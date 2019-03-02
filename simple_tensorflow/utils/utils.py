import grpc
import numpy as np
from ..utils import tf_pb2, tf_pb2_grpc


def connect_master(addr):
    conn = grpc.insecure_channel(addr)
    return tf_pb2_grpc.masterStub(channel=conn)


def connect_worker(addr):
    conn = grpc.insecure_channel(addr)
    return tf_pb2_grpc.workerStub(channel=conn)


def connect_ps(addr):
    conn = grpc.insecure_channel(addr)
    return tf_pb2_grpc.psStub(channel=conn)


def to_array_proto(value):
    np_value = np.array(value)
    shape = list(np_value.shape)
    if len(shape) == 0:
        shape.append(0)
    value = np_value.flatten().tolist()
    return tf_pb2.array(value=value, shape=shape)


def to_value(proto):
    value = proto.value
    shape = proto.shape
    if len(shape) == 1 and shape[0] == 0:
        shaped_value = value[0]
    else:
        shaped_value = (np.array(value).reshape(tuple(shape))).tolist()
    return shaped_value


def gene_data_iter(inputs, labels):
    num = len(inputs)
    for i in range(num):
        yield tf_pb2.data(x=to_array_proto(inputs[i]), y=to_array_proto(labels[i]))


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def compute_grad(x, y, w, b):
    y_ = softmax(np.matmul(x, w) + b)
    cross_entropy = -np.sum(y * np.log(y_))
    grad_w = np.matmul(np.transpose(x), y - y_)
    grad_b = np.sum(y - y_)
    for i in range(len(y_)):
        output = np.where(y_[i] == np.max(y_[i]))[0][0]
        y_[i] = [0]*len(y_[i])
        y_[i][output]=1
    return grad_w, grad_b, y_, cross_entropy


def compute_grad_s(x, y, w, b):
    y_ = sigmoid(np.matmul(x, w) + b)
    square_error = 0.5 * np.square(y - y_).sum()
    print(square_error)
    grad_b = (y_ - y) * (1 - y_) * y_
    grad_w = np.transpose(x) * grad_b
    return grad_w, grad_b, y_