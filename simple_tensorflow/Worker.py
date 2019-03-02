import time

from .utils.utils import *


def register_worker(grpc_server):
    tf_pb2_grpc.add_workerServicer_to_server(Worker(), grpc_server)


fuc_map = {"softmax":softmax, "sigmoid": sigmoid, "cross": compute_grad, "square_error": compute_grad_s }


class Worker(tf_pb2_grpc.workerServicer):
    def __init__(self):
        self.ps = None
        self.act_fuc = "softmax"
        self.loss_fuc = "cross"
        self.para_pool = {}

    def register_cluster(self, request, context):
        ps_addr = request.ps
        self.para_pool = {}
        self.ps = connect_ps(ps_addr)
        return tf_pb2.status(code=200, mes="worker register cluster success")

    def register_graph(self, request, context):
        self.act_fuc = request.active_fuc
        self.loss_fuc = request.loss_fuc
        return tf_pb2.status(code=200, mes="worker register graph success")

    def run(self, request, context):
        step = request.step
        x = to_value(request.x)
        y = to_value(request.y)
        paras = self.ps.send_paras(tf_pb2.step(step=step-1))
        w = to_value(paras.w)
        b = to_value(paras.b)
        grad_w, grad_b, y_ ,cross_entropy =fuc_map[self.loss_fuc](x, y, w, b)
        print("step {},cross_entropy：{}".format(step, cross_entropy))
        self.para_pool[step] = (tf_pb2.paras(w=to_array_proto(grad_w), b=to_array_proto(grad_b)))
        return tf_pb2.res(output=to_array_proto(y_))

    def send_paras(self, request, context):
        step = request.step
        while step not in self.para_pool.keys():
            time.sleep(0.1)
        return self.para_pool[step]