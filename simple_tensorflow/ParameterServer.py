import time

from .utils.utils import *


def register_parameterServer(grpc_server):
    tf_pb2_grpc.add_psServicer_to_server(ParameterServer(), grpc_server)


class ParameterServer(tf_pb2_grpc.psServicer):
    def __init__(self):
        self.workers = []
        self.para_pool = {}
        self.w = None
        self.b = None
        self.c = None

    def register_cluster(self, request, context):
        self.para_pool = {}
        for ws_addr in request.workers:
            self.workers.append(connect_worker(ws_addr))
        return tf_pb2.status(code=200, mes="ps register cluster success")

    def register_graph(self, request, context):
        self.w = to_value(request.w)
        self.b = to_value(request.b)
        self.c = request.c
        self.para_pool[0] = tf_pb2.paras(w=request.w, b=request.b)
        return tf_pb2.status(code=200, mes="ps register graph success")

    def run(self, request, context):
        step = request.step
        n = len(self.workers)
        w_grads = [0]*n
        b_grads=[0]*n
        for i in range(n):
            res = self.workers[i].send_paras.future(request).result()
            w_grads[i] = to_value(res.w)
            b_grads[i] = to_value(res.b)
        total_w, total_b = w_grads[0], b_grads[0]
        for i in range(n-1):
            total_w = np.add(total_w,w_grads[i+1])
            total_b = np.add(total_b, b_grads[i+1])

        self.w += np.array(total_w) * self.c / n
        self.b += np.array(total_b) * self.c / n
        print("step {}  W更新为：\n {} \n b更新为：\n {} \n".format(step,self.w,self.b))

        self.para_pool[step]=tf_pb2.paras(w=to_array_proto(self.w),b=to_array_proto(self.b))
        return tf_pb2.status(code=200, mes="successfully update paras in step {}".format(step))

    def send_paras(self, request, context):
        step = request.step
        while step not in self.para_pool.keys():
            time.sleep(0.1)
        return self.para_pool[step]
