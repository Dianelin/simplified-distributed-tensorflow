from .utils.utils import *


def register_master(grpc_server):
    tf_pb2_grpc.add_masterServicer_to_server(Master(), grpc_server)


class Master(tf_pb2_grpc.masterServicer):
    def __init__(self):
        self.ps = None
        self.step = 0
        self.workers = []

    def register_cluster(self, request, context):
        ps_addr = request.ps
        self.step=0
        self.workers = []
        self.ps = connect_ps(ps_addr)
        print(self.ps.register_cluster(request).mes)
        for ws_addr in request.workers:
            worker = connect_worker(ws_addr)
            print(worker.register_cluster(request).mes)
            self.workers.append(worker)
        return tf_pb2.status(code=200, mes="master register cluster success")

    def register_graph(self, request, context):
        self.ps.register_graph(request.ps)
        for worker in self.workers:
            worker.register_graph(request.ws)
        return tf_pb2.status(code=200, mes="master register graph success")

    def train(self, request, context):
        self.step += 1
        inputs = to_value(request.x)
        labels = to_value(request.y)
        n_workers = len(self.workers)
        n =int(len(inputs) / n_workers)
        y = [0] * n_workers
        for i in range(n_workers - 1):
            y[i] = self.workers[i].run.future(
                tf_pb2.step_data(step=self.step, x=to_array_proto(inputs[i * n:(i + 1) * n]),
                                 y=to_array_proto(labels[i * n: (i + 1) * n]))).result().output

        y[-1] = self.workers[-1].run.future(tf_pb2.step_data(step=self.step, x=to_array_proto(inputs[(n_workers-1)*n:-1]),
                                 y=to_array_proto(labels[(n_workers-1)*n:-1]))).result().output
        output = []
        status = self.ps.run.future(tf_pb2.step(step=self.step))
        print(status.result().mes)
        for i in range(n_workers):
            output += to_value(y[i])
        return tf_pb2.res(output=to_array_proto(output))
