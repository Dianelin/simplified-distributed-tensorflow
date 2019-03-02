from .utils.utils import *


class Client(object):
    def __init__(self):
        self.master = None

    def register_cluster(self, cluster_def):
        self.master = connect_master(cluster_def['master'])
        try:
            status = self.master.register_cluster(
                tf_pb2.cluster(ps=cluster_def['ps'], workers=cluster_def['workers']))
            print(status.mes)
        except grpc._channel._Rendezvous as e:
            print("master服务异常，请检查")
            exit(0)

    def register_graph(self, w, b, c, active_fuc, loss_fuc):
        try:
            ps = tf_pb2.ps_def(w=to_array_proto(w), b=to_array_proto(b), c=c)
            ws = tf_pb2.ws_def(active_fuc=active_fuc, loss_fuc=loss_fuc)
            status = self.master.register_graph(tf_pb2.graph_def(ps=ps, ws=ws))
            print(status.mes)
        except grpc._channel._Rendezvous as e:
            print("master服务异常，请检查")
            exit(0)

    def train(self, inputs, labels):
        try:
            result = self.master.train(tf_pb2.data(x=to_array_proto(inputs),y=to_array_proto(labels)))
            return to_value(result.output)
        except grpc._channel._Rendezvous as e:
            print("master服务异常，请检查")
            exit(0)
