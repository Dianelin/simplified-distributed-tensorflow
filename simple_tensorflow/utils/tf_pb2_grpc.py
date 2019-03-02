# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from ..utils import tf_pb2 as tf__pb2


class masterStub(object):
  """表示Master服务类，定义其可供远程调用的方法
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.register_cluster = channel.unary_unary(
        '/protos.master/register_cluster',
        request_serializer=tf__pb2.cluster.SerializeToString,
        response_deserializer=tf__pb2.status.FromString,
        )
    self.register_graph = channel.unary_unary(
        '/protos.master/register_graph',
        request_serializer=tf__pb2.graph_def.SerializeToString,
        response_deserializer=tf__pb2.status.FromString,
        )
    self.train = channel.unary_unary(
        '/protos.master/train',
        request_serializer=tf__pb2.data.SerializeToString,
        response_deserializer=tf__pb2.res.FromString,
        )


class masterServicer(object):
  """表示Master服务类，定义其可供远程调用的方法
  """

  def register_cluster(self, request, context):
    """初始化集群，建立与ps,worker间连接
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def register_graph(self, request, context):
    """请求将图分别注册到ps、worker中
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def train(self, request, context):
    """请求master执行一次训练过程
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_masterServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'register_cluster': grpc.unary_unary_rpc_method_handler(
          servicer.register_cluster,
          request_deserializer=tf__pb2.cluster.FromString,
          response_serializer=tf__pb2.status.SerializeToString,
      ),
      'register_graph': grpc.unary_unary_rpc_method_handler(
          servicer.register_graph,
          request_deserializer=tf__pb2.graph_def.FromString,
          response_serializer=tf__pb2.status.SerializeToString,
      ),
      'train': grpc.unary_unary_rpc_method_handler(
          servicer.train,
          request_deserializer=tf__pb2.data.FromString,
          response_serializer=tf__pb2.res.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'protos.master', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class psStub(object):
  """ParameterServer服务类
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.register_cluster = channel.unary_unary(
        '/protos.ps/register_cluster',
        request_serializer=tf__pb2.cluster.SerializeToString,
        response_deserializer=tf__pb2.status.FromString,
        )
    self.register_graph = channel.unary_unary(
        '/protos.ps/register_graph',
        request_serializer=tf__pb2.ps_def.SerializeToString,
        response_deserializer=tf__pb2.status.FromString,
        )
    self.run = channel.unary_unary(
        '/protos.ps/run',
        request_serializer=tf__pb2.step.SerializeToString,
        response_deserializer=tf__pb2.status.FromString,
        )
    self.send_paras = channel.unary_unary(
        '/protos.ps/send_paras',
        request_serializer=tf__pb2.step.SerializeToString,
        response_deserializer=tf__pb2.paras.FromString,
        )


class psServicer(object):
  """ParameterServer服务类
  """

  def register_cluster(self, request, context):
    """初始化集群，建立与workers间连接
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def register_graph(self, request, context):
    """注册图，即初始化参数
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def run(self, request, context):
    """运行一次step，将参数发送给workers
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def send_paras(self, request, context):
    """接收并存储worker计算出的参数梯度
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_psServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'register_cluster': grpc.unary_unary_rpc_method_handler(
          servicer.register_cluster,
          request_deserializer=tf__pb2.cluster.FromString,
          response_serializer=tf__pb2.status.SerializeToString,
      ),
      'register_graph': grpc.unary_unary_rpc_method_handler(
          servicer.register_graph,
          request_deserializer=tf__pb2.ps_def.FromString,
          response_serializer=tf__pb2.status.SerializeToString,
      ),
      'run': grpc.unary_unary_rpc_method_handler(
          servicer.run,
          request_deserializer=tf__pb2.step.FromString,
          response_serializer=tf__pb2.status.SerializeToString,
      ),
      'send_paras': grpc.unary_unary_rpc_method_handler(
          servicer.send_paras,
          request_deserializer=tf__pb2.step.FromString,
          response_serializer=tf__pb2.paras.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'protos.ps', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class workerStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.register_cluster = channel.unary_unary(
        '/protos.worker/register_cluster',
        request_serializer=tf__pb2.cluster.SerializeToString,
        response_deserializer=tf__pb2.status.FromString,
        )
    self.register_graph = channel.unary_unary(
        '/protos.worker/register_graph',
        request_serializer=tf__pb2.ws_def.SerializeToString,
        response_deserializer=tf__pb2.status.FromString,
        )
    self.run = channel.unary_unary(
        '/protos.worker/run',
        request_serializer=tf__pb2.step_data.SerializeToString,
        response_deserializer=tf__pb2.res.FromString,
        )
    self.send_paras = channel.unary_unary(
        '/protos.worker/send_paras',
        request_serializer=tf__pb2.step.SerializeToString,
        response_deserializer=tf__pb2.paras.FromString,
        )


class workerServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def register_cluster(self, request, context):
    """初始化集群，建立与ps间连接
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def register_graph(self, request, context):
    """注册图，即选择激活函数和损失函数
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def run(self, request, context):
    """执行计算过程，计算后将结果发送给ps
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def send_paras(self, request, context):
    """接受ps传送的参数
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_workerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'register_cluster': grpc.unary_unary_rpc_method_handler(
          servicer.register_cluster,
          request_deserializer=tf__pb2.cluster.FromString,
          response_serializer=tf__pb2.status.SerializeToString,
      ),
      'register_graph': grpc.unary_unary_rpc_method_handler(
          servicer.register_graph,
          request_deserializer=tf__pb2.ws_def.FromString,
          response_serializer=tf__pb2.status.SerializeToString,
      ),
      'run': grpc.unary_unary_rpc_method_handler(
          servicer.run,
          request_deserializer=tf__pb2.step_data.FromString,
          response_serializer=tf__pb2.res.SerializeToString,
      ),
      'send_paras': grpc.unary_unary_rpc_method_handler(
          servicer.send_paras,
          request_deserializer=tf__pb2.step.FromString,
          response_serializer=tf__pb2.paras.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'protos.worker', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))