import grpc
from concurrent import futures
import time
import simple_tensorflow as stf

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT = '2222'


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))  # 创建grpcServer
    stf.register_master(grpcServer)  # 注册master服务到该grpcServer上
    grpcServer.add_insecure_port('[::]:{}'.format(_PORT)) #绑定端口
    grpcServer.start()   # 开始运行服务
    print("start server at port {} ...".format(_PORT))
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()
