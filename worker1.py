import time
import grpc
from concurrent import futures
import simple_tensorflow as stf

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT = '2225'


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    stf.register_worker(grpcServer)
    grpcServer.add_insecure_port('[::]:{}'.format(_PORT))
    grpcServer.start()
    print("start server at port {} ...".format(_PORT))
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()
