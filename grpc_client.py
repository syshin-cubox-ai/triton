import grpc
import numpy as np
import tritonclient.grpc as grpcclient

url = '211.168.94.232:8001'
model_name = 'scrfd_2.5g_bnkps'
model_version = ''

channel = grpc.insecure_channel(url)
grpc_stub = grpcclient.service_pb2_grpc.GRPCInferenceServiceStub(channel)

# Health
try:
    request = grpcclient.service_pb2.ServerLiveRequest()
    response = grpc_stub.ServerLive(request)
    print("server {}".format(response))
except Exception as ex:
    print(ex)

request = grpcclient.service_pb2.ServerReadyRequest()
response = grpc_stub.ServerReady(request)
print("server {}".format(response))

request = grpcclient.service_pb2.ModelReadyRequest(name=model_name,
                                                   version=model_version)
response = grpc_stub.ModelReady(request)
print("model {}".format(response))

# Metadata
request = grpcclient.service_pb2.ServerMetadataRequest()
response = grpc_stub.ServerMetadata(request)
print("server metadata:\n{}".format(response))

request = grpcclient.service_pb2.ModelMetadataRequest(name=model_name,
                                                      version=model_version)
response = grpc_stub.ModelMetadata(request)
print("model metadata:\n{}".format(response))

# Configuration
request = grpcclient.service_pb2.ModelConfigRequest(name=model_name,
                                                    version=model_version)
response = grpc_stub.ModelConfig(request)
print("model config:\n{}".format(response))

# 클라이언트 인스턴스 생성
triton_client = grpcclient.InferenceServerClient(url)

# 입출력 설정
input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
inputs = [grpcclient.InferInput('img', list(input_data.shape), 'FP32')]
inputs[0].set_data_from_numpy(input_data)
outputs = [grpcclient.InferRequestedOutput('pred')]

# 추론
result = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
statistics = triton_client.get_inference_statistics(model_name=model_name)
print(statistics)
if len(statistics.model_stats) != 1:
    print('FAILED: Inference Statistics')
    exit(1)

# 결과 받기
output_data = result.as_numpy('pred')
