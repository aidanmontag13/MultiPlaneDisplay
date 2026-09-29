import numpy as np
import cv2
import hailo_platform as hpf
from hailo_platform import VDevice

hef = hpf.HEF("yolov8m_pose.hef")

with hpf.VDevice() as target:
    configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]

    input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
    output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

    input_shape = input_vstream_info.shape
    output_shape = output_vstream_info.shape

    print(f"Input shape: {input_shape}, Output shape: {output_shape}")

    with network_group.activate(network_group_params):
        with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            for _ in range(10):
                random_input = np.random.rand(*input_shape).astype(np.float32)
                input_data = {input_vstream_info.name: np.expand_dims(random_input, axis=0)}
                results = infer_pipeline.infer(input_data)
                output_data = results[output_vstream_info.name]
                print(f"Inference output: {output_data}")