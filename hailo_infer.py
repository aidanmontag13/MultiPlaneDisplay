from typing import Tuple, Dict, List
import os
from typing import Callable, Optional
from functools import partial
import numpy as np
import cv2
from hailo_platform import (HEF, VDevice,FormatType, HailoSchedulingAlgorithm)
from pose_estimation_utils import PoseEstPostProcessing
from hailo_platform.pyhailort.pyhailort import FormatOrder

class HailoInfer:
    def __init__(
        self, hef_path: str, batch_size: int = 1,
            input_type: Optional[str] = None, output_type: Optional[str] = None,
            priority: Optional[int] = 0) -> None:

        """
        Initialize the HailoAsyncInference class to perform asynchronous inference using a Hailo HEF model.

        Args:
            hef_path (str): Path to the HEF model file.
            batch_size (optional[int]): Number of inputs processed per inference. Defaults to 1.
            input_type (Optional[str], optional): Input data type format. Common values: 'UINT8', 'UINT16', 'FLOAT32'.
            output_type (Optional[str], optional): Output data type format. Common values: 'UINT8', 'UINT16', 'FLOAT32'.
            priority (optional[int]): Scheduler priority value for the model within the shared VDevice context. Defaults to 0.
        """
        params = VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        vDevice = VDevice(params)

        self.target = vDevice
        hef_path = os.fspath(hef_path)
        self.hef = HEF(hef_path)

        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        self._set_input_type(input_type)
        self._set_output_type(output_type)

        self.config_ctx = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(priority)
        self.last_infer_job = None


    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """

        if input_type is not None:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type: Optional[str] = None) -> None:
        """
        Set the output type for each model output.

        Args:
            output_type (Optional[str]): Desired output data type. Common values:
                'UINT8', 'UINT16', 'FLOAT32'.
        """

        self.nms_postprocess_enabled = False

        # If the model uses HAILO_NMS_WITH_BYTE_MASK format (e.g.,instance segmentation),
        if self.infer_model.outputs[0].format.order == FormatOrder.HAILO_NMS_WITH_BYTE_MASK:
            # Use UINT8 and skip setting output formats
            self.nms_postprocess_enabled = True
            self.output_type = self._output_data_type2dict("UINT8")
            return

        # Otherwise, set the format type based on the provided output_type argument
        self.output_type = self._output_data_type2dict(output_type)

        # Apply format to each output layer
        for name, dtype in self.output_type.items():
            self.infer_model.output(name).set_format_type(getattr(FormatType, dtype))


    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )

    def get_hef(self) -> HEF:
        """
        Get a HEF instance
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input


    def run(self, input_batch: List[np.ndarray], inference_callback_fn) -> object:
        """
        Run an asynchronous inference job on a batch of preprocessed inputs.

        This method reuses a preconfigured model (no reconfiguration overhead),
        prepares input/output bindings, launches async inference, and returns
        the job handle so that the caller can wait on it if needed.

        Args:
            input_batch (List[np.ndarray]): A batch of preprocessed model inputs.
            inference_callback_fn (Callable): Function to be invoked when inference is complete.
                                              It receives `bindings_list` and additional context.

        Returns:
            Async job handle returned by `run_async`, which can be used to wait for completion or check status.
        """
        bindings_list = self._create_bindings(self.configured_model, input_batch)
        self.configured_model.wait_for_async_ready(timeout_ms=10000)

        # Launch async inference and attach the result handler
        self.last_infer_job = self.configured_model.run_async(
            bindings_list,
            partial(inference_callback_fn, bindings_list=bindings_list)
        )
        return self.last_infer_job

    def _create_bindings(self, configured_model, input_batch):
        """
        Create a list of input-output bindings for a batch of frames.

        Args:
            configured_model: The configured inference model.
            input_batch (List[np.ndarray]): List of input frames, preprocessed and ready.

        Returns:
            List[Bindings]: A list of bindings for each frame's input and output buffers.
        """

        def _frame_binding(frame: np.ndarray):
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
                for name in self.output_type
            }

            binding = configured_model.create_bindings(output_buffers=output_buffers)
            binding.input().set_buffer(np.array(frame))
            return binding

        return [_frame_binding(frame) for frame in input_batch]



    def is_nms_postprocess_enabled(self) -> bool:
        """
        Returns True if the HEF model includes an NMS postprocess node.
        """
        return self.nms_postprocess_enabled

    def _output_data_type2dict(self, data_type: Optional[str]) -> Dict[str, str]:
        """
        Generate a dictionary mapping each output layer name to its corresponding
        data type. If no data type is provided, use the type defined in the HEF.

        Args:
            data_type (Optional[str]): The desired data type for all output layers.
                                       Valid values: 'float32', 'uint8', 'uint16'.
                                       If None, uses types from the HEF metadata.

        Returns:
            Dict[str, str]: A dictionary mapping output layer names to data types.
        """
        valid_types = {"float32", "uint8", "uint16"}
        data_type_dict = {}

        for output_info in self.hef.get_output_vstream_infos():
            name = output_info.name
            if data_type is None:
                # Extract type from HEF metadata
                hef_type = str(output_info.format.type).split(".")[-1]
                data_type_dict[name] = hef_type
            else:
                if data_type.lower() not in valid_types:
                    raise ValueError(f"Invalid data_type: {data_type}. Must be one of {valid_types}")
                data_type_dict[name] = data_type

        return data_type_dict


    def close(self):
        # Wait for the final job to complete before exiting
        if self.last_infer_job is not None:
            self.last_infer_job.wait(10000)

        if self.config_ctx:
            self.config_ctx.__exit__(None, None, None)

def inference_callback(completion_info, bindings_list, infer):
    if completion_info.exception:
        print("Inference error:", completion_info.exception)
        return

    outputs = {}

    for binding in bindings_list:
        for name in binding._output_names:
            outputs[name] = np.array(binding.output(name).get_buffer())

    infer.last_result = outputs

def decode_pose_outputs(outputs, score_thresh=0.5):

    import numpy as np
    import torch

    class Keypoints:
        def __init__(self, kpts):
            self.data = torch.tensor(kpts[..., :2], dtype=torch.float32)
            self.conf = torch.tensor(kpts[..., 2], dtype=torch.float32)

    class Result:
        def __init__(self, kpts):
            self.keypoints = Keypoints(kpts)

    best_score = 0
    best_kpts = None

    scales = [
        ("conv59","conv60","conv61",8),
        ("conv75","conv76","conv77",16),
        ("conv90","conv91","conv92",32)
    ]

    for box_layer, score_layer, kpt_layer, stride in scales:

        scores = outputs[f"yolov8m_pose/{score_layer}"]
        keypoints = outputs[f"yolov8m_pose/{kpt_layer}"]

        h, w, _ = scores.shape

        for y in range(h):
            for x in range(w):

                score = scores[y, x, 0]

                if score < score_thresh:
                    continue

                if score > best_score:

                    kpts = keypoints[y, x].reshape(17,3).copy()

                    for i in range(17):
                        kpts[i,0] = (kpts[i,0] * 2 + x) * stride
                        kpts[i,1] = (kpts[i,1] * 2 + y) * stride

                    best_score = score
                    best_kpts = kpts

    if best_kpts is None:
        keypoints_array = np.zeros((0,17,3), dtype=np.float32)
    else:
        keypoints_array = np.array([best_kpts], dtype=np.float32)

    return [Result(keypoints_array)]

def main():
    hef_path = "yolov8m_pose.hef"

    # Create inference engine
    infer = HailoInfer(
        hef_path=hef_path,
        batch_size=1,
        input_type="UINT8",
        output_type="FLOAT32",
        priority=0
    )

    # Create dummy input using model shape
    input_shape = infer.get_input_shape()
    print("Model input shape:", input_shape)
    dummy_frame = np.random.randint(
        0, 255, size=input_shape, dtype=np.uint8
    )
    dummy_frame = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
    dummy_frame = cv2.resize(dummy_frame, (input_shape[1], input_shape[1]))
    print("Dummy frame shape:", dummy_frame.shape)

    # Run inference
    job = infer.run([dummy_frame], partial(inference_callback, infer=infer))

    job.wait(10000)

    result = infer.last_result

    poses = decode_pose_outputs(result, score_thresh=0.5)

    for pose in poses:
        keypoints = pose["keypoints"]

        print("Keypoints shape:", keypoints.shape)
        print(keypoints)

    # Cleanup
    infer.close()


if __name__ == "__main__":
    main()