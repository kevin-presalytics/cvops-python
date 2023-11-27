#  Factories to generate schemas objects from C types
import typing
import ctypes
import pathlib
import json
import numpy
import cvops.schemas
import cvops.image_processor
import cvops.inference.c_interfaces as _types


def inference_result_from_c_type(c_inference_result: _types.InferenceResult) -> cvops.schemas.InferenceResult:
    """ Converts an InferenceResult from the C library to a serializable type to send over the data topic """
    if c_inference_result is None:
        raise ValueError("Inference result cannot be None")
    elif isinstance(c_inference_result, _types.c_inference_result_p) or isinstance(c_inference_result, ctypes.POINTER(_types.InferenceResult)):
        c_inference_result = c_inference_result.contents
    if not isinstance(c_inference_result, _types.InferenceResult):
        raise TypeError()
    boxes = []
    labels = []
    result_type = cvops.schemas.InferenceResultTypes.BOXES
    # TODO: Add support for meshes
    for i in range(0, c_inference_result.boxes_count):
        next_box = cvops.schemas.Box(
            height=c_inference_result.boxes[i].height,
            width=c_inference_result.boxes[i].width,
            x=c_inference_result.boxes[i].x,
            y=c_inference_result.boxes[i].y,
            class_id=c_inference_result.boxes[i].class_id,
            class_name=c_inference_result.boxes[i].class_name,
            object_id=c_inference_result.boxes[i].object_id,
            confidence=c_inference_result.boxes[i].confidence
        )
        boxes.append(next_box)
        label = cvops.schemas.Label(
            class_id=c_inference_result.boxes[i].class_id,
            class_name=c_inference_result.boxes[i].class_name
        )
        if label not in labels:
            labels.append(label)
    return cvops.schemas.InferenceResult(
        boxes=boxes,
        labels=labels,
        result_type=result_type,
        milliseconds=c_inference_result.milliseconds
    )


def inference_result_to_c_type(inference_result: cvops.schemas.InferenceResult) -> _types.InferenceResult:
    """ Converts a cvops.schemas.InferenceResult to a C type for rendering"""
    boxes_list = []
    for box in inference_result.boxes:
        c_box = _types.Box(
            height=ctypes.c_int(box.height),
            width=ctypes.c_int(box.width),
            x=ctypes.c_int(box.x),
            y=ctypes.c_int(box.y),
            class_id=ctypes.c_int(box.class_id),
            class_name=ctypes.c_char_p(box.class_name.encode('utf-8')),
            object_id=ctypes.c_int(box.object_id),
            confidence=ctypes.c_float(box.confidence)
        )
        boxes_list.append(c_box)
    c_inference_result = _types.InferenceResult(
        boxes=(_types.Box * len(inference_result.boxes))(*boxes_list),
        boxes_count=ctypes.c_int(len(inference_result.boxes)),
        milliseconds=ctypes.c_float(inference_result.milliseconds),
        image=None,
        image_size=ctypes.c_int(0),
        image_width=ctypes.c_int(0),
        image_height=ctypes.c_int(0),
    )
    return c_inference_result


def inference_result_to_c_type_ptr(inference_result: cvops.schemas.InferenceResult) -> _types.c_inference_result_p:
    """ Converts a cvops.schemas.InferenceResult to a C type for rendering"""
    c_inference_result = inference_result_to_c_type(inference_result)
    return ctypes.pointer(c_inference_result)


def get_c_model_platform(model_platform: cvops.schemas.ModelPlatforms) -> ctypes.c_int:
    """ Returns the C representation of the model Framework """
    model_platform = _types.MODEL_PLATFORM_C_MAP.get(model_platform, None)
    if model_platform is None:
        raise Exception("Invalid model platform")  # pylint: disable=broad-exception-raised
    return ctypes.c_int(model_platform)


def create_inference_session_request(
    model_platform: cvops.schemas.ModelPlatforms,
    model_path: pathlib.Path,
    metadata: typing.Dict[str, typing.Any],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> _types.InferenceSessionRequest:
    """ Creates an InferenceSessionRequest  from the given parameters """
    c_model_platform = get_c_model_platform(model_platform)
    c_model_path = ctypes.c_char_p(str(model_path.resolve()).encode('utf-8'))
    c_metadata = ctypes.c_char_p(json.dumps(metadata).encode('utf-8'))
    c_confidence_threshold = ctypes.c_float(confidence_threshold)
    c_iou_threshold = ctypes.c_float(iou_threshold)

    request = _types.InferenceSessionRequest(
        model_platform=c_model_platform,
        model_path=c_model_path,
        metadata=c_metadata,
        confidence_threshold=c_confidence_threshold,
        iou_threshold=c_iou_threshold,
    )
    return request


def tracker_state_ptr_to_boxes(tracker_state: _types.c_tracker_state_p) -> typing.List[cvops.schemas.Box]:
    """ Converts a tracker state to a list of boxes """
    boxes = []
    for i in range(0, tracker_state.contents.boxes_count):
        next_box = cvops.schemas.Box(
            height=tracker_state.contents.boxes[i].height,
            width=tracker_state.contents.boxes[i].width,
            x=tracker_state.contents.boxes[i].x,
            y=tracker_state.contents.boxes[i].y,
            class_id=tracker_state.contents.boxes[i].class_id,
            class_name=tracker_state.contents.boxes[i].class_name,
            object_id=tracker_state.contents.boxes[i].object_id,
            confidence=tracker_state.contents.boxes[i].confidence
        )
        boxes.append(next_box)
    return boxes


def frame_to_cv_mat_data(frame: numpy.ndarray) -> typing.Tuple[ctypes.c_void_p, int, int, int]:
    """ Converts an image into values that can be passed to the cv::Mat constructor in the C Library """
    num_channels = frame.shape[-1] if frame.ndim == 3 else 1
    return (frame.ctypes._data, frame.shape[0], frame.shape[1], num_channels)  # pylint: disable=protected-access
