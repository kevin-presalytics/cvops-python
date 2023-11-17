#  Factories to generate schemas objects from C types
import typing
import ctypes
import pathlib
import json
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
