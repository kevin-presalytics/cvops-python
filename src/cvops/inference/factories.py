#  Factories to generate schemas objects from C types
import ctypes
import cvops.schemas
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
            height = c_inference_result.boxes[i].height,
            width = c_inference_result.boxes[i].width,
            x = c_inference_result.boxes[i].x,
            y = c_inference_result.boxes[i].y,
            class_id = c_inference_result.boxes[i].class_id,
            class_name = c_inference_result.boxes[i].class_name,
            object_id = c_inference_result.boxes[i].object_id,
            confidence = c_inference_result.boxes[i].confidence
        )
        boxes.append(next_box)
        label = cvops.schemas.Label(
            class_id = c_inference_result.boxes[i].class_id,
            class_name = c_inference_result.boxes[i].class_name
        )
        if label not in labels:
            labels.append(label)
    return cvops.schemas.InferenceResult(
        boxes=boxes,
        labels=labels,
        result_type=result_type
    )