""" DTOs between the CVOps C library and the Python library """
import ctypes
import cvops.schemas

c_float_p = ctypes.POINTER(ctypes.c_float)


class InferenceSessionRequest(ctypes.Structure):
    """ Creates a request to start an inference session """
    _fields_ = [
        ("model_platform", ctypes.c_int),
        ("model_path", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("confidence_threshold", ctypes.c_float),
        ("iou_threshold", ctypes.c_float),
    ]


c_inference_session_request_p = ctypes.POINTER(InferenceSessionRequest)


class InferenceRequest(ctypes.Structure):
    """ Class to represent an inference request """
    _fields_ = [
        ("bytes", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("size", ctypes.c_int),
        ("draw_detections", ctypes.c_bool),
    ]


c_inference_request_p = ctypes.POINTER(InferenceRequest)


class Box(ctypes.Structure):
    """ Returns a Box object """
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("class_id", ctypes.c_int),
        ("class_name", ctypes.c_char_p),
        ("object_id", ctypes.c_int),
        ("confidence", ctypes.c_float),
    ]


c_box_p = ctypes.POINTER(Box)


class InferenceResult(ctypes.Structure):
    """ Response for an inference request  from the C library """
    _fields_ = [
        ("boxes", c_box_p),
        ("boxes_count", ctypes.c_int),
        ("image", ctypes.c_char_p),
        ("image_size", ctypes.c_int),
        ("image_width", ctypes.c_int),
        ("image_height", ctypes.c_int),
        ("milliseconds", ctypes.c_float),
    ]


c_inference_result_p = ctypes.POINTER(InferenceResult)


def dispose_inference_result(result: c_inference_result_p) -> None:
    """ Disposes the inference result
    Notes:
        * This is called by the python garbage collector when the object is no longer referenced
        * Import is inside the function to avoid circular imports
    """
    import cvops.inference.loader  # pylint: disable=import-outside-toplevel, redefined-outer-name
    dll = cvops.inference.loader.get_dll_instance()
    dll.dispose_inference_result(result)


c_inference_result_p.__del__ = dispose_inference_result


class IInferenceManager(ctypes.Structure):
    """ Interface for the inference manager """


c_i_inference_manager_p = ctypes.POINTER(IInferenceManager)


MODEL_PLATFORM_C_MAP = {
    cvops.schemas.ModelPlatforms.YOLO: 1,
    cvops.schemas.ModelPlatforms.DETECTRON: 2,
}
