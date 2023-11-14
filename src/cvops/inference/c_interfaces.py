import ctypes

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
    ]


c_inference_result_p = ctypes.POINTER(InferenceResult)


class IInferenceManager(ctypes.Structure):
    pass

c_i_inference_manager_p = ctypes.POINTER(IInferenceManager)



