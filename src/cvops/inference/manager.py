import typing
import ctypes
import json
import pathlib
import numpy
import cv2
import cvops.schemas
import cvops.inference
import cvops.inference.c_api
import cvops.inference.c_interfaces as _types
import cvops.inference.factories

MODEL_PLATFORM_C_MAP = {
    cvops.schemas.ModelPlatforms.YOLO: 1,
    cvops.schemas.ModelPlatforms.DETECTRON: 2,
}

class InferenceSessionManager(cvops.inference.c_api.CApi):
    """ Wrapper class around the calls to inference methods of the C API """
    session: typing.Optional[ctypes.POINTER(cvops.inference.c_interfaces.IInferenceManager)]

    def __int__(self):
        super().__init__()
        self.session = None

    def _start_session(self, request: cvops.inference.c_interfaces.InferenceSessionRequest):
        self.session = self.dll.start_inference_session(request)
        if self.session is None:
            raise Exception("Unable to start inference session")  # pylint: disable=broad-exception-raised
    
    def start_session(self,
                      model_platform: cvops.schemas.ModelPlatforms,
                      model_path: pathlib.Path,
                      metadata: typing.Dict[str, typing.Any],
                      confidence_threshold: float = 0.5,
                      iou_threshold: float = 0.5) -> None:
        """ Starts an inference session with the given parameters """
        c_model_platform = self._get_c_model_platform(model_platform)
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
        self._start_session(request)

    def _get_c_model_platform(self, model_platform: cvops.schemas.ModelPlatforms) -> ctypes.c_int:
        """ Returns the C representation of the model Framework """
        model_platform = MODEL_PLATFORM_C_MAP.get(model_platform, None)
        if model_platform is None:
            raise Exception("Invalid model platform")  # pylint: disable=broad-exception-raised
        return ctypes.c_int(model_platform)

    def _run_inference(self, request: cvops.inference.c_interfaces.InferenceRequest) -> cvops.inference.c_interfaces.InferenceResult:
        result = _types.InferenceResult()
        self.dll.run_inference(self.session, request, result)
        return result

    
    def run_inference(self, 
                      image: typing.Union[numpy.ndarray, bytes], 
                      name: str = "", 
                      draw_detections: bool = False
                      ) -> cvops.inference.c_interfaces.InferenceResult:
        """ Runs inference on the given image """
        if isinstance(image, numpy.ndarray):
            image = cv2.imencode('.png', image)[1].tobytes().encode('utf-8')  # pylint: disable=no-member
        if not isinstance(image, bytes):
            raise TypeError("Image must be a numpy array or bytes")  # pylint: disable=broad-except
        c_image = ctypes.c_char_p(image)
        c_name = ctypes.c_char_p(name.encode('utf-8'))
        c_size = ctypes.c_int(len(image))
        c_draw_detections = ctypes.c_bool(draw_detections)

        request = _types.InferenceRequest(
            bytes=c_image,
            name=c_name,
            size=c_size,
            draw_detections=c_draw_detections,
        )
        c_inference_result = self._run_inference(request)
        inference_result = cvops.inference.factories.inference_result_from_c_type(c_inference_result)
        return inference_result


    def end_session(self) -> None:
        """ closes session in memory """
        self.dll.end_inference_session(self.session)
        self.session = None

    def get_error(self) -> str:
        """ Gets error message from the C Library's global error container """
        c_error = self.dll.error_message() or ""
        if isinstance(c_error, bytes):
            return c_error.decode('utf-8')
        return c_error