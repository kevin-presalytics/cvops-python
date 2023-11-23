""" Classes for Video Streaming and Processing """
import abc
import typing
import pathlib
import os
import time
import threading
import queue
import multiprocessing
import multiprocessing.queues
import logging
import cv2
import numpy
import cvops
import cvops.config
import cvops.schemas
import cvops.inference.manager
import cvops.inference.factories


logger = logging.getLogger(__name__)


class VideoPlayerBase(abc.ABC):
    """ Base class for video players """
    video_source: str
    show_video: bool
    cap: cv2.VideoCapture
    fps: float
    limit_fps: bool

    def __init__(self,
                 video_source: typing.Union[str, pathlib.Path],
                 show_video: bool = True,
                 limit_fps: bool = True,
                 **kwargs) -> None:
        super().__init__()
        if isinstance(video_source, pathlib.Path):
            video_source = str(video_source)
        self.video_source = video_source
        # This probably needs to be replaced with a `validate_source` function that checks for remote streams, etc.
        self.cap = cv2.VideoCapture(str(self.video_source))
        self.show_video = show_video
        self.limit_fps = limit_fps

        # Get openCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        # With webcam get(CV_CAP_PROP_FPS) does not work.
        # Let's see for ourselves.

        if int(major_ver) < 3:
            self.fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def image_stream(self) -> typing.Generator[numpy.ndarray, None, None]:
        """ Returns a generator that yields images from the video """
        while self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                self.process_frame(frame)
                if ret:
                    yield frame
                else:
                    self.stop()
                    break
            except KeyboardInterrupt:
                self.stop()
                break

    def play(self) -> None:
        """ Plays the video continuously at the video fps"""
        prev = time.time()
        for image in self.image_stream:
            if self.limit_fps:
                time_elapsed = time.time() - prev
                while time_elapsed < 1 / self.fps:
                    time_elapsed = time.time() - prev
                prev = time.time()
            image = self.process_frame(image)
            if self.show_video:
                cv2.imshow("Video", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop()

    def stream(self) -> None:
        """ Returns a generator that yields images from the video """
        for image in self.image_stream:
            image = self.process_frame(image)
            yield image

    def stop(self) -> None:
        """ Stops the video """
        if self.show_video:
            cv2.destroyAllWindows()
        self.show_video = False
        self.cap.release()

    @abc.abstractmethod
    def process_frame(self, frame: numpy.ndarray) -> numpy.ndarray:
        """ Processes a frame of video and returns the processed frame """
        return NotImplemented

    def __del__(self) -> None:
        try:
            self.stop()
        except BaseException:  # pylint: disable=bare-except
            pass


class InferenceThread(threading.Thread):
    """ Moves inference into a subprocess to let Video Player classes avoid binding """
    model_path: pathlib.Path
    request_queue: queue.Queue
    result_queue: queue.Queue
    debug: bool
    model_platform: cvops.schemas.ModelPlatforms
    confidence_threshold: float
    iou_threshold: float
    metadata: typing.Dict[str, typing.Any]

    def __init__(self,
                 model_path: pathlib.Path,
                 request_queue: "queue.Queue[numpy.ndarray]",
                 result_queue: "queue.Queue[cvops.schemas.InferenceResult]",
                 model_platform: cvops.schemas.ModelPlatforms,
                 metadata: typing.Dict[str, typing.Any],
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.4,
                 debug: bool = False
                 ) -> None:
        super().__init__()
        assert isinstance(model_path, pathlib.Path), "`model_path` must be a Path"
        assert model_path.exists(), "Supplied `model_path` does not exist"
        assert model_path.suffix == ".onnx", "Model must be onnx format"
        self.model_path = model_path
        assert isinstance(request_queue, queue.Queue), "`request_queue` must be a Queue"
        self.request_queue = request_queue
        assert isinstance(result_queue, queue.Queue), "`result_queue` must be a Queue"
        self.result_queue = result_queue
        self.name = "python-cvops-inference"
        self.is_listening = True
        self.debug = debug
        assert isinstance(
            model_platform, cvops.schemas.ModelPlatforms), "`model_platform` must be a ModelPlatforms enum"
        self.model_platform = model_platform
        assert isinstance(confidence_threshold, float), "`confidence_threshold` must be a float"
        self.confidence_threshold = confidence_threshold
        assert isinstance(iou_threshold, float), "`iou_threshold` must be a float"
        self.iou_threshold = iou_threshold
        assert isinstance(metadata, dict), "`metadata` must be a dict"
        self.metadata = metadata

    def run(self) -> None:
        """ Starts the inference process """
        try:
            if self.debug:
                logger.debug("Starting Inference thread %s on pid %s", self.name, os.getpid())
            session_request = cvops.inference.factories.create_inference_session_request(
                self.model_platform,
                self.model_path,
                metadata=self.metadata,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold
            )
            with cvops.inference.manager.InferenceSessionManager(session_request) as mgr:
                if self.debug:
                    self.result_queue.put(True)
                while self.is_listening:
                    try:
                        image = self.request_queue.get()
                        logger.debug("Received image from queue")
                        assert isinstance(image, numpy.ndarray), "`image` must be a numpy.ndarray"
                        c_inference_result = mgr.run_inference(
                            image,
                            "",
                            draw_detections=False
                        )
                        logger.debug("Ran inference")
                        if not c_inference_result:
                            raise RuntimeError("Inference returned NULL Pointer")
                        # Note: c-inference result is a pointer to a struct (that itself contains pointers), not the struct itself
                        # it cannot be serialized an sent over the wire, the pointer will be orphaned
                        serializable_result = cvops.inference.factories.inference_result_from_c_type(c_inference_result)
                        while not self.result_queue.empty():
                            self.result_queue.get_nowait()
                        self.result_queue.put(serializable_result)
                        mgr.dispose_inference_result(c_inference_result)
                    except Exception as ex:  # pylint: disable=broad-exception-caught
                        logger.exception(ex, "Error in Inference Process.  Failed Inference attempt.")
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.exception(ex, "Error in Inference Process. Unable to start session.")
        finally:
            logger.debug("Exiting Inference Thread %s on pid %s", self.name, os.getpid())


class InferenceProcess(multiprocessing.Process):
    """ Moves inference into a subprocess to let Video Player classes avoid binding """
    model_path: pathlib.Path
    request_queue: multiprocessing.Queue
    result_queue: multiprocessing.Queue
    debug: bool
    model_platform: cvops.schemas.ModelPlatforms
    confidence_threshold: float
    iou_threshold: float
    metadata: typing.Dict[str, typing.Any]

    def __init__(self,
                 model_path: pathlib.Path,
                 request_queue: "multiprocessing.queues.Queue[numpy.ndarray]",
                 result_queue: "multiprocessing.queues.Queue[cvops.schemas.InferenceResult]",
                 model_platform: cvops.schemas.ModelPlatforms,
                 metadata: typing.Dict[str, typing.Any],
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.4,
                 debug: bool = False
                 ) -> None:
        super().__init__()
        if debug:
            multiprocessing.log_to_stderr(logging.DEBUG)
        assert isinstance(model_path, pathlib.Path), "`model_path` must be a Path"
        assert model_path.exists(), "Supplied `model_path` does not exist"
        assert model_path.suffix == ".onnx", "Model must be onnx format"
        self.model_path = model_path
        assert isinstance(
            request_queue, multiprocessing.queues.Queue), "`request_queue` must be a multiprocessing.Queue"
        self.request_queue = request_queue
        assert isinstance(result_queue, multiprocessing.queues.Queue), "`result_queue` must be a multiprocessing.Queue"
        self.result_queue = result_queue
        self.name = "python-cvops-inference"
        self.is_listening = True
        self.debug = debug
        assert isinstance(
            model_platform, cvops.schemas.ModelPlatforms), "`model_platform` must be a ModelPlatforms enum"
        self.model_platform = model_platform
        assert isinstance(confidence_threshold, float), "`confidence_threshold` must be a float"
        self.confidence_threshold = confidence_threshold
        assert isinstance(iou_threshold, float), "`iou_threshold` must be a float"
        self.iou_threshold = iou_threshold
        assert isinstance(metadata, dict), "`metadata` must be a dict"
        self.metadata = metadata

    def run(self) -> None:
        """ Starts the inference process """
        try:
            if self.debug:
                logger.debug("Starting Inference Process %s on pid %s", self.name, os.getpid())
            session_request = cvops.inference.factories.create_inference_session_request(
                self.model_platform,
                self.model_path,
                metadata=self.metadata,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold
            )
            with cvops.inference.manager.InferenceSessionManager(session_request) as mgr:
                if self.debug:
                    self.result_queue.put(True)
                while self.is_listening:
                    try:
                        image = self.request_queue.get()
                        logger.debug("Received image from queue")
                        assert isinstance(image, numpy.ndarray), "`image` must be a numpy.ndarray"
                        c_inference_result = mgr.run_inference(
                            image,
                            "",
                            draw_detections=False
                        )
                        logger.debug("Ran inference")
                        if not c_inference_result:
                            raise RuntimeError("Inference returned NULL Pointer")
                        # Note: c-inference result is a pointer to a struct (that itself contains pointers), not the struct itself
                        # it cannot be serialized an sent over the wire, the pointer will be orphaned
                        serializable_result = cvops.inference.factories.inference_result_from_c_type(c_inference_result)
                        while not self.result_queue.empty():
                            self.result_queue.get_nowait()
                        self.result_queue.put(serializable_result)
                        mgr.dispose_inference_result(c_inference_result)
                    except Exception as ex:  # pylint: disable=broad-exception-caught
                        logger.exception(ex, "Error in Inference Process.  Failed Inference attempt.")
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.exception(ex, "Error in Inference Process. Unable to start session.")
        finally:
            logger.debug("Exiting Inference Process %s on pid %s", self.name, os.getpid())


class LocalModelVideoPlayer(cvops.inference.manager.InferenceResultRenderer, VideoPlayerBase):
    """ Video Player that uses a local model for inference
        This class is
    """
    model_path: pathlib.Path
    inference_request_queue: "queue.Queue[numpy.ndarray]"
    inference_result_queue: "queue.Queue[cvops.schemas.InferenceResult]"
    inference_process: InferenceProcess
    last_result: typing.Optional[cvops.inference.c_interfaces.InferenceResult]
    _queue_initialized: bool
    debug: bool

    def __init__(self,
                 model_path: typing.Union[str, pathlib.Path],
                 model_platform: cvops.schemas.ModelPlatforms,
                 metadata: typing.Dict[str, typing.Any],
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.4,
                 debug: bool = False,
                 **kwargs
                 ) -> None:
        # Add metadata to kwargs
        kwargs.update(**metadata)
        super().__init__(**kwargs)
        if isinstance(model_path, str):
            model_path = pathlib.Path(model_path)
        assert model_path.exists(), "Supplied model path does not exist"
        self.model_path = model_path
        self.inference_request_queue = queue.Queue()
        self.inference_result_queue = queue.Queue()
        self.inference_process = InferenceThread(
            model_path=self.model_path,
            model_platform=model_platform,
            request_queue=self.inference_request_queue,
            result_queue=self.inference_result_queue,
            metadata=metadata,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            debug=debug or cvops.config.SETTINGS.debug
        )
        self.last_result = None
        self._queue_initialized = False
        self.debug = debug

    def __enter__(self) -> "LocalModelVideoPlayer":
        try:
            super().__enter__()
            self.inference_process.start()
            if self.debug:
                is_started = self.inference_result_queue.get()
                if not is_started:
                    raise RuntimeError("Unable to start inference process")
            return self
        except Exception as ex:
            raise RuntimeError("Unable to start inference process") from ex

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.inference_process.is_listening = False
            self.inference_process.join()
            # self.inference_process.terminate()
            # while self.inference_process.is_alive():
            #     time.sleep(0.1)
            # self.inference_process.close()
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.exception(ex, "Unable to end inference process")
        finally:
            super().__exit__(exc_type, exc_value, traceback)

    def process_frame(self, frame: numpy.ndarray) -> numpy.ndarray:
        """ Processes a frame of video and returns the processed frame """
        try:
            # Send the first frame to the inference process
            if self.last_result is None and not self._queue_initialized:
                self.inference_request_queue.put_nowait(frame)
                self._queue_initialized = True

            # Check for new results
            inference_result = None
            # Get the latest result
            while not self.inference_result_queue.empty():
                # Returns a serializable result
                inference_result = self.inference_result_queue.get_nowait()
            if inference_result is not None:
                # Convert to a ctype for the render method
                logger.debug("Received inference result")
                self.last_result = cvops.inference.factories.inference_result_to_c_type_ptr(inference_result)
                # Result queue is cleared, send the current frame to the inference process
                self.inference_request_queue.put_nowait(frame)
            # Render results to frame if available
            if self.last_result:
                self.render(self.last_result, frame)
        except queue.Full:
            pass
        except queue.Empty:
            pass
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.exception(ex, "Error processing frame")
        return frame
