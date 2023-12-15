""" Classes for Video Streaming and Processing """
import abc
import typing
import pathlib
import os
import time
import queue
import multiprocessing
import multiprocessing.queues
import logging
import cv2
import numpy
import cvops
import cvops.config
import cvops.schemas
import cvops.image_processor
import cvops.inference.manager
import cvops.inference.factories


logger = logging.getLogger(__name__)


class VideoPlayerBase(cvops.schemas.CooperativeBaseClass):
    """ Base class for video players """
    video_source: str
    show_video: bool
    cap: cv2.VideoCapture
    fps: float
    limit_fps: bool

    def __init__(self,
                 video_source: typing.Union[str, pathlib.Path] = "0",
                 show_video: bool = True,
                 limit_fps: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(video_source, pathlib.Path):
            video_source = str(video_source)
        self.video_source = video_source
        # This probably needs to be replaced with a `validate_source` function that checks for remote streams, etc.
        self.cap = cv2.VideoCapture(str(self.video_source))
        self.show_video = show_video
        self.limit_fps = limit_fps

        # Get openCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')  # type: ignore[attr-defined] # pylint: disable=no-member, unused-variable

        # With webcam get(CV_CAP_PROP_FPS) does not work.
        # Let's see for ourselves.

        if int(major_ver) < 3:
            self.fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)  # type: ignore[attr-defined]
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
            if image is None:
                self.stop()
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
        """ Play the video as a stream"""
        for image in self.image_stream:
            self.process_frame(image)

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
        except BaseException:  # pylint: disable=broad-exception-caught
            pass


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
                 request_queue: "multiprocessing.queues.Queue[bytes]",
                 result_queue: "multiprocessing.queues.Queue[cvops.schemas.InferenceResult]",
                 model_platform: cvops.schemas.ModelPlatforms,
                 metadata: typing.Dict[str, typing.Any],
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.4,
                 process_name: typing.Optional[str] = None,
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
        if process_name is not None:
            assert isinstance(process_name, str), "`process_name` must be a str"
        self.name = process_name or "python-cvops-inference"
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
                    self.result_queue.put(self.name)
                while self.is_listening:
                    try:
                        image_bytes = self.request_queue.get()
                        # logger.debug("Received image from queue")
                        image = cvops.image_processor.extract_image(image_bytes)
                        assert isinstance(image, numpy.ndarray), "`image` must be a numpy.ndarray"
                        # logger.debug("Running inference on image.  Image size: %s", len(image_bytes))
                        c_inference_result = mgr.run_inference(image)
                        # logger.debug("Inference complete")
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


class LocalModelVideoPlayer(cvops.tracking.VideoObjectTrackerMixin, VideoPlayerBase):
    """ Video Player that uses a local model for inference
        This class is
    """
    model_path: pathlib.Path
    inference_request_queue: "multiprocessing.queues.Queue[bytes]"
    inference_result_queue: "multiprocessing.queues.Queue[cvops.schemas.InferenceResult]"
    inference_processes: typing.List[InferenceProcess]
    last_result: typing.Optional[cvops.inference.c_interfaces.InferenceResult]
    _queue_initialized: bool
    debug: bool
    last_request_time: int
    num_inference_processes: int

    def __init__(self,
                 model_path: typing.Union[str, pathlib.Path],
                 model_platform: cvops.schemas.ModelPlatforms,
                 metadata: typing.Dict[str, typing.Any],
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.4,
                 num_inference_processes: int = 1,
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
        self.inference_request_queue = multiprocessing.Queue()
        self.inference_result_queue = multiprocessing.Queue()
        assert isinstance(num_inference_processes, int), "`num_inference_processes` must be an int"
        self.num_inference_processes = num_inference_processes
        self.inference_processes = []
        for i in range(self.num_inference_processes):
            self.inference_processes.append(
                InferenceProcess(
                    model_path=self.model_path,
                    model_platform=model_platform,
                    request_queue=self.inference_request_queue,
                    result_queue=self.inference_result_queue,
                    metadata=metadata,
                    process_name=f"python-cvops-inference-{i}",
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                    debug=debug or cvops.config.SETTINGS.debug
                )
            )
        self.last_result = None
        self._queue_initialized = False
        self.debug = debug
        self.last_request_time = int(time.time() * 1000)

    def __enter__(self) -> "LocalModelVideoPlayer":
        try:
            super().__enter__()
            started = [False for _ in range(self.num_inference_processes)]
            _ = [p.start() for p in self.inference_processes]  # type: ignore
            if self.debug:
                logger.debug("Waiting for inference processes to start")
                # TODO: This code to wait for inferences processes doesn't work
                while not all(started):
                    process_name = self.inference_result_queue.get()
                    for i, process in enumerate(self.inference_processes):
                        if process.name == process_name:
                            started[i] = True
                            logger.debug("Inference Process %s started", process.name)
                            break
            return self
        except Exception as ex:
            raise RuntimeError("Unable to start inference process") from ex

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            # # For threading
            # self.inference_process.is_listening = False
            # self.inference_process.join()

            # Same, but for multiprocessing
            _ = [p.terminate() for p in self.inference_processes]  # type: ignore[func-returns-value]
            # _ = [p.join() for p in self.inference_processes]
            # _ = [p.close() for p in self.inference_processes]
            logger.debug("Inference processes terminated")
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.exception(ex, "Unable to end inference process")
        finally:
            super().__exit__(exc_type, exc_value, traceback)

    def process_frame(self, frame: numpy.ndarray) -> numpy.ndarray:
        """ Processes a frame of video and returns the processed frame """
        try:
            new_result = False
            # Send the first frame to the inference process
            if self.last_result is None and not self._queue_initialized:
                image_bytes = cvops.image_processor.image_to_bytes(frame)
                for _ in range(self.num_inference_processes):
                    self.inference_request_queue.put_nowait(image_bytes)
                    self._queue_initialized = True

            # Check for new results
            inference_result = None
            # Get the latest result
            while not self.inference_result_queue.empty():
                # Returns a serializable result
                inference_result = self.inference_result_queue.get_nowait()
            if inference_result is not None:
                # Convert to a ctype for the render method
                # logger.debug("Received inference result")
                self.last_result = cvops.inference.factories.inference_result_to_c_type_ptr(inference_result)
                new_result = True
            if new_result:
                # Send the current frame to the inference process "Just in Time"
                # Use the last inference time to estimate when to send the next frame
                milliseconds_since_last_request = int(time.time() * 1000) - self.last_request_time
                if milliseconds_since_last_request > (
                        self.last_result.contents.milliseconds /
                        self.num_inference_processes):
                    # Clear the queue prior to inserting the new frame
                    while not self.inference_request_queue.empty():
                        # logger.debug("Removed request queue item")  # This should only happen is
                        # something funny happens on the inference thread.
                        self.inference_request_queue.get_nowait()
                    if frame is not None:
                        image_bytes = cvops.image_processor.image_to_bytes(frame)
                        self.inference_request_queue.put_nowait(image_bytes)
                        self.last_request_time = int(time.time() * 1000)
                
                # update image and tracker with new results
                self.update_tracker(image, self.last_result)
            else:
                # Advance the tracker one frame and render
                self.track_image(image)
            self.new_result = False
                
        except queue.Full:
            pass
        except queue.Empty:
            pass
        except Exception as ex:  # pylint: disable=broad-exception-caught
            logger.exception(ex, "Error processing frame")
        return frame
