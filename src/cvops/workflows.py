""" This module contains functions that simplify common workflows for CVOps deployments"""
import logging
import typing
import os
import pathlib
import json
import tempfile
import cv2
import numpy
import cvops.util
import cvops.schemas
import cvops.deployments
import cvops.image_processor
import cvops.inference.factories
import cvops.inference.manager


logger = logging.getLogger(__name__)


def deploy_model_to_devices(
    filepath: typing.Union[str, pathlib.Path],
    model_type: typing.Union[str, cvops.schemas.ModelTypes] = cvops.schemas.ModelTypes.IMAGE_SEGMENTATION,
    model_framework: typing.Union[str, cvops.schemas.ModelFrameworks] = cvops.schemas.ModelFrameworks.ONNX,
    device_ids: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    **kwargs
) -> None:
    """ Deploys model to devices in workspace"""
    try:
        manager = cvops.deployments.DeploymentManager(**kwargs)
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)
        if not device_ids:
            logger.info("No device ids provided.  Attempting to deploy to all devices in workspace")
            manager.device_manager.get_workspace()
            assert manager.device_manager.workspace
            assert manager.device_manager.workspace.devices
            device_ids = [d.id for d in manager.device_manager.workspace.devices]
        assert isinstance(device_ids, list)
        if not isinstance(model_type, cvops.schemas.ModelTypes):
            model_type = cvops.schemas.ModelTypes(model_type)
        assert model_type in cvops.schemas.ModelTypes
        manager.deploy(
            model_type=model_type,
            model_framework=model_framework,
            ml_model_source=cvops.schemas.ModelSourceTypes.LOCAL_FILE,
            path=filepath,
            device_ids=device_ids,
            **kwargs
        )
    except AssertionError as err:
        logger.error(err.args[0])
    except Exception as ex:  # pylint: disable=broad-except
        logger.exception(ex, "Unhandled exception in deployment workflow.")


YOLO_SEGMENT_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
YOLO_OBJECT_DETECTION_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"


def deploy_YOLOv8(
    device_ids: typing.Optional[typing.List[str]] = None,
    path: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    **kwargs
):
    """ Deploys a YoloV8 model to devices in workspace.  If no path is provided, downloads a pretrained model from ultralytics"""
    if path:
        target_file_path = pathlib.Path(path)
    else:
        target_file_path = pathlib.Path(os.getcwd(), "yolov8n-seg.pt")
        cvops.util.download_file(YOLO_SEGMENT_URL, target_file_path)
    cvops.util.export_to_onnx(target_file_path, cvops.schemas.ModelPlatforms.YOLO, kwargs)
    model_file = target_file_path.with_suffix(".onnx")
    deploy_onnx_model(model_file, device_ids, **kwargs)
    os.remove(target_file_path)
    os.remove(model_file)


def deploy_onnx_model(
    path: typing.Union[str, pathlib.Path],
    device_ids: typing.Optional[typing.List[str]] = None,
    model_type: typing.Union[str, cvops.schemas.ModelTypes] = cvops.schemas.ModelTypes.IMAGE_SEGMENTATION,
    **kwargs
) -> None:
    """ Deploys an onnx model to devices in workspace from a local filepath or url"""
    file_path = path
    if not isinstance(path, pathlib.Path) and str(path).startswith("http"):
        file_path = cvops.util.download_file(path)
    deploy_model_to_devices(file_path, model_type, cvops.schemas.ModelFrameworks.ONNX, device_ids, **kwargs)


def test_onnx_inference(
    model_path: typing.Union[str, pathlib.Path],
    image_path: typing.Union[str, pathlib.Path],
    model_platform: typing.Union[str, cvops.schemas.ModelPlatforms] = cvops.schemas.ModelPlatforms.YOLO,
    metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> None:
    """ Tests an onnx model by running inference on a local image"""
    if isinstance(model_path, str):
        model_path = pathlib.Path(model_path)
    if isinstance(model_platform, str):
        model_platform = cvops.schemas.ModelPlatforms(model_platform)
    if isinstance(image_path, str):
        image_path = pathlib.Path(image_path)

    args = {
        "model_platform": model_platform,
        "model_path": model_path,
        "metadata": metadata,
    }

    with cvops.image_processor.AcceleratedImageProcessor(**args) as image_processor:
        inference_result = image_processor.run(image_path)
        output_path = pathlib.Path(os.getcwd(), "inference_result.png")
        image_processor.visualize_inference(inference_result, output_path)


def run_inference_on_directory(
    input_directory: typing.Union[str, pathlib.Path],
    model_path: typing.Union[str, pathlib.Path],
    model_platform: typing.Union[str, cvops.schemas.ModelPlatforms] = cvops.schemas.ModelPlatforms.YOLO,
    output_directory: typing.Optional[typing.Union[str, pathlib.Path, tempfile.TemporaryDirectory]] = None,
    metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
    metadata_path: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    color_palette: typing.Optional[typing.List[typing.Tuple[int, int, int]]] = None,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.4
) -> typing.List[cvops.schemas.InferenceResult]:
    """ Runs inference on all images in a directory and saves the results to an output directory

    Args:
        input_directory (typing.Union[str, pathlib.Path]): Directory containing images to run inference on
        model_path (typing.Union[str, pathlib.Path]): Path to the model file
        model_platform (typing.Union[str, cvops.schemas.ModelPlatforms], optional): Platform for the model. Defaults to cvops.schemas.ModelPlatforms.YOLO.
        output_directory (typing.Optional[typing.Union[str, pathlib.Path]], optional): Directory to save results to. Defaults to None.
        metadata (typing.Optional[typing.Dict[str, typing.Any]], optional): Metadata for the model. Defaults to None.
        metadata_path (typing.Optional[typing.Union[str, pathlib.Path]], optional): Path to metadata JSON file. Defaults to None.
        confidence_threshold (float, optional): Confidence threshold for inference. Defaults to 0.5.
        iou_threshold (float, optional): IOU threshold for inference result processing.  Used for NMS supression on overlapping bounding boxes. Defaults to 0.4.

    Returns:
        typing.List[cvops.schemas.InferenceResult]: List of inference results

    """
    if isinstance(model_path, str):
        model_path = pathlib.Path(model_path)
    if isinstance(model_platform, str):
        model_platform = cvops.schemas.ModelPlatforms(model_platform)
    if isinstance(input_directory, str):
        input_directory = pathlib.Path(input_directory)
    if input_directory.is_file():
        raise ValueError("Input directory must be a directory")
    if not input_directory.exists():
        raise ValueError("Input directory does not exist")
    if not model_path.exists():
        raise ValueError("Model path does not exist")
    if not output_directory:
        output_directory = pathlib.Path(os.getcwd(), "out")
    if isinstance(output_directory, str):
        output_directory = pathlib.Path(output_directory)

    # Create output directory if it doesn't exist and it is not a temporary directory
    if isinstance(output_directory, pathlib.Path):
        if not output_directory.exists():
            output_directory.mkdir()
    if not metadata:
        if metadata_path:
            if isinstance(metadata_path, str):
                metadata_path = pathlib.Path(metadata_path)
            if metadata_path.exists():
                with open(metadata_path, "r", encoding='utf-8') as metadata_file:
                    metadata = json.load(metadata_file)
            else:
                raise ValueError("Metadata path does not exist")
        else:
            metadata = {}

    assert isinstance(metadata, dict), "Metadata must be a dictionary"
    assert isinstance(confidence_threshold, float), "Confidence threshold must be a float"
    assert isinstance(iou_threshold, float), "IOU threshold must be a float"

    model_classes = metadata.get("classes", None)
    if not model_classes:
        raise ValueError("Metadata must contain classes")

    assert isinstance(model_classes, dict), "Classes must be a dictionary"
    assert all(isinstance(value, str) for value in model_classes.values()), "Classes values (names) must be strings"

    session_request = cvops.inference.factories.create_inference_session_request(
        model_platform,
        model_path,
        metadata=metadata,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold
    )

    num_classes = len(model_classes.keys())
    if not color_palette:
        color_palette = cvops.image_processor.generate_color_palette(num_classes)
    render_args = {
        "color_palette": color_palette,
        "classes": model_classes
    }
    results = []
    with cvops.inference.manager.InferenceSessionManager(session_request) as inference_manager:
        with cvops.inference.manager.InferenceResultRenderer(**render_args) as renderer:
            for file in input_directory.iterdir():
                try:
                    if file.is_file():
                        if not file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                            logger.info("File has invalid image type: %s", file.name)
                            continue
                        output_path = output_directory.joinpath(file.name) \
                            if isinstance(output_directory, pathlib.Path) \
                            else pathlib.Path(output_directory.name).joinpath(file.name)
                        image = cvops.image_processor.extract_image(file)
                            
                        inference_result_ptr = inference_manager.run_inference(image)
                        inference_results_dto = cvops.inference.factories.inference_result_from_c_type(
                            inference_result_ptr)
                        results.append(inference_results_dto)
                        renderer.render(inference_result_ptr, image)
                        cv2.imwrite(str(output_path), image)
                except Exception as ex:  # pylint: disable=broad-except
                    logger.exception(ex, "Unable to process file: %s", file.name)
    logger.debug("Average Inference time: %s milliseconds", numpy.average([r.milliseconds for r in results]))
    return results
