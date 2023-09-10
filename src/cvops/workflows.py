""" This module contains functions that simplify common workflows for CVOps deployments"""
import logging
import typing
import os
import pathlib
import cvops.util
import cvops.schemas
import cvops.deployments


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
            path=filepath,
            device_ids=device_ids,
            **kwargs
        )
    except AssertionError as err:
        logger.error(err.args[0])
    except Exception as ex:  # pylint: disable=broad-except
        logger.exception(ex, "Unhandled exception in deployment workflow.")


YOLO_SEGMENT_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"


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
    cvops.util.export_to_onnx(target_file_path, cvops.schemas.ModelFrameworks.TORCH, kwargs)
    deploy_onnx_model(target_file_path, device_ids, **kwargs)
    os.remove(target_file_path)


def deploy_onnx_model(
    path: typing.Union[str, pathlib.Path],
    device_ids: typing.Optional[typing.List[str]] = None,
    model_type: typing.Union[str, cvops.schemas.ModelTypes] = cvops.schemas.ModelTypes.IMAGE_SEGMENTATION,
    **kwargs
) -> None:
    """ Depolys an onnx model to devices in workspace from a local filepath or url"""
    file_path = path
    if not isinstance(path, pathlib.Path) and str(path).startswith("http"):
        file_path = cvops.util.download_file(path)
    deploy_model_to_devices(file_path, model_type, cvops.schemas.ModelFrameworks.ONNX, device_ids, **kwargs)
