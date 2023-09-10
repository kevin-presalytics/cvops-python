""" Model classes for CVOps Hub Entities"""
import abc
import datetime
import typing
import enum
import pydantic
import pydantic.alias_generators
import cvops.config


def now():
    """ Returns the current time in UTC"""
    return datetime.datetime.now(datetime.timezone.utc)


class LowerCaseEnum(enum.Enum):
    """ Base Class for Enums with greedy parsing of strings into enums """

    @classmethod
    def _missing_(cls, value):
        value = str(value).lower()
        for member in cls:
            if str(member.value).lower() == value:
                return member


class EditorTypes(LowerCaseEnum):
    """ Enum Indicating the type of editor for an entity """
    SYSTEM = "system"
    USER = "user"
    DEVICE = "device"


class BaseEntity(pydantic.BaseModel, abc.ABC):
    """ Base class for all CVOps Hub Entities"""
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel)
    id: str
    date_created: datetime.datetime = pydantic.Field(default_factory=now)
    user_created: typing.Optional[str] = None
    date_modified: datetime.datetime = pydantic.Field(default_factory=now)
    user_modified: typing.Optional[str] = None
    created_by: typing.Optional[EditorTypes] = None
    modified_by: typing.Optional[EditorTypes] = None


class WorkspaceRole(LowerCaseEnum):
    """ Workspace Editor Permissions for a user """
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"


class WorkspaceUser(pydantic.BaseModel):
    """ CVOPS Workspace User Data Transfer Object"""
    name: str = ""
    email: str = ""
    role: WorkspaceRole = WorkspaceRole.VIEWER


class Device(BaseEntity):
    """Dto for the the current device """
    description: str
    device_info: typing.Dict[str, typing.Any]
    workspace_id: str
    activation_status: str


class ModelSourceTypes(LowerCaseEnum):
    """ Enum Indicating the type of editor for an entity """
    LOCAL_S3_BUCKET = "LocalS3Bucket"  # Indicates that the model is local to this device, and tr
    REMOTE_S3_BUCKET = "RemoteS3Bucket"  # Indicates that the model is remote to this device, and will be downloaded from a remote S3 bucket
    # Indicates that the model is remote to this device, and will be downloaded from weights and biases
    WEIGHTS_AND_BIASES = "WeightsAndBiases"
    MLFLOW = "MLFlow"  # Indicates that the model is remote to this device, and will be downloaded from mlflow


class ModelTypes(LowerCaseEnum):
    """ Enum for the type of model being deployed.  Ties to data model of inference output. """
    IMAGE_SEGMENTATION = "ImageSegmentation"
    IMAGE_CLASSIFICATION = "ImageClassification"
    OBJECT_DETECTION = "ObjectDetection"
    CHATBOT = "Chatbot"


class DeploymentStatusTypes(LowerCaseEnum):
    """ Enum indicating the status of a deployment """
    NONE = "None"
    CREATED = "Created"
    MODEL_UPLOADING = "ModelUploading"
    MODEL_UPLOADED = "ModelUploaded"
    MODEL_VALIDATED = "ModelValidated"
    MODEL_DEPLOYING = "ModelDeploying"
    MODEL_DEPLOYED = "ModelDeployed"
    ACTIVE = "Active"
    FAILED = "Failed"
    DELETED = "Deleted"


class DeviceDeploymentStatusTypes(LowerCaseEnum):
    """ Enum indicating the status of a deployment """
    NONE = "None"
    WAITING_FOR_MODEL = "WaitingForModel"
    DOWNLOADING = "Downloading"
    DOWNLOADED = "Downloaded"
    ACTIVE = "Active"
    OBSOLETE = "Obsolete"
    FAILED = "Failed"


class DeviceDeploymentStatus(pydantic.BaseModel):
    """ Status of a deployment on a device"""
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel)
    deployment_id: str
    device_id: str
    status: DeviceDeploymentStatusTypes = DeviceDeploymentStatusTypes.NONE
    message: typing.Optional[str] = None


class Deployment(BaseEntity):
    """ Record in CVOps Hub for a deployment deployment of an AI Model to a set"""
    deployment_initiator_id: str = cvops.config.SETTINGS.device_id
    deployment_initiator_type: EditorTypes = EditorTypes.DEVICE
    ml_model_source: ModelSourceTypes = pydantic.Field(
        ModelSourceTypes.LOCAL_S3_BUCKET,
        alias="model_source",
        serialization_alias="modelSource")
    device_ids: typing.List[str]
    workspace_id: str
    devices_status: typing.List[DeviceDeploymentStatus] = []
    bucket_name: str
    object_name: str
    ml_model_metadata: typing.Dict[str, typing.Any] = pydantic.Field(
        {}, alias="model_metadata", serialization_alias="modelMetadata")
    status: DeploymentStatusTypes = DeploymentStatusTypes.NONE


class Workspace(BaseEntity):
    """ CVOPS Workspace Data Transfer Object"""
    name: str = ""
    description: str = ""
    users: typing.List[WorkspaceUser] = []
    devices: typing.List[Device] = []
    deployments: typing.List[Deployment] = []


class DeploymentCreatedPayload(pydantic.BaseModel):
    """ DTO for creating a deployment """
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel)
    workspace_id: str
    device_ids: typing.List[str]
    ml_model_source: ModelSourceTypes = pydantic.Field(
        ModelSourceTypes.LOCAL_S3_BUCKET,
        alias="model_source",
        serialization_alias="modelSource")
    ml_model_type: ModelTypes = pydantic.Field(
        ModelTypes.IMAGE_SEGMENTATION,
        alias="model_type",
        serialization_alias="modelType")
    deployment_initiator_id: str
    deployment_initiator_type: EditorTypes = EditorTypes.DEVICE
    bucket_name: typing.Optional[str] = None
    object_name: typing.Optional[str] = None
    ml_model_metadata: typing.Dict[str, typing.Any] = pydantic.Field(
        {},
        alias="model_metadata",
        serialization_alias="modelMetadata"
    )


class ModelFrameworks(LowerCaseEnum):
    """ Enum for model framework that"""
    TENSORFLOW = "tensorflow"
    TORCH = "torch"
    ONNX = "onnx"

    @classmethod
    def parse_from_filename(cls, filename: str) -> 'ModelFrameworks':
        """ Factory method to parse enum from a filename """
        if filename.endswith(".pt") or filename.endswith(".pth"):
            return cls.TORCH
        if filename.endswith(".onnx"):
            return cls.ONNX
        return cls.TENSORFLOW

    @classmethod
    def _missing_(cls, value):
        if str(value).lower() in ["torch", "pytorch", "py", "pt", "pth", "torchvision", "pytorchlite", "pytorchmobile"]:
            return cls.TORCH
        if str(value).lower() in ["tf", "tensorflow", "tflite"]:
            return cls.TENSORFLOW
        return super()._missing_(value)


class StorageMessageTypes(LowerCaseEnum):
    """ Message type on the workspace storage channel"""
    PUT_URL_REQUEST = "PutUrlRequest"
    PUT_URL_RESPONSE = "PutUrlResponse"
    GET_URL_REQUEST = "GetUrlRequest"
    GET_URL_RESPONSE = "GetUrlResponse"


class StorageMessagePayload(pydantic.BaseModel):
    """ Dto for messages from the device storage topic"""
    type: StorageMessageTypes
    url: typing.Optional[str] = None
