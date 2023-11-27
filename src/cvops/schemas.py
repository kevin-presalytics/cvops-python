""" Model classes for CVOps Hub Entities"""
import abc
import datetime
import typing
import enum
import uuid
import pydantic
import pydantic.alias_generators
import cvops.config


def now():
    """ Returns the current time in UTC"""
    return datetime.datetime.now(datetime.timezone.utc)


class CooperativeBaseClass(abc.ABC):
    """ Base class to facilitate cooperative inheritance """

    def __init__(self, **kwargs):
        super().__init__()


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
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    date_created: datetime.datetime = pydantic.Field(default_factory=now)
    user_created: typing.Optional[str] = None
    date_modified: datetime.datetime = pydantic.Field(default_factory=now)
    user_modified: typing.Optional[str] = None
    created_by: typing.Optional[EditorTypes] = None
    modified_by: typing.Optional[EditorTypes] = None


class TimeSeriesEntity(BaseEntity):
    """ Base class for all CVOps Hub Entities that have a time series"""
    time: datetime.datetime = pydantic.Field(default_factory=now)


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
    LOCAL_FILE = "LocalFile"  # Indicates that the model is local to this device, and tr
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
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    deployment_id: str
    device_id: str
    status: DeviceDeploymentStatusTypes = DeviceDeploymentStatusTypes.NONE
    message: typing.Optional[str] = None


class DeviceDeploymentStatusList(pydantic.BaseModel):
    """" Wrapper class for device deployment status list"""
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    devices: typing.List[DeviceDeploymentStatus] = []


class Deployment(BaseEntity):
    """ Record in CVOps Hub for a deployment deployment of an AI Model to a set"""
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    deployment_initiator_id: str = cvops.config.SETTINGS.device_id
    deployment_initiator_type: EditorTypes = EditorTypes.DEVICE
    ml_model_source: ModelSourceTypes = pydantic.Field(ModelSourceTypes.LOCAL_FILE, alias="modelSource")
    workspace_id: str
    devices_status: DeviceDeploymentStatusList = DeviceDeploymentStatusList()
    bucket_name: str
    object_name: str
    ml_model_metadata: typing.Dict[str, typing.Any] = pydantic.Field({}, alias="modelMetadata")
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
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    workspace_id: str
    device_ids: typing.List[str]
    ml_model_source: ModelSourceTypes = pydantic.Field(ModelSourceTypes.LOCAL_FILE, alias="modelSource")
    ml_model_type: ModelTypes = pydantic.Field(ModelTypes.IMAGE_SEGMENTATION, alias="modelType")
    deployment_initiator_id: str
    deployment_initiator_type: EditorTypes = EditorTypes.DEVICE
    bucket_name: typing.Optional[str] = None
    object_name: typing.Optional[str] = None
    ml_model_metadata: typing.Dict[str, typing.Any] = pydantic.Field({}, alias="modelMetadata")


class DeploymentMessageTypes(LowerCaseEnum):
    """ Message type on the deployment channel"""
    CREATED = "Created"
    UPDATED = "Updated"
    DELETED = "Deleted"
    DEVICE_STATUS = "DeviceStatus"


class DeploymentMessage(pydantic.BaseModel):
    """ Dto for messages from the deployment topic"""
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    type: DeploymentMessageTypes
    payload: typing.Union[DeploymentCreatedPayload, Deployment]


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
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    type: StorageMessageTypes
    url: typing.Optional[str] = None
    object_name: typing.Optional[str] = None


class StorageMessage(pydantic.BaseModel):
    """ Dto for messages from the device storage topic"""
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    response_topic: typing.Optional[str] = None
    payload: StorageMessagePayload


class ModelPlatforms(LowerCaseEnum):
    """ Platform for the model. Typically an organization that manages IP for model packaging"""
    YOLO = "yolo"
    DETECTRON = "detectron"  # Not Implemented


class InferenceResultTypes(LowerCaseEnum):
    """ Enum for the type of inference result being returned """
    BOXES = "boxes"
    MESHES = "meshes"
    LABELS = "labels"


class Box(pydantic.BaseModel):
    """ DTO for bounding boxes"""
    x: int
    y: int
    height: int
    width: int
    class_id: int
    class_name: str
    object_id: int
    confidence: float

    def __iter__(self):
        return iter((self.x, self.y, self.width, self.height))


class Label(pydantic.BaseModel):
    """ DTO for detected classes """
    class_id: int
    class_name: str

    def __eq__(self, other):
        if isinstance(other, Label):
            return self.class_id == other.class_id
        return False


class InferenceResult(TimeSeriesEntity):
    """ DTO for inference results"""
    model_config = pydantic.ConfigDict(alias_generator=pydantic.alias_generators.to_camel, populate_by_name=True)
    boxes: typing.List[typing.Dict[str, typing.Any]] = []
    workspace_id: typing.Optional[str] = None
    device_id: str = cvops.config.SETTINGS.device_id
    result_type: InferenceResultTypes = InferenceResultTypes.BOXES
    boxes: typing.Optional[typing.Sequence[Box]] = []
    meshes: typing.Optional[typing.Sequence] = []
    labels: typing.Optional[typing.Sequence] = []
    milliseconds: float = 0.0


class TrackingAlgorithmTypes(LowerCaseEnum):
    """ Enum for the type of object tracking algorithm """
    BOOSTING = "boosting"
    MIL = "mil"
    KCF = "kcf"
    TLD = "tld"
    MEDIANFLOW = "medianflow"
    GOTURN = "goturn"
    MOSSE = "mosse"
    CSRT = "csrt"
