"""Deployment manager for AI Model deployments to CVOps devices."""
import typing
import pathlib
import json
import datetime
import logging
import pydantic
import cvops.config
import cvops.schemas
import cvops.device
import cvops.events
import cvops.mqtt
import cvops.util


logger = logging.getLogger(__name__)


class DeploymentSession(pydantic.BaseModel):
    """ A session for a deployment """
    deployment: typing.Optional[cvops.schemas.Deployment] = None
    start_time: datetime.datetime = pydantic.Field(default_factory=cvops.schemas.now)
    end_time: typing.Optional[datetime.datetime] = None
    is_active: bool = True


class DeploymentManager(cvops.schemas.CooperativeBaseClass):
    """ Manager for deployments of AI Models to CVOps devices. """
    session: typing.Optional[DeploymentSession]
    device_manager: 'cvops.device.DeviceManager'
    wait_for_completion: bool
    update_actions: typing.Dict[cvops.schemas.DeploymentStatusTypes, typing.Callable[[], None]]
    deployment_timeout: float
    close_signal: typing.Optional[cvops.mqtt.CallbackSignal]

    def __init__(self,
                 device_manager: typing.Optional['cvops.device.DeviceManager'] = None,
                 wait_for_completion: bool = True,
                 deployment_timeout: typing.Optional[float] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.deployment_timeout = deployment_timeout or cvops.config.SETTINGS.deployment_timeout
        self.session = None
        self.device_manager = device_manager or cvops.device.DeviceManager()
        self.wait_for_completion = wait_for_completion
        self.close_signal = None

        self.device_manager.set_event_callback(
            cvops.events.EventTypes.DEPLOYMENT_CREATED,
            self.handle_deployment_created
        )

        self.device_manager.set_event_callback(
            cvops.events.EventTypes.DEPLOYMENT_UPDATED,
            self.handle_deployment_updated
        )

        self.device_manager.set_event_callback(
            cvops.events.EventTypes.DEPLOYMENT_DELETED,
            self.handle_deployment_deleted_event
        )

        self.update_actions = {
            cvops.schemas.DeploymentStatusTypes.MODEL_UPLOADED: self.handle_upload_complete,
            cvops.schemas.DeploymentStatusTypes.MODEL_DEPLOYED: self.handle_model_deployed,
            cvops.schemas.DeploymentStatusTypes.FAILED: self.handle_deployment_failed,
            cvops.schemas.DeploymentStatusTypes.DELETED: self.handle_deployment_deleted,
            cvops.schemas.DeploymentStatusTypes.ACTIVE: self.handle_deployment_active
        }

    @property
    def workspace_deployments_topic(self) -> str:
        """ Returns the topic for the workspace deployment """
        assert self.device_manager.workspace
        return f"workspace/{self.device_manager.workspace.id}/deployments"

    def _start(self):
        if not self.session:
            self.session = DeploymentSession()
        if not self.device_manager.workspace:
            self.device_manager.get_workspace()

    # def handle_workspace_deployment_event(self, *args) -> None:
    #     """ Handles that workspace deployment event"""
    #     try:

    # event: cvops.events.AnyEvent = cvops.events.BaseEvent.factory(*args,
    # manager=self.device_manager)  # type: ignore

    #         if isinstance(event, cvops.events.DeploymentCreatedEvent):
    #             self.handle_deployment_created(event)
    #         elif isinstance(event, cvops.events.DeploymentUpdatedEvent):
    #             self.handle_deployment_updated(event)
    #         elif isinstance(event, cvops.events.DeploymentDeletedEvent):
    #             self.handle_deployment_deleted_event(event)
    #         else:
    #             raise ValueError(f"Unknown event type: {event.event_type}")
    #     except (ValueError, AssertionError) as err:
    #         logger.error(err.args[0])
    #     except Exception as ex:  # pylint: disable=broad-except
    #         logger.exception(ex, "Error handling workspace deployment event.")

    def deploy(self,
               model_type: cvops.schemas.ModelTypes,
               path: typing.Optional[pathlib.Path],
               ml_model_source: cvops.schemas.ModelSourceTypes,
               device_ids: typing.Optional[typing.List[str]] = None,
               bucket_name: typing.Optional[str] = None,
               object_name: typing.Optional[str] = None,
               model_metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
               wait_for_completion: typing.Optional[bool] = None,
               **kwargs
               ) -> None:
        """Initiates a deployment."""
        #  Connect to MQTT
        try:
            with self.device_manager.open(timeout=self.deployment_timeout) as close_signal:
                self.close_signal = close_signal
                self._start()

                # Check that device startup is completed successfully
                assert self.device_manager
                assert self.device_manager.device
                if wait_for_completion:
                    self.wait_for_completion = wait_for_completion
                # def workspace_event_handler(*args):
                #     self.device_manager.handle_platform_event(*args)
                self.device_manager.subscribe(
                    self.device_manager.workspace_events_topic,
                    self.device_manager.handle_platform_event  # type: ignore
                )
                # Validate and Normalize arguments
                if not model_metadata:
                    model_metadata = {}
                if not self.device_manager.workspace:
                    raise ValueError("No workspace set.")
                if not device_ids:
                    device_ids = [d.id for d in self.device_manager.workspace.devices]
                if len(device_ids) == 0:
                    raise ValueError("No devices to deploy to.")
                if isinstance(ml_model_source, str):
                    ml_model_source = cvops.schemas.ModelSourceTypes(ml_model_source)
                if self.device_manager.device_id is None:
                    raise ValueError("No device id set.")
                if path:
                    model_metadata["local_filepath"] = path
                if kwargs:
                    model_metadata.update(kwargs)
                else:
                    if not bucket_name and not object_name:
                        raise ValueError(
                            "If no local file path is provided, a bucket_name and object name must be set.")

                # Create Deployment Created Payload
                created_payload = cvops.schemas.DeploymentCreatedPayload(
                    deployment_initiator_id=self.device_manager.device.id,
                    deployment_initiator_type=cvops.schemas.EditorTypes.DEVICE,
                    workspace_id=self.device_manager.workspace.id,
                    ml_model_source=cvops.schemas.ModelSourceTypes.LOCAL_FILE,  # type: ignore
                    bucket_name=bucket_name,
                    object_name=object_name,
                    ml_model_metadata=model_metadata or {},
                    model_type=model_type,
                    device_ids=device_ids
                )

                payload = cvops.schemas.DeploymentMessage(
                    type=cvops.schemas.DeploymentMessageTypes.CREATED,
                    payload=created_payload
                )

                #  Send message via MQTT to create a deployment
                message = cvops.mqtt.MqttMessage(
                    topic=self.workspace_deployments_topic,
                    payload=payload.model_dump_json(by_alias=True),
                    qos=cvops.mqtt.QualityOfService.AT_MOST_ONCE
                )
                self.device_manager.publish(message)
                close_statuses = [
                    cvops.schemas.DeploymentStatusTypes.DELETED,
                    cvops.schemas.DeploymentStatusTypes.FAILED,
                    cvops.schemas.DeploymentStatusTypes.ACTIVE
                ]
                if not self.wait_for_completion or self.device_manager.device_id not in device_ids:
                    close_statuses.append(cvops.schemas.DeploymentStatusTypes.MODEL_DEPLOYING)

                def listen_for_close_status(event: cvops.events.DeploymentUpdatedEvent):
                    deployment = event.deserialize_event_data()
                    if deployment.status in close_statuses:
                        close_signal.complete_callback()
                self.device_manager.set_event_callback(
                    cvops.events.EventTypes.DEPLOYMENT_UPDATED,
                    listen_for_close_status
                )
                close_signal.wait_for_callback()

        except AssertionError as a_err:
            logger.error(a_err.args[0])
            self.mark_deployment_as_failed()
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex, "Unknown error deploying model.")
            self.mark_deployment_as_failed()

    def handle_model_deployed(self):
        """ Handles the model deployed status """
        logger.info("Model deploying to devices.")

    def handle_upload_complete(self):
        """ Runs with upload status complete """
        logger.info("Model upload complete.")

    def handle_deployment_failed(self):
        """ Runs when deployment fails """
        logger.info("Deployment failed.")
        self.mark_deployment_as_failed()

    def handle_deployment_active(self):
        """ Runs when deployment is active """
        logger.info("Model successfully deployed to all devices.")
        self.close_session()

    def handle_deployment_deleted(self):
        """ Runs when deployment status is changed to to deleted"""
        logger.info("Deployment deleted.")
        self.close_session()

    def _request_upload_url(self) -> None:
        try:
            assert self.session
            assert self.session.deployment
            assert self.device_manager.workspace

            def url_received_callback(*args) -> None:
                """ Command callback for an upload url received. """
                self.url_received_callback(*args)

            self.device_manager.subscribe(
                self.device_manager.device_command_topic,
                url_received_callback
            )
            self.session.deployment.status = cvops.schemas.DeploymentStatusTypes.MODEL_UPLOADING
            self.session.deployment.bucket_name = self.device_manager.workspace.id
            self.session.deployment.object_name = f"deployment/{self.session.deployment.id}"
            self.update_deployment(self.session.deployment)

            payload = cvops.schemas.StorageMessagePayload(
                type=cvops.schemas.StorageMessageTypes.PUT_URL_REQUEST,
                object_name=self.session.deployment.object_name,
            )
            message_content = cvops.schemas.StorageMessage(
                response_topic=self.device_manager.device_command_topic,
                payload=payload
            )
            message = cvops.mqtt.MqttMessage(
                topic=self.device_manager.workspace_storage_topic,
                payload=message_content.model_dump_json(by_alias=True),
                qos=cvops.mqtt.QualityOfService.EXACTLY_ONCE,
                response_topic=self.device_manager.device_command_topic
            )
            self.device_manager.publish(message)
        except AssertionError as a_err:
            logger.error(a_err.args[0])
            self.mark_deployment_as_failed()
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex, "Unknown Error requesting upload url for model.")
            self.mark_deployment_as_failed()

    def url_received_callback(self, *args) -> None:
        """ Command callback for an upload url received. """
        try:
            assert self.session
            assert self.session.deployment
            msg = args[2]
            data = json.loads(msg.payload)
            url_response = cvops.schemas.StorageMessage(**data)
            url = url_response.payload.url
            if url:
                path = self.session.deployment.ml_model_metadata.get("local_filepath", None)
                if path:
                    if isinstance(path, str):
                        path = pathlib.Path(path)
                    logger.info("Uploading model file `%s` to CVOps hub.", path.name)
                    cvops.util.upload_file(url, path)
                    logger.info("Model upload complete.")
                    self.session.deployment.status = cvops.schemas.DeploymentStatusTypes.MODEL_UPLOADED
                    self.update_deployment(self.session.deployment)
                else:
                    raise ValueError("Deployment metadata has no `local_filepath` attribute.")
            else:
                raise ValueError("Url received message contains no url")
        except (ValueError, AssertionError) as err:
            logger.error(err.args[0])
            self.mark_deployment_as_failed()
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex, "Exception uploading model to CVOps hub.")
            self.mark_deployment_as_failed()

    def mark_deployment_as_failed(self) -> None:
        """ Update deployment as failed and close session """
        if self.session:
            self.session.end_time = cvops.schemas.now()  # pylint: disable=attribute-defined-outside-init
            self.close_session(failed=True)

    def close_session(self, failed: bool = False) -> None:
        """ Closes the local deployment session """
        if self.session:
            if self.session.deployment:
                active = self.session.is_active
                self.session.is_active = False  # pylint: disable=attribute-defined-outside-init
                if failed:
                    self.session.deployment.status = cvops.schemas.DeploymentStatusTypes.FAILED
                if self.device_manager.workspace and active:
                    self.update_deployment(self.session.deployment)
        if self.close_signal:
            self.close_signal.complete_callback()

    def handle_deployment_created(self, event: cvops.events.DeploymentCreatedEvent) -> None:
        """ Handles the deployment created event """
        try:
            assert self.session
            assert self.device_manager
            self.session.deployment = event.deserialize_event_data()
            path = self.session.deployment.ml_model_metadata.get("local_filepath")
            if path:
                self._request_upload_url()
            else:
                self.session.deployment.status = cvops.schemas.DeploymentStatusTypes.MODEL_UPLOADING
                self.update_deployment(self.session.deployment)
        except AssertionError as err:
            logger.error(err.args[0])
            self.mark_deployment_as_failed()
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex, "Exception handling deployment created event.")
            self.mark_deployment_as_failed()

    def update_deployment(self,
                          deployment: typing.Optional[cvops.schemas.Deployment] = None,
                          **kwargs) -> None:
        """ Updates a deployment """
        try:
            assert self.session
            assert self.session.deployment
            assert self.device_manager.workspace
            if deployment and isinstance(deployment, cvops.schemas.Deployment):
                self.session.deployment = deployment  # pylint: disable=attribute-defined-outside-init
            else:
                for key, val in kwargs.items():
                    if hasattr(self.session.deployment, key):
                        setattr(self.session.deployment, key, val)
                    else:
                        raise ValueError(f"Deployment has no attribute: \"{key}\"")
            message_content = cvops.schemas.DeploymentMessage(
                type=cvops.schemas.DeploymentMessageTypes.UPDATED,
                payload=self.session.deployment
            )
            message = cvops.mqtt.MqttMessage(
                topic=self.workspace_deployments_topic,
                payload=message_content.model_dump_json(by_alias=True),
                qos=cvops.mqtt.QualityOfService.EXACTLY_ONCE
            )
            self.device_manager.publish(message)
        except (ValueError, AssertionError) as err:
            logger.error(err.args[0])
            self.mark_deployment_as_failed()
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex, "Exception updating deployment.")

    @staticmethod
    def upload_file(url: str, file_path: pathlib.Path) -> None:
        """ UploAds a local file path to a url """
        cvops.util.upload_file(url, file_path)

    def handle_deployment_updated(self, event: cvops.events.DeploymentUpdatedEvent) -> None:
        """ Handles the deployment updated event"""
        try:
            assert self.session
            deployment = event.deserialize_event_data()
            if self.session.deployment:
                if deployment.id == self.session.deployment.id:
                    self.session.deployment = deployment  # pylint: disable=attribute-defined-outside-init
                else:
                    raise ValueError(
                        f"Deployment id mismatch. Expected: \"{self.session.deployment.id}\". Received: \"{deployment.id}\"")
            else:
                self.session.deployment = deployment  # pylint: disable=attribute-defined-outside-init
            logger.info("Deployment status: %s", self.session.deployment.status.value)
            update_action = self.update_actions.get(self.session.deployment.status, None)
            if update_action:
                update_action()

        except AssertionError as err:
            logger.error(err.args[0])
            self.mark_deployment_as_failed()
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex, "Exception handling deployment updated event.")
            self.mark_deployment_as_failed()

    def handle_deployment_deleted_event(self, event: cvops.events.DeploymentDeletedEvent) -> None:  # pylint: disable=unused-argument
        """ Handles the deployment deleted event """
        logger.info("Deployment deleted.")
        self.close_session()
