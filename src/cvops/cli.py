""" CVOps Command Line Interface"""
import argparse
import sys
import logging
import typing
import pathlib
import cvops
import cvops.schemas
import cvops.device
import cvops.events
import cvops.workflows


logger = logging.getLogger(__name__)


class CLI(object):
    """
    CVOps Command Line Interface
    --------------------------

    Automate and manage your CVOps devices.

    Command Line Instructions
    -------------------------
    """

    parser: argparse.ArgumentParser
    subparsers: argparse.ArgumentParser
    command_map: typing.Dict[str, typing.Callable[[argparse.Namespace], None]]
    listen_parser: argparse.ArgumentParser
    version_parser: argparse.ArgumentParser
    workspace_parser: argparse.ArgumentParser
    deploy_parser: argparse.ArgumentParser
    deploy_subparsers: argparse.ArgumentParser

    EPILOG = """
    Further documentation is available at <https://cvops.io/docs/>."
    """

    class Commands:
        """Commands for the CLI"""
        LISTEN = "listen"
        VERSION = "version"
        WORKSPACE = "workspace"
        DEPLOY = "deploy"
        RUN_INFERENCE = 'run-inference'

    class Descriptions:
        """Descriptions for the CLI commands"""
        LISTEN = """
            Listen for device events and AI inference results.
            This command will block until the process is interrupted.
        """
        VERSION = "View the current version of the CVOps SDK"
        WORKSPACE = "View info about the CVOps workspace that this deice is connected to."
        DEPLOY = "Deploy a model to devices in your workspace"
        RUN_INFERENCE = "Run inference on a local model file for testing purposes"

    class DeployDescriptions:
        """Descriptions for the deploy command"""
        MODEL_FRAMEWORK = "Framework of the model to deploy.  Can be one of: [onnx, tensorflow, pytorch].  Defaults to \"onnx\""
        TYPE = "Type of model to deploy.  Can be one of: [image-classification, object-detection, image-segmentation, chatbot].  Defaults to \"image-segmentation\""
        FILEPATH = "Local file path of the model file to deploy"
        DEVICE_IDS = "Device ids to deploy to.  Defaults to all devices in workspace."
        DEPLOY = """
            Deploy a model to devices in your CVOps workspace
        """
    
    class DeploySubCommands:
        LOCAL = "local"
        YOLOV8 = "yolov8"

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=self.__doc__,
            epilog=self.EPILOG,
            formatter_class=argparse.RawTextHelpFormatter,
            prog='cvops'
        )
        self.subparsers = self.parser.add_subparsers(
            title="Commands",
            description="The following commands are available:",
            dest="command",
            metavar="command",
            prog='cvops'
        )
        self.listen_parser = self.subparsers.add_parser('listen', help=self.Descriptions.LISTEN)
        self.version_parser = self.subparsers.add_parser('version', help=self.Descriptions.VERSION)
        self.workspace_parser = self.subparsers.add_parser('workspace', help=self.Descriptions.WORKSPACE)
        self.deploy_parser = self.subparsers.add_parser('deploy', 
            help=self.Descriptions.DEPLOY,
            description=self.DeployDescriptions.DEPLOY
        )
        self.run_inference_parser = self.subparsers.add_parser('run-inference', help=self.Descriptions.RUN_INFERENCE)
        self.run_inference_parser.add_argument(
            "-m", "--model-path", 
            help="Path to the model file to test",
        )
        self.run_inference_parser.add_argument(
            "-i", "--image-path",
            help="Path to the image file to run inference on",
        )
        self.run_inference_parser.add_argument(
            "-p", "--model-platform",
            help="Model platform of the model to test.  Can be one of: [YOLO, Detectron].  Defaults to \"YOLO\"",
        )


        self.deploy_subparsers = self.deploy_parser.add_subparsers(
            title="Deployment Commands",
            description="The following deployment commands are available:",
            dest="deployment_command",
            metavar="deployment_command",
            prog='cvops'
        )

        self.local_deploy_parser = self.deploy_subparsers.add_parser(
            self.DeploySubCommands.LOCAL,
            help="Deploy a local model file to devices in your workspace",
        )
        
        self.local_deploy_parser.add_argument(
            "filepath", 
            help=self.DeployDescriptions.FILEPATH,
        )

        self.local_deploy_parser.add_argument(
            "-m", "--model-framework", 
            help=self.DeployDescriptions.MODEL_FRAMEWORK,
            default="onnx",
            required=False
        )
        
        self.local_deploy_parser.add_argument(
            "-t", "--type", 
            help=self.DeployDescriptions.TYPE,
            required=True
        )

        self.local_deploy_parser.add_argument(
            "-d", "--device-ids", 
            help=self.DeployDescriptions.DEVICE_IDS,
            default=None,
            required=False
        )

        self.yolo_deploy_parser = self.deploy_subparsers.add_parser(
            "yolov8",
            help="Deploy a YOLOv8 model to devices in your workspace",
            description="Deploys a local YoloV8 model to devices in your workspace.  If no path is provided, downloads a pretrained model from ultralytics"
        )

        self.yolo_deploy_parser.add_argument(
            "-d", "--device-ids", 
            help=self.DeployDescriptions.DEVICE_IDS,
            default=None,
            required=False
        )

        self.yolo_deploy_parser.add_argument(
            "-f", "--filepath", 
            help=self.DeployDescriptions.FILEPATH + "  If not provided, downloads a pretrained model from ultralytics",
            default=None,
            required=False
        )

        self.command_map = {
            self.Commands.LISTEN: self.listen_handler,
            self.Commands.VERSION: self.version_handler,
            self.Commands.WORKSPACE: self.workspace_handler,
            self.Commands.DEPLOY: self.deploy_handler,
            self.Commands.RUN_INFERENCE: self.run_inference_handler
        }

        self.deployment_command_map = {
            self.DeploySubCommands.LOCAL: self.local_deploy_handler,
            self.DeploySubCommands.YOLOV8: self.yolov8_deploy_handler
        }

    @staticmethod
    def version_handler(args: argparse.Namespace) -> None:  # pylint: disable=unused-argument
        """Handles the version command"""
        print(f"CVOps SDK Version: {cvops.__version__}")

    @staticmethod
    def listen_handler(args: argparse.Namespace) -> None:  # pylint: disable=unused-argument
        """Handles the listen command"""
        device_manager = cvops.device.DeviceManager()
        device_manager.listen()

    @staticmethod
    def print_json_payload(event: 'cvops.events.AnyEvent') -> None:
        """Prints the JSON payload of an event"""
        print(f"{event.event_type}")
        print(event.model_dump_json(indent=2))

    @staticmethod
    def workspace_handler(args: argparse.Namespace) -> None:  # pylint: disable=unused-argument
        """Handles the workspace command"""

        device_manager = cvops.device.DeviceManager()

        def print_workspace(event: cvops.events.WorkspaceDetailsResponseEvent) -> None:
            print("Current Workspace:")
            print(event.deserialize_event_data().model_dump_json(indent=2))

        device_manager.set_event_callback(
            cvops.events.EventTypes.WORKSPACE_DETAILS_RESPONSE,
            print_workspace,
            disconnect_after_callback=True)
        device_manager.get_workspace()

    @staticmethod
    def run_inference_handler(args: argparse.Namespace) -> None:  # pylint: disable=unused-argument
        """Handles the run inference command"""
        platform = cvops.schemas.ModelPlatforms(args.model_platform)
        model_path = pathlib.Path(args.model_path)
        image_path = pathlib.Path(args.image_path)
        cvops.workflows.test_onnx_inference(model_path, image_path, platform)

    def deploy_handler(self, args: argparse.Namespace) -> None:  # pylint: disable=unused-argument
        """Handles the deploy command"""
        cmd = self.deployment_command_map.get(args.deployment_command)
        if cmd:
            cmd(args)
        else:
            self.deploy_parser.print_help()
            sys.exit(1)

    def local_deploy_handler(self, args: argparse.Namespace) -> None:  # pylint: disable=unused-argument 
        """Handles the local deploy subcommand"""
        logger.info("Local deploy")
        cvops.workflows.deploy_onnx_model(
            args.filepath,
            args.device_ids,
            args.type,
            model_framework=args.model_framework
        )

    def yolov8_deploy_handler(self, args: argparse.Namespace) -> None:  # pylint: disable=unused-argument
        """ handles the yolov8 deploy subcommand"""
        logger.info("YOLOv8 deploy")
        cvops.workflows.deploy_YOLOv8(
            args.device_ids,
            args.filepath
        )

    def handle_command(self) -> None:
        """Handles the command"""
        args = self.parser.parse_args()
        command_handler = self.command_map.get(args.command)
        if command_handler is None:
            self.parser.print_help()
            sys.exit(1)
        command_handler(args)


def main():
    """Main entry point for the application script"""
    try:
        CLI().handle_command()
    except KeyboardInterrupt:
        logger.info("\r\nUser cancellation request. Exiting...")
        sys.exit(0)
    except Exception as ex:  # pylint: disable=broad-except
        logger.exception("Unhandled exception.", exc_info=ex)
        sys.exit(1)
