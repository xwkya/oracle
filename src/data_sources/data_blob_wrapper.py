import configparser
import os

from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient


class DataBlobWrapper:
    """
    A wrapper class for simplifying file interactions in Azure Blob Storage.
    This class defaults to a container named 'data', but can be overridden.

    :param container_name: The container to interact with. Defaults to 'data'.
    """

    def __init__(
            self,
            container_name: str = "data"
    ):
        config = configparser.ConfigParser()
        config.read("appsettings.ini")

        credential = DefaultAzureCredential()

        account_url = config["StorageAccountKey"]["StorageUrl"]
        self.blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential
        )
        self.container_name = container_name
        self.container_client: ContainerClient = self.blob_service_client.get_container_client(container_name)
        self.config = config

        try:
            self.container_client.create_container()
        except ResourceExistsError:
            pass
        except Exception as e:
            raise e

    def upload_file(self, local_file_path: str, blob_name: str = None, overwrite: bool = True) -> None:
        """
        Uploads a single file to this container.

        :param local_file_path: Path to the local file that will be uploaded.
        :type local_file_path: str
        :param blob_name: The name of the blob in Azure Storage. If None, the local file name is used.
        :type blob_name: str, optional
        :param overwrite: Determines whether to overwrite the blob if it already exists.
        :type overwrite: bool
        """
        if blob_name is None:
            blob_name = os.path.basename(local_file_path)

        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)

    def download_file(self, blob_name: str, download_file_path: str = None) -> None:
        """
        Downloads a single file from this container to a local path.

        :param blob_name: The name of the blob to download.
        :type blob_name: str
        :param download_file_path: The local destination path. If None, uses the blob's name as the file name.
        :type download_file_path: str, optional
        """
        if download_file_path is None:
            download_file_path = os.path.basename(blob_name)

        blob_client = self.container_client.get_blob_client(blob_name)
        with open(download_file_path, "wb") as file_data:
            stream = blob_client.download_blob()
            file_data.write(stream.readall())

    def upload_folder(self, local_folder_path: str, overwrite: bool = True, blob_prefix: str = "") -> None:
        """
        Recursively uploads an entire folder to this container.
        Subfolders are represented by the blob name's path-like structure.

        :param local_folder_path: Path to the local folder that will be uploaded recursively.
        :type local_folder_path: str
        :param overwrite: Determines whether to overwrite blobs if they already exist.
        :type overwrite: bool
        :param blob_prefix: A prefix to prepend to all blob names, usually a path-like structure.
        :type blob_prefix: str
        """
        for root, _, files in os.walk(local_folder_path):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(full_path, local_folder_path)
                blob_name = relative_path.replace("\\", "/") # Support for Windows paths
                blob_name = os.path.join(blob_prefix, blob_name)

                blob_client = self.container_client.get_blob_client(blob_name)
                with open(full_path, "rb") as file_data:
                    blob_client.upload_blob(file_data, overwrite=overwrite)

    def download_folder(self, local_folder_path: str, blob_prefix: str = "") -> None:
        """
        Recursively downloads all blobs (optionally filtered by a prefix) from this container.
        Subfolders are recreated on the local filesystem based on blob paths.

        :param local_folder_path: Local destination path where blobs will be downloaded.
        :type local_folder_path: str
        :param blob_prefix: Filters which blobs are downloaded by matching this prefix in their name.
        :type blob_prefix: str, optional
        """
        blobs = self.container_client.list_blobs(name_starts_with=blob_prefix)
        for blob in blobs:
            local_path = os.path.join(local_folder_path, blob.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            blob_client = self.container_client.get_blob_client(blob.name)
            with open(local_path, "wb") as file_data:
                stream = blob_client.download_blob()
                file_data.write(stream.readall())

class DatasetUploader:
    """
    A utility class for uploading datasets to Azure Blob Storage.
    """
    def __init__(self):
        self.blob_wrapper = DataBlobWrapper()
        config = configparser.ConfigParser()
        config.read("appsettings.ini")
        self.config = config

    def upload_cepii_dataset(self) -> None:
        """
        Uploads the Cepii dataset to Azure Blob Storage.
        Currently contains the BACI data and the Gravity Data.
        """

        self.blob_wrapper.upload_folder(self.config["datasets"]["CepiFolderPath"], overwrite=False, blob_prefix="cepi/")

    def download_cepii_dataset(self) -> None:
        """
        Downloads the Cepii dataset from Azure Blob Storage.
        Currently contains the BACI data and the Gravity Data.
        """

        self.blob_wrapper.download_folder(self.config["datasets"]["DatasetRootPath"], blob_prefix="cepi/")