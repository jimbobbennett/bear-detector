import json
import os

from dotenv import load_dotenv

from azure.iot.device import IoTHubDeviceClient, ProvisioningDeviceClient, Message

# Load the environment variables
load_dotenv()

# Get the connection details from the .env file for Azure IoT Central
ID_SCOPE = os.environ['ID_SCOPE']
DEVICE_ID = os.environ['DEVICE_ID']
PRIMARY_KEY = os.environ['PRIMARY_KEY']

def connect_device() -> IoTHubDeviceClient:
    """Connects this device to IoT Central
    """
    # Connect to the device provisioning service and request the connection details
    # for the device
    provisioning_device_client = ProvisioningDeviceClient.create_from_symmetric_key(
        provisioning_host='global.azure-devices-provisioning.net',
        registration_id=DEVICE_ID,
        id_scope=ID_SCOPE,
        symmetric_key=PRIMARY_KEY)
    registration_result = provisioning_device_client.register()

    # Build the connection string - this is used to connect to IoT Central
    conn_str = 'HostName=' + registration_result.registration_state.assigned_hub + \
                ';DeviceId=' + DEVICE_ID + \
                ';SharedAccessKey=' + PRIMARY_KEY

    # The device client object is used to interact with Azure IoT Central.
    device_client = IoTHubDeviceClient.create_from_connection_string(conn_str)

    # Connect to the IoT Central hub
    device_client.connect()

    # Return the device client
    return device_client

def send_detection_telemetry(device_client: IoTHubDeviceClient, detected: bool) -> None:
    """Sends a telemetry message to IoT Central
    """
    # Create the telemetry message
    message = Message(json.dumps({ 'bear_detected': detected }))

    # Send the message to IoT Central
    device_client.send_message(message)