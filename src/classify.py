#!/usr/bin/env python

import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

import cv2
import os
import sys
import signal
import time

from edge_impulse_linux.image import ImageImpulseRunner

from PIL import Image, ImageDraw

from azure_device_client import connect_device, send_detection_telemetry

# Create a runner now so we can stop it when the user exits the program
runner = None

def sigint_handler(_, __):
    '''
    A signal handler to catch ctrl+c and stop the model runner
    '''
    if (runner):
        runner.stop()
    sys.exit(0)

# Set up the signal handler - this just makes exiting with ctrl+c a bit nicer
signal.signal(signal.SIGINT, sigint_handler)

def ellipse(image_path, x, y):
    '''
    Draws a red elipse on the given area of the image to
    indicate where the bear was detected
    '''
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.ellipse((x-8, y-8, x+8, y+8), outline="red")
    image.save(image_path)

def main():
    '''
    The main function - runs the model and sends telemetry to IoT Central
    '''

    # Connect to IoT Central
    device_client = connect_device()

    # Get the path to the Edge Impulse model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, 'modelfile.eim')
    print('MODEL: ' + modelfile)

    # Create a model runner
    with ImageImpulseRunner(modelfile) as runner:
        try:
            # Initialize the runner
            runner.init()

            # Track the last time a bear was detected. This allows us to send
            # telemetry only when a bear is detected for a certain amount of time
            last_bear_detected_time = 0.
            bear_detected = False

            # Get the next frame from the camera
            # Under the hood this used OpenCV to access the camera.
            # 0 is the first camera, if you have more attached then set this appropriately
            for img in runner.get_frames(0):
                # Get the center of the image as a square - the model only runs on 
                # square images, so crop the center to a square. This also converts
                # the image to grey scale
                # Then extract the image features from the image using the Edge Impulse model.
                features, cropped = runner.get_features_from_image(img, 'center')

                # Run the classifier against the model
                res = runner.classify(features)

                # Save a copy of the image to help with debugging
                cv2.imwrite('debug.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

                # If a bear is detected, the bounding boxes are set to the location that a bear is found in the image
                # For now, assume only one bear is ever detected, not more than one
                if "bounding_boxes" in res["result"].keys() and len(res["result"]["bounding_boxes"]) > 0:
                    # If we have a bounding box, then a bear is detected!

                    # Get the bounding box and draw it on the debuf image
                    box = res["result"]["bounding_boxes"][0]
                    ellipse('debug.jpg', box['x'], box['y'])

                    # If a bear was not previously detected, log to the console and send telemetry
                    # to IoT Central
                    if not bear_detected:
                        print('Bear detected!')
                        send_detection_telemetry(device_client, True)

                    # Set the last detected time to now
                    last_bear_detected_time = time.time()

                    # Set the bear detected flag to true
                    bear_detected = True
                else:
                    # If we dont have a bounding box, then no bear was detected.
                    # If the last reading was a bear was detected more than 5 seconds ago, then flag as
                    # not detected and send telemetry to IoT Central
                    if bear_detected and time.time() > last_bear_detected_time + 5:
                        # Print that no bear was detected
                        print('No bear detected!')

                        # Set the bear detected flag to false
                        bear_detected = False

                        # Send telemetry to IoT Central
                        send_detection_telemetry(device_client, False)

        finally:
            # Once we exit, stop the model runner
            if (runner):
                runner.stop()

# Start the app running
main()