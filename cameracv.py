import cv2
import numpy as np
import tensorflow as tf
import socket
from PIL import Image, ImageDraw, ImageFont

# Load the model
model = tf.keras.models.load_model('model (1).h5')

# TCP/IP connection settings
HOST = '192.168.255.85'  # Raspberry Pi's IP address
PORT = 12345  # Port number to be used

FONT_PATH = 'fonts\FontsFree-Net-NotoSans-Black.ttf'
FONT_SIZE = 32
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# Connect to Raspberry Pi
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

# Start the camera
cap = cv2.VideoCapture(1)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    frame_array = Image.fromarray(frame, 'RGB')
    frame_array = frame_array.resize((224, 224))
    frame_array = np.array(frame_array)
    frame_array = np.expand_dims(frame_array, axis=0)

    # Resize the frame and convert it to the appropriate format for the model
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make a prediction using the model
    prediction = model.predict(frame_array)

    if prediction >= 0.7:
        label = "Rotten Fruit"
        command = "ROT"  # Command for rotten fruit
        color = (0, 0, 255)
    else:
        label = "Fresh Fruit"
        command = "FRESH"  # Command for fresh fruit
        color = (0, 255, 0)

    # Send the command to Raspberry Pi
    client_socket.sendall(command.encode())

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle(((8, 50), (140, 100)), fill=color)
    draw.text((10, 50), label, font=font, fill=(255, 255, 255))
    frame = np.array(img_pil)

    cv2.imshow('Frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
client_socket.close()
