import cv2
import numpy as np
import tensorflow as tf
import socket
from PIL import Image, ImageDraw, ImageFont
# Modeli yükle
model = tf.keras.models.load_model('model (1).h5')

# TCP/IP bağlantısı için ayarlar
HOST = '192.168.255.85'  # Raspberry Pi'nin IP adresi
PORT = 12345  # Kullanılacak port numarası

FONT_PATH = 'FontsFree-Net-NotoSans-Black.ttf'
FONT_SIZE = 32
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# Raspberry Pi'ye bağlan
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

# Kamerayı başlat
cap = cv2.VideoCapture(1)

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()

    frame_array = Image.fromarray(frame, 'RGB')
    frame_array = frame_array.resize((224, 224))
    frame_array = np.array(frame_array)
    frame_array = np.expand_dims(frame_array, axis=0)

    # Kareyi boyutlandır ve modele uygun formata dönüştür
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Modeli kullanarak tahmin yap
    prediction = model.predict(frame_array)

    
    if prediction >= 0.7:
        label = "Rotten Fruit"
        command = "ROT"  # Çürük meyve komutu
        color = (0, 0, 255)
    else:
        label = "Fresh Fruit"
        command = "FRESH"  # Taze meyve komutu
        color = (0, 255, 0)

    # Komutu Raspberry Pi'ye gönder
    client_socket.sendall(command.encode())

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle(((8, 50), (140, 100)), fill=color)
    draw.text((10, 50), label, font=font, fill=(255, 255, 255))
    frame = np.array(img_pil)

    cv2.imshow('Frame', frame)
     
    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera bağlantısını kapat ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
client_socket.close()
