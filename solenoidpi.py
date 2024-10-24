import socket
import RPi.GPIO as GPIO

# GPIO ayarları
LED_PIN = 18  # LED'in bağlı olduğu GPIO pin numarası
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# TCP/IP bağlantısı için ayarlar
HOST = '192.168.8.85'  # Raspberry Pi'nin IP adresi
PORT = 12345  # Kullanılacak port numarası

# Sunucu soketi oluştur ve başlat
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print('Raspberry Pi Solenoid kontrol sunucusu başlatıldı...')

# Bağlantıyı kabul et
conn, addr = server_socket.accept()
print(f'Bağlantı kabul edildi: {addr}')

try:
    while True:
        # Bilgisayardan gelen komutu al
        data = conn.recv(1024)
        if not data:
            break
        command = data.decode()

        # Gelen komuta göre LED'i kontrol et
        if command == "ROT":
            GPIO.output(LED_PIN, GPIO.HIGH)  # LED'i yak
        elif command == "FRESH":
            GPIO.output(LED_PIN, GPIO.LOW)  # LED'i söndür

finally:
    # Temizleme işlemleri
    conn.close()
    server_socket.close()
    GPIO.cleanup()
