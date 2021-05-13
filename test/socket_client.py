
import socket


HOST = '103.46.128.53'
PORT = 24876
BUFSIZ = 1024
ADDR = (HOST, PORT)

connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connection.connect(ADDR)

while True:
    ask = input('>')
    if ask == 'q':
        break
    connection.send(ask.encode())
    response = connection.recv(BUFSIZ)
    print(response.decode('utf-8'))
connection.close()
