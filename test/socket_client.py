
import socket


HOST = '47.102.112.240'
PORT = 3022
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
