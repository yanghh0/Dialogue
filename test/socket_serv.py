
import sys
import time
import socket

HOST = '172.17.0.1'
PORT = 21567
BUFSIZ = 1024
agent_svr_addr = (HOST, PORT)


def recv_proc(connection):
    while True:
        try:
            recv_data = connection.recv(1024)
            recv_data = recv_data.decode()
            send_data = "hello"
            send_data = send_data.encode()
            connection.sendall(send_data)
        except socket.error as e:
            time.sleep(0.1)

def run_agent_server():
    extsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    extsocket.bind(agent_svr_addr)
    extsocket.listen(5)

    while True:
        print('waiting for connection...')
        connection, ext_addr = extsocket.accept()
        print('...connnecting from:', ext_addr)
        recv_proc(connection)


if __name__ == "__main__":
    run_agent_server()