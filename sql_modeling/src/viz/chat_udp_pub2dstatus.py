import socket
import json
import time
import random

def generate_data():
    data = [{"X": x, "Y": y, "fraction": round(random.uniform(0, 1), 3)} for x in range(10) for y in range(10)]
    return data

def send_data(udp_socket, udp_host, udp_port):
    while True:
        data = generate_data()
        json_data = json.dumps(data, separators=(',', ':'))

        # Send JSON data over UDP
        udp_socket.sendto(json_data.encode(), (udp_host, udp_port))
        time.sleep(0.1) # speed up later

if __name__ == "__main__":
    udp_host = "127.0.0.1"  # localhost
    udp_port = 5555

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        send_data(udp_socket, udp_host, udp_port)

