import socket
import json
import pygame
import sys

def process_data(json_data, screen):
    data = json.loads(json_data)

    screen.fill((0, 0, 0))  # Clear the screen

    for point in data:
        x = int(point["X"] * 50)
        y = int(point["Y"] * 50)
        fraction = point["fraction"]
        color_value = int(fraction * 255)

        pygame.draw.circle(screen, (color_value, color_value, color_value), (x, y), 25)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    udp_host = "127.0.0.1"  # localhost
    udp_port = 5555

    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    clock = pygame.time.Clock()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.bind((udp_host, udp_port))
        while True:
            data, _ = udp_socket.recvfrom(4096)
            json_data = data.decode()
            process_data(json_data, screen)

            clock.tick(30)  # Control the frame rate

