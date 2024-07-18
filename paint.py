import pygame
import numpy as np
from models import FromScratchModel
from models import decode
from models import PytorchModel
import pickle
import torch


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen_dimensions = (500, 500)
mlp_resolution = (28, 28)

screen = pygame.display.set_mode(screen_dimensions)
draw_surface = pygame.Surface(mlp_resolution)
draw_surface.fill(BLACK)

isPressed = False
last_pos = None
running = True

pytorch_model = PytorchModel().to("cpu")
pytorch_model.load_state_dict(torch.load("pytorch_model.pth"))

with open("sklearn_model.pkl", "rb") as file:
    sklearn_model = pickle.load(file)

my_model = FromScratchModel()
my_model.load()


def getScreenData():
    area = pygame.Rect(0, 0, mlp_resolution[0], mlp_resolution[1])
    sub_surface = draw_surface.subsurface(area)
    pixel_data = pygame.surfarray.array2d(sub_surface)
    pixel_data = pixel_data.T
    pixel_data[pixel_data > 0] = 1.0
    return pixel_data


def detect(data):
    data_flat = data.flatten()
    print("sklearn: ", sklearn_model.predict([data_flat])[0])

    float_data = data.astype(np.float32)
    tensor_data = torch.tensor(float_data.reshape(1, 28, 28))
    print("pytorch: ", pytorch_model.predict(tensor_data, "cpu"))

    data_flat = data_flat[:, np.newaxis]
    item = my_model.forward(data_flat)
    print("my implementation: ", decode(item)[0])


def drawThickAntialiasedLine(surface, start_pos, end_pos, color, thickness=1):
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        return

    dx /= length
    dy /= length

    for i in range(-thickness // 2, thickness // 2 + 1):
        offset_start = (start_pos[0] + i * dy, start_pos[1] - i * dx)
        offset_end = (end_pos[0] + i * dy, end_pos[1] - i * dx)
        pygame.draw.aaline(surface, color, offset_start, offset_end)


def blit_scaled_draw_surface():
    scaled_surface = pygame.transform.scale(draw_surface, screen_dimensions)
    screen.blit(scaled_surface, (0, 0))


while running:
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                isPressed = True
            elif event.button == 3:
                draw_surface.fill((0, 0, 0))
            elif event.button == 2:
                detect(getScreenData())
        elif event.type == pygame.MOUSEBUTTONUP:
            isPressed = False
        elif event.type == pygame.MOUSEMOTION and isPressed:
            current_pos = pygame.mouse.get_pos()
            if last_pos is not None:
                start_pos = (last_pos[0] * mlp_resolution[0] // screen_dimensions[0],
                             last_pos[1] * mlp_resolution[1] // screen_dimensions[1])
                end_pos = (current_pos[0] * mlp_resolution[0] // screen_dimensions[0],
                           current_pos[1] * mlp_resolution[1] // screen_dimensions[1])
                drawThickAntialiasedLine(draw_surface, start_pos, end_pos, WHITE)
            last_pos = current_pos
        elif event.type == pygame.QUIT:
            running = False
        else:
            last_pos = None
    screen.fill(BLACK)
    blit_scaled_draw_surface()
    pygame.display.flip()
