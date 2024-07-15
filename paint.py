import pygame
import numpy as np
from models import FromScratchModel
from models import decode
from models import PytorchModel
import pickle
import torch

WHITE = (255, 255, 255)
screen_dimensions = (500, 500)
mlp_resolution = (28, 28)
screen = pygame.display.set_mode(screen_dimensions)
isPressed = False
running = True

pytorch_model = PytorchModel().to("cpu")
pytorch_model.load_state_dict(torch.load("pytorch_model.pth"))

with open("sklearn_model.pkl", "rb") as file:
    sklearn_model = pickle.load(file)

my_model = FromScratchModel()
my_model.load()


def getScreenData():
    area = pygame.Rect(0, 0, mlp_resolution[0], mlp_resolution[1])
    low_res = pygame.transform.scale(screen, mlp_resolution)
    sub_surface = low_res.subsurface(area)
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


def drawSquare(screen, x, y):
    square_size = 40
    pygame.draw.rect(screen, WHITE, (x - square_size / 2, y - square_size / 2, square_size, square_size))


while running:
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                isPressed = True
            elif event.button == 3:
                screen.fill((0, 0, 0))
            elif event.button == 2:
                detect(getScreenData())
        elif event.type == pygame.MOUSEBUTTONUP:
            isPressed = False
        elif event.type == pygame.MOUSEMOTION and isPressed:
            (x, y) = pygame.mouse.get_pos()
            drawSquare(screen, x, y)
        elif event.type == pygame.QUIT:
            running = False
    pygame.display.flip()
