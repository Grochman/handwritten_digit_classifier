import pygame
import numpy as np
from onlyFormScratchMLP import Model
from onlyFormScratchMLP import decode
from sklearn.neural_network import MLPClassifier
import pickle

WHITE = (255, 255, 255)
screen_dimensions = (500, 500)
mlp_resolution = (28,28)
screen = pygame.display.set_mode(screen_dimensions)
isPressed = False
running = True
clf = MLPClassifier(solver='sgd', alpha=0, hidden_layer_sizes=(16, 16), batch_size=64,
                        learning_rate_init=0.01, max_iter=5, shuffle=False, verbose=True, momentum=0)

with open("sklearn_model.pkl", "rb") as file:
    clf = pickle.load(file)

model = Model()
model.load()


def getScreenData():
    area = pygame.Rect(0, 0, mlp_resolution[0], mlp_resolution[1])
    low_res = pygame.transform.scale(screen, mlp_resolution)
    sub_surface = low_res.subsurface(area)
    pixel_data = pygame.surfarray.array2d(sub_surface)
    pixel_data = pixel_data.T
    pixel_data[pixel_data>0] = 1
    pixel_data = pixel_data.flatten()
    return pixel_data


def detect(data):
    print("sklearn: ",clf.predict([data]))
    data = data[:, np.newaxis]
    item = model.forward(data)
    print("my implementation: ", decode(item))


def drawSquare(screen, x, y):
    square_size = 40
    pygame.draw.rect(screen, WHITE, (x-square_size/2, y-square_size/2, square_size, square_size))


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
            (x, y) = pygame.mouse.get_pos()  # returns the position of mouse cursor
            drawSquare(screen, x, y)
        elif event.type == pygame.QUIT:
            running = False
    pygame.display.flip()

