import pygame
import random

WINDOW_HEIGHT = 600
WINDOW_WIDTH = 600
BOARD_WIDTH = 15
SQUARE_LENGTH = int(WINDOW_WIDTH / BOARD_WIDTH)
BOARD_HEIGHT = int(WINDOW_HEIGHT / SQUARE_LENGTH)

def color_square(surface, color, row, col):
    pygame.draw.rect(surface, color, [col * SQUARE_LENGTH, row * SQUARE_LENGTH, SQUARE_LENGTH, SQUARE_LENGTH])

def draw_checkerboard(surface):
    surface.fill((0,190,0))
    for i in range(BOARD_WIDTH):
        for j in range(BOARD_HEIGHT):
            if (i+j) % 2 == 0:
                color_square(surface, (0,200,0), i, j)

class Snake():
    def __init__(self, row, col, length, dir, color):
        self.row = row
        self.col = col
        self.dir = dir
        self.color = color
        self.pos = []
        self.length = length
        for i in range(length):
            self.pos.insert(0, (row - dir[0] * i, col - dir[1] * i))
    
    def draw(self, surface):
        for square in self.pos:
            color_square(surface, self.color, square[0], square[1])
    
    def update_pos(self): # Returns survived boolean
        if self.row + self.dir[0] < 0 or self.col + self.dir[1]< 0:
            return False
        if self.row + self.dir[0] >= BOARD_HEIGHT or self.col + self.dir[1] >= BOARD_WIDTH:
            return False
        if (self.row + self.dir[0], self.col + self.dir[1]) in self.pos:
            return False
        
        self.row += self.dir[0]
        self.col += self.dir[1]
        if len(self.pos) == self.length:
            self.pos.pop(0)
        self.pos.append((self.row, self.col))
        return True
    
    def change_dir(self, dir): # Returns whether or not the direction was valid
        if dir != (-self.dir[0], -self.dir[1]):
            self.dir = dir
            return True
        return False
    
    def handle_collection(self, apple): # Returns whether or not an apple was collected
        if self.pos[-1] == (apple.row, apple.col):
            self.length += 1
            if len(self.pos) == BOARD_HEIGHT * BOARD_WIDTH:
                print("you win")
            while (apple.row, apple.col) in self.pos:
                apple.row = random.randint(0,BOARD_HEIGHT-1)
                apple.col = random.randint(0,BOARD_WIDTH-1)
            return True
        return False


class Apple():
    def __init__(self, row, col):
        self.row = row
        self.col = col
    
    def draw(self, surface):
        color_square(surface, (200,0,0), self.row, self.col)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))

player = Snake(BOARD_HEIGHT // 2, 4, 4, (0, 1), (0,0,200))
apple = Apple(BOARD_HEIGHT // 2, BOARD_WIDTH - 5)

running = True

clock = pygame.time.Clock()
while running:
    draw_checkerboard(screen)
    player.draw(screen)
    apple.draw(screen)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_LEFT:
                player.change_dir((0,-1))
            elif event.key == pygame.K_RIGHT:
                player.change_dir((0,1))
            elif event.key == pygame.K_UP:
                player.change_dir((-1,0))
            elif event.key == pygame.K_DOWN:
                player.change_dir((1,0))

    player.update_pos()
    player.handle_collection(apple)

    pygame.display.flip()
    clock.tick(10)

pygame.quit()