import pygame
import random
import numpy as np

# ================== CONFIG ==================
WINDOW_SIZE = 600
BOARD_SIZE = 15
SQUARE = WINDOW_SIZE // BOARD_SIZE

MAX_STEPS = 500

LR = 0.001
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05

RENDER_EVERY = 100  # render every N episodes

# ================== GAME ==================
class Snake:
    def __init__(self, row, col, length, direction):
        self.row = row
        self.col = col
        self.dir = direction
        self.length = length
        self.pos = []

        for i in range(length):
            self.pos.insert(0, (row - direction[0]*i, col - direction[1]*i))

    def update(self):
        nr = self.row + self.dir[0]
        nc = self.col + self.dir[1]

        # collision
        if nr < 0 or nc < 0 or nr >= BOARD_SIZE or nc >= BOARD_SIZE:
            return False
        if (nr, nc) in self.pos:
            return False

        self.row, self.col = nr, nc

        if len(self.pos) == self.length:
            self.pos.pop(0)

        self.pos.append((nr, nc))
        return True


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = Snake(BOARD_SIZE//2, 4, 4, (0,1))
        self.spawn_apple()
        return self.get_state()

    def spawn_apple(self):
        free = [(r,c) for r in range(BOARD_SIZE)
                        for c in range(BOARD_SIZE)
                        if (r,c) not in self.snake.pos]
        self.apple = random.choice(free)

    def apply_action(self, action):
        dirs = [(0,1),(1,0),(0,-1),(-1,0)]
        idx = dirs.index(self.snake.dir)

        if action == 1:
            self.snake.dir = dirs[(idx - 1) % 4]
        elif action == 2:
            self.snake.dir = dirs[(idx + 1) % 4]

    def step(self, action):

        old_dist = abs(self.snake.row - self.apple[0]) + abs(self.snake.col - self.apple[1])

        self.apply_action(action)

        new_dist = abs(self.snake.row - self.apple[0]) + abs(self.snake.col - self.apple[1])

        alive = self.snake.update()

        if not alive:
            return self.get_state(), -10, True

        reward = -0.1

        if (self.snake.row, self.snake.col) == self.apple:
            self.snake.length += 1
            reward = 10
            self.spawn_apple()
        
        if new_dist < old_dist:
            reward += 0.2
        else:
            reward -= 0.2

        return self.get_state(), reward, False

    def get_state(self):
        head_r, head_c = self.snake.row, self.snake.col
        dr, dc = self.snake.dir

        def danger(r, c):
            return (
                r < 0 or c < 0 or
                r >= BOARD_SIZE or c >= BOARD_SIZE or
                (r,c) in self.snake.pos
            )

        left = (-dc, dr)
        right = (dc, -dr)

        state = [
            danger(head_r+dr, head_c+dc),
            danger(head_r+left[0], head_c+left[1]),
            danger(head_r+right[0], head_c+right[1]),

            dr == -1, dr == 1, dc == -1, dc == 1,

            self.apple[0] < head_r,
            self.apple[0] > head_r,
            self.apple[1] < head_c,
            self.apple[1] > head_c,
        ]

        return np.array(state, dtype=float)

    def render(self, screen, episode):
        screen.fill((0,180,0))

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r+c)%2 == 0:
                    pygame.draw.rect(screen,(0,200,0),
                        (c*SQUARE,r*SQUARE,SQUARE,SQUARE))

        for r,c in self.snake.pos:
            pygame.draw.rect(screen,(0,0,200),
                (c*SQUARE,r*SQUARE,SQUARE,SQUARE))

        pygame.draw.rect(screen,(200,0,0),
            (self.apple[1]*SQUARE,self.apple[0]*SQUARE,SQUARE,SQUARE))
        
        font = pygame.font.SysFont("Arial", 24)
        text = font.render(f"Episode: {episode}", True, (255,255,255))
        screen.blit(text, (10, 10))

        pygame.display.flip()


# ================== NN ==================
class DQN:
    def __init__(self):
        self.W1 = np.random.randn(11, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 3) * 0.1
        self.b2 = np.zeros(3)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def train(self, x, target):
        out = self.forward(x)

        error = out - target

        dW2 = np.outer(self.a1, error)
        db2 = error

        da1 = error @ self.W2.T
        dz1 = da1 * (self.z1 > 0)

        dW1 = np.outer(x, dz1)
        db1 = dz1

        self.W2 -= LR * dW2
        self.b2 -= LR * db2
        self.W1 -= LR * dW1
        self.b1 -= LR * db1

    def save(self, filename="snake_model.npz"):
        np.savez(filename,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2
        )

    def load(self, filename="snake_model.npz"):
        data = np.load(filename)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]


# ================== TRAIN ==================
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

game = SnakeGame()
model = DQN()

if input("Load existing model? (y/n)").lower() == "y":
    try:
        model.load()
        print("Model loaded!")
        if input("Train or test?").lower() == "test":
            epsilon, EPSILON_MIN = 0, 0
    except:
        print("No saved model found, starting fresh.")

epsilon = EPSILON

episode = 0
tickspeed = 15

while True:
    state = game.reset()
    total_reward = 0

    for step in range(MAX_STEPS):

        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                if input("Overwrite existing model? (y/n)").lower() == "y":
                    model.save()
                exit()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT]:
            tickspeed = 40
        elif keys[pygame.K_LEFT]:
            tickspeed = 10
        else:
            tickspeed = 20

        # choose action
        if random.random() < epsilon:
            action = random.randint(0,2)
        else:
            q = model.forward(state)
            action = np.argmax(q)

        next_state, reward, done = game.step(action)

        # Q-learning target
        target = model.forward(state)
        if done:
            target[action] = reward
        else:
            target[action] = reward + GAMMA * np.max(model.forward(next_state))

        model.train(state, target)

        state = next_state
        total_reward += reward

        # render occasionally
        if episode % RENDER_EVERY == 0:
            game.render(screen, episode)
            clock.tick(tickspeed)

        if done:
            break

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if episode % RENDER_EVERY == 0:
        print(f"Episode {episode}, Score: {len(game.snake.pos)}, Reward: {total_reward:.2f}")
    
    episode += 1