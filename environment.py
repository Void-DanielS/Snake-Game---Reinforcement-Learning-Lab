import numpy as np
from collections import deque

GRID_SIZE = 10
CELL_SIZE = 30

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

DIRECTIONS = {
    UP:    (0, -1),
    DOWN:  (0,  1),
    LEFT:  (-1, 0),
    RIGHT: (1,  0),
}


class SnakeEnv:
    def __init__(self, grid_size=GRID_SIZE, num_obstacles=0, obstacle_positions=None):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        if obstacle_positions:
            self.obstacles = set(obstacle_positions)
        else:
            self.obstacles = set()
            self._generate_obstacles()
        self.reset()

    def _generate_obstacles(self):
        mid = self.grid_size // 2
        forbidden = {(mid, mid), (mid, mid - 1), (mid, mid - 2)}
        self.obstacles = set()
        attempts = 0
        while len(self.obstacles) < self.num_obstacles:
            pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            )
            if pos not in forbidden:
                self.obstacles.add(pos)
            attempts += 1
            if attempts > self.num_obstacles * 200:
                break

    def reset(self):
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid), (mid, mid - 1), (mid, mid - 2)])
        self.direction = RIGHT
        self.score = 0
        self.steps_without_food = 0
        self._place_food()
        return self._get_state()

    def _place_food(self):
        occupied = set(self.snake) | self.obstacles
        while True:
            pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            )
            if pos not in occupied:
                self.food = pos
                break

    def _get_state(self):
        head = self.snake[0]
        dx, dy = DIRECTIONS[self.direction]

        # Relative directions: front, left, right based on current heading
        left_dir  = (self.direction - 1) % 4
        right_dir = (self.direction + 1) % 4

        front_cell = (head[0] + dx,                 head[1] + dy)
        left_cell  = (head[0] + DIRECTIONS[left_dir][0],  head[1] + DIRECTIONS[left_dir][1])
        right_cell = (head[0] + DIRECTIONS[right_dir][0], head[1] + DIRECTIONS[right_dir][1])

        danger_front = int(self._is_collision(front_cell))
        danger_left  = int(self._is_collision(left_cell))
        danger_right = int(self._is_collision(right_cell))

        food_up    = int(self.food[1] < head[1])
        food_down  = int(self.food[1] > head[1])
        food_left  = int(self.food[0] < head[0])
        food_right = int(self.food[0] > head[0])

        return (
            danger_front, danger_left, danger_right,
            self.direction,
            food_up, food_down, food_left, food_right,
        )

    def _is_collision(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if pos in list(self.snake)[1:]:
            return True
        if pos in self.obstacles:
            return True
        return False

    def step(self, action):
        # Prevent reversing direction
        opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if action != opposite[self.direction]:
            self.direction = action

        dx, dy = DIRECTIONS[self.direction]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        # Check death
        if self._is_collision(new_head):
            return self._get_state(), -10, True

        self.snake.appendleft(new_head)
        self.steps_without_food += 1

        # Check food
        if new_head == self.food:
            self.score += 1
            self.steps_without_food = 0
            self._place_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1

        # Kill loop if starving (prevent infinite loops)
        done = self.steps_without_food > self.grid_size * self.grid_size * 2

        return self._get_state(), reward, done
