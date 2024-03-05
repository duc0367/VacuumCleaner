import pygame
import numpy as np

pygame.init()

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)

COLORS = ['blue', 'red', 'green']

MOVE_UP = np.array([1, 0, 0, 0, 0])
MOVE_LEFT = np.array([0, 1, 0, 0, 0])
MOVE_DOWN = np.array([0, 0, 1, 0, 0])
MOVE_RIGHT = np.array([0, 0, 0, 1, 0])
MOVE_FREEZE = np.array([0, 0, 0, 0, 1])

EMPTY_STATE = 0
CLEANED_STATE = 1
CLEANER_STATE = 2
BARRIER_STATE = 3

IS_PROCESSING = 1


def draw_lines(screen, window_size: int, line_width: int, block_size: int):
    for x in range(3, window_size, block_size + line_width):
        pygame.draw.line(screen, 'black', (x, 0), (x, window_size), line_width)

    for y in range(3, window_size, block_size + line_width):
        pygame.draw.line(screen, 'black', (0, y), (window_size, y), line_width)


def draw_cleaners(screen, cleaners, line_width: int, block_size: int):
    for idx, cleaner in enumerate(cleaners):
        pygame.draw.circle(screen, COLORS[idx], (cleaner[0] * (block_size + line_width) + line_width + block_size // 2,
                           cleaner[1] * (block_size + line_width) + line_width + block_size // 2),
                           block_size // 2 - 8, 4)


def draw_state_rect(screen, state, line_width: int, block_size: int):
    for idx, x in enumerate(state):
        for idy, y in enumerate(x):
            rect = pygame.Rect(idx * (block_size + line_width) + line_width,
                               idy * (block_size + line_width) + line_width,
                               block_size, block_size)

            if y == CLEANED_STATE or y == CLEANER_STATE:
                pygame.draw.rect(screen, 'pink', rect)

            if y == BARRIER_STATE:
                pygame.draw.rect(screen, 'purple', rect)


class VacuumCleaner:
    def __init__(self, w=512, h=512, num_cleaner=3, is_human=True):
        self.w = w
        self.h = h
        self.screen = pygame.display.set_mode((w, h))
        self.clock = pygame.time.Clock()
        self.num_block_per_dim = 8
        self.state = np.zeros((self.num_block_per_dim, self.num_block_per_dim))
        self.line_width = 8
        self.block_size = (w - self.line_width) // 8 - self.line_width
        self.num_cleaner = num_cleaner
        self.cleaners = np.zeros((num_cleaner, 2))
        self.is_human = is_human
        self.num_barrier = num_cleaner

    def place_object(self):
        init_place = np.random.randint(0, self.num_block_per_dim - 1, size=(2,))
        if self.state[init_place[0], init_place[1]] == EMPTY_STATE:
            return init_place
        return self.place_object()

    def reset(self):
        self.state = np.zeros((8, 8), dtype=int)
        self.cleaners = np.zeros((self.num_cleaner, 2), dtype=int)
        for idx in range(self.num_cleaner):
            cleaner = self.place_object()
            self.cleaners[idx] = cleaner
            self.state[cleaner[0], cleaner[1]] = CLEANER_STATE

        for idx in range(self.num_barrier):
            barrier = self.place_object()
            self.state[barrier[0], barrier[1]] = BARRIER_STATE

    def is_done(self):
        return np.count_nonzero(self.state == 0) == 0

    def move_cleaner(self, idx, cleaner_status, actions):
        cleaner_status[idx] = IS_PROCESSING
        action = actions[idx]
        cleaner = self.cleaners[idx]
        new_cleaner = cleaner.copy()

        if (action == MOVE_FREEZE).all():
            return -1

        if (action == MOVE_UP).all():
            new_cleaner[1] -= 1
        elif (action == MOVE_LEFT).all():
            new_cleaner[0] -= 1
        elif (action == MOVE_DOWN).all():
            new_cleaner[1] += 1
        elif (action == MOVE_RIGHT).all():
            new_cleaner[0] += 1

        if new_cleaner[0] < 0 or new_cleaner[0] >= self.num_block_per_dim or new_cleaner[1] < 0 or new_cleaner[1] >= self.num_block_per_dim:
            return -5

        new_state = self.state[new_cleaner[0], new_cleaner[1]]

        if new_state == BARRIER_STATE:
            return -5

        reward = -1
        if new_state == CLEANER_STATE:
            collision_with_idx = np.where((self.cleaners == new_cleaner).all(axis=1))[0][0]
            if cleaner_status[collision_with_idx] != IS_PROCESSING:
                next_reward = self.move_cleaner(collision_with_idx, cleaner_status, actions)

                if next_reward == -5 or (actions[collision_with_idx] == MOVE_FREEZE).all():  # Collision or freeze
                    return -5

        elif new_state == CLEANED_STATE:
            reward = -1

        elif new_state == EMPTY_STATE:
            reward = 3

        self.state[cleaner[0], cleaner[1]] = CLEANED_STATE
        self.cleaners[idx] = new_cleaner
        self.state[new_cleaner[0], new_cleaner[1]] = CLEANER_STATE

        return reward

    def step(self, actions=None):
        # Move cleaners
        rewards = np.zeros((self.num_cleaner, ), dtype=int)
        cleaner_status = np.zeros((self.num_cleaner, ), dtype=int)

        if actions is not None:
            for idx, action in enumerate(actions):
                if cleaner_status[idx] == IS_PROCESSING:
                    continue
                reward = self.move_cleaner(idx, cleaner_status, actions)
                rewards[idx] = reward

        if self.is_human:
            self.update_ui()

        return rewards, self.is_done()

    def update_ui(self):
        # Draw background
        self.screen.fill('white')

        # Draw the lines
        draw_lines(self.screen, self.w, self.line_width, self.block_size)

        # Update cleaned state

        draw_state_rect(self.screen, self.state, self.line_width, self.block_size)

        # Draw cleaners
        draw_cleaners(self.screen, self.cleaners, self.line_width, self.block_size)

        pygame.display.flip()

        self.clock.tick(60)


# vacuum = VacuumCleaner()
# vacuum.reset()
#
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             quit()
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 vacuum.step([MOVE_LEFT, MOVE_LEFT, MOVE_LEFT])
#                 continue
#             elif event.key == pygame.K_RIGHT:
#                 vacuum.step([MOVE_RIGHT, MOVE_RIGHT, MOVE_RIGHT])
#                 continue
#             elif event.key == pygame.K_UP:
#                 vacuum.step([MOVE_UP, MOVE_UP, MOVE_UP])
#                 continue
#             elif event.key == pygame.K_DOWN:
#                 vacuum.step([MOVE_DOWN, MOVE_DOWN, MOVE_DOWN])
#                 continue
#     vacuum.step()
