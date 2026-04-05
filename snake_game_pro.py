import pygame
import random
import os

pygame.init()

WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 20
SNAKE_SPEED = 15
HIGHSCORE_FILE = "highscore.txt"

display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('贪吃蛇')
clock = pygame.time.Clock()

font_style = pygame.font.Font(None, 25)
score_font = pygame.font.Font(None, 30)

YELLOW = (255, 255, 102)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
BLACK = (0, 0, 0)

def load_high_score():
    if not os.path.exists(HIGHSCORE_FILE):
        return 0
    try:
        with open(HIGHSCORE_FILE, "r") as f:
            return int(f.read())
    except:
        return 0

def save_high_score(score):
    try:
        with open(HIGHSCORE_FILE, "w") as f:
            f.write(str(score))
    except:
        pass

def display_status(score, high_score):
    value = score_font.render(f"Score: {score}  Best: {high_score}", True, YELLOW)
    display.blit(value, [10, 10])

def draw_snake(snake_list):
    for seg in snake_list:
        pygame.draw.rect(display, GREEN, (seg[0], seg[1], BLOCK_SIZE, BLOCK_SIZE))

def message(msg, color):
    txt = font_style.render(msg, True, color)
    display.blit(txt, (WIDTH/6, HEIGHT/3))

def game_loop():
    game_over = False
    game_close = False

    x, y = WIDTH//2, HEIGHT//2
    dx, dy = 0, 0
    snake_list = []
    snake_len = 1
    high_score = load_high_score()

    food_x = random.randrange(0, WIDTH - BLOCK_SIZE, BLOCK_SIZE)
    food_y = random.randrange(0, HEIGHT - BLOCK_SIZE, BLOCK_SIZE)

    while not game_over:
        # 处理游戏结束画面
        while game_close:
            display.fill(BLUE)
            cur_score = snake_len - 1
            if cur_score > high_score:
                high_score = cur_score
                save_high_score(high_score)
                msg = "新纪录! 按 Q-退出 或 C-重新开始"
            else:
                msg = "游戏结束! 按 Q-退出 或 C-重新开始"
            message(msg, RED)
            display_status(cur_score, high_score)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()
                        return

        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # 使用键盘状态检测（更灵敏）
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and dx == 0:
            dx, dy = -BLOCK_SIZE, 0
        elif keys[pygame.K_RIGHT] and dx == 0:
            dx, dy = BLOCK_SIZE, 0
        elif keys[pygame.K_UP] and dy == 0:
            dy, dx = -BLOCK_SIZE, 0
        elif keys[pygame.K_DOWN] and dy == 0:
            dy, dx = BLOCK_SIZE, 0

        # 边界碰撞
        if x >= WIDTH or x < 0 or y >= HEIGHT or y < 0:
            game_close = True

        x += dx
        y += dy
        display.fill(BLACK)
        pygame.draw.rect(display, RED, (food_x, food_y, BLOCK_SIZE, BLOCK_SIZE))

        snake_head = [x, y]
        snake_list.append(snake_head)
        if len(snake_list) > snake_len:
            del snake_list[0]

        # 自身碰撞
        for seg in snake_list[:-1]:
            if seg == snake_head:
                game_close = True

        draw_snake(snake_list)
        display_status(snake_len - 1, high_score)
        pygame.display.update()

        # 吃到食物
        if x == food_x and y == food_y:
            food_x = random.randrange(0, WIDTH - BLOCK_SIZE, BLOCK_SIZE)
            food_y = random.randrange(0, HEIGHT - BLOCK_SIZE, BLOCK_SIZE)
            snake_len += 1

        clock.tick(SNAKE_SPEED)

    pygame.quit()
    quit()

if __name__ == "__main__":
    game_loop()