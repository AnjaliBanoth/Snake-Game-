

import cv2
import numpy as np
import mediapipe as mp
import random
import math
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Game Variables
snake = [(300, 300)]
snake_length = 20
snake_direction = (0, 0)
speed = 10
food = (random.randint(100, 500), random.randint(100, 400))
score = 0
high_score = 0
is_game_over = False

# Direction smoothing buffer
smooth_x, smooth_y = 0, 0
alpha = 0.2  # Smoothing factor

# OpenCV setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Timing
prev_time = time.time()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    if is_game_over:
        cv2.putText(img, f"GAME OVER! Score: {score}", (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(img, "Press 'R' to Restart", (180, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Snake Game", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            snake = [(300, 300)]
            snake_length = 20
            food = (random.randint(100, 500), random.randint(100, 400))
            score = 0
            is_game_over = False
            snake_direction = (0, 0)
            smooth_x, smooth_y = 0, 0
        elif key == ord('q'):
            break
        continue

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            index_tip = lm[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            dx = x - snake[0][0]
            dy = y - snake[0][1]
            dist = math.hypot(dx, dy)
            if dist != 0:
                smooth_x = (1 - alpha) * smooth_x + alpha * dx / dist * speed
                smooth_y = (1 - alpha) * smooth_y + alpha * dy / dist * speed
                snake_direction = (smooth_x, smooth_y)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if snake_direction != (0, 0):
        new_head = (int(snake[0][0] + snake_direction[0]),
                    int(snake[0][1] + snake_direction[1]))
        snake.insert(0, new_head)
        if len(snake) > snake_length:
            snake.pop()

    if math.hypot(snake[0][0] - food[0], snake[0][1] - food[1]) < 20:
        food = (random.randint(50, 590), random.randint(50, 430))
        snake_length += 10
        score += 1
        speed += 0.3

    # Self-collision detection (after snake is long enough)
    if len(snake) > 50:
        for pt in snake[15:]:
            if math.hypot(snake[0][0] - pt[0], snake[0][1] - pt[1]) < 15:
                is_game_over = True
                break

    if not (0 < snake[0][0] < 640 and 0 < snake[0][1] < 480):
        is_game_over = True

    cv2.circle(img, food, 10, (0, 0, 255), -1)

    for i, point in enumerate(snake):
        color = (0, 255 - min(i * 5, 255), min(i * 5, 255))
        cv2.circle(img, point, 8, color, -1)

    cv2.putText(img, f"Score: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (530, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

