"""Same cart, track, five coloured links, force indicator and keyboard controls."""

import argparse
import math
import os
import pygame
from pathlib import Path
from environment import PendulumEnvironment, TRACK_LIMIT, CONTROL_STEP
from evaluate import load_controller
from physics import LENGTHS

COLORS = [(255, 95, 95), (95, 190, 255), (120, 235, 140), (255, 205, 95), (205, 120, 255)]


def draw(screen, environment, action, font, paused, last_result):
    screen.fill((18, 18, 22))
    center_x = 550
    ground_y = 390
    cart_x = int(center_x + environment.states[0, 0] * 820 / (2 * TRACK_LIMIT))
    pygame.draw.line(screen, (210, 210, 210), (140, ground_y), (960, ground_y), 4)
    pygame.draw.rect(screen, (235, 235, 235), (cart_x - 50, ground_y - 17, 100, 35), border_radius=6)
    joint_x = float(cart_x)
    joint_y = float(ground_y - 17)
    pygame.draw.circle(screen, (255, 255, 255), (int(joint_x), int(joint_y)), 7)
    for link in range(5):
        angle = environment.states[0, 2 + link]
        end_x = joint_x + math.sin(angle) * LENGTHS[link] * 145
        end_y = joint_y - math.cos(angle) * LENGTHS[link] * 145
        pygame.draw.line(screen, COLORS[link], (int(joint_x), int(joint_y)), (int(end_x), int(end_y)), 8)
        pygame.draw.circle(screen, COLORS[link], (int(end_x), int(end_y)), 8)
        joint_x, joint_y = end_x, end_y
    pygame.draw.line(screen, (255, 255, 255), (cart_x, ground_y + 55), (cart_x + int(action * 80), ground_y + 55), 4)
    lines = [f"{'PAUSED' if paused else 'RUNNING'}   Force: {action * 45:+.2f} N",
             f'Hold: {environment.hold_seconds[0]:.2f}s / 2.00s',
             f'Cart: {environment.states[0, 0]:+.3f} m', last_result]
    for index, text in enumerate(lines):
        screen.blit(font.render(text, True, (235, 235, 235)), (25, 25 + index * 30))
    for link in range(5):
        angle = math.degrees(environment.states[0, 2 + link])
        screen.blit(font.render(f'Link {link + 1}: {angle:+6.2f} deg', True, COLORS[link]), (25, 205 + link * 25))
    screen.blit(font.render('R reset   D disturb   SPACE pause   ESC quit   arrows: manual force', True, (190, 190, 190)), (25, 725))


def play(args):
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.init()
    screen = pygame.display.set_mode((1100, 760))
    pygame.display.set_caption('Five-Link Inverted Pendulum')
    font = pygame.font.SysFont('consolas', 20)
    clock = pygame.time.Clock()
    environment = PendulumEnvironment(seed=args.seed)
    controller = None if args.manual else load_controller(args.model)
    paused = False
    running = True
    last_result = ''
    frames = 0
    recording = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    environment.reset()
                elif event.key == pygame.K_d:
                    link = environment.random.integers(5)
                    environment.states[0, 7 + link] += environment.random.choice([-0.8, 0.8])
        action = 0.0
        if not paused:
            if args.manual:
                keys = pygame.key.get_pressed()
                action = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
            else:
                action = float(controller.action(environment.states, environment.step_number)[0])
            environment.step([action])
            if environment.finished[0]:
                passed = environment.success[0] and not environment.failed[0]
                last_result = f'Last episode: success={passed}, best hold={environment.best_hold[0]:.2f}s'
                environment.reset()
        draw(screen, environment, action, font, paused, last_result)
        pygame.display.flip()
        if args.gif and frames % 5 == 0:
            from PIL import Image
            frame = Image.frombytes('RGB', screen.get_size(), pygame.image.tobytes(screen, 'RGB'))
            recording.append(frame.resize((770, 532)))
        frames += 1
        if args.frames and frames >= args.frames:
            if args.screenshot:
                pygame.image.save(screen, args.screenshot)
            running = False
        if not args.headless:
            clock.tick(round(1 / CONTROL_STEP))
    pygame.quit()
    if recording:
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        recording[0].save(args.gif, save_all=True, append_images=recording[1:], duration=100, loop=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/scheduled.pt')
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--screenshot')
    parser.add_argument('--gif')
    play(parser.parse_args())
