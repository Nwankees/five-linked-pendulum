import argparse
import math

import pygame
import torch

from environment import FivePoleEnv
from controllers import make_controller, load_bundle


WIDTH = 1100
HEIGHT = 760


def world_to_screen_x(x, track_limit):
    usable_width = 820
    center_x = WIDTH // 2
    scale = usable_width / (2 * track_limit)
    return int(center_x + x * scale)


def play(args):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Five-Link Inverted Pendulum RL")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)
    small_font = pygame.font.SysFont("consolas", 17)

    torch.set_num_threads(1)
    if args.manual:
        env = FivePoleEnv(seed=args.seed)
        controller = None
    else:
        swingup, catch, config, _ = load_bundle(args.model, 'cpu')
        env = FivePoleEnv(seed=args.seed, config=config)
        controller = make_controller(env.batch, swingup, catch, args.mode)
    observation = env.get_observation()
    info = {"balanced_time": 0.0, "max_hold_time": 0.0, "success": False}
    last_result = ''

    running = True
    paused = False
    total_reward = 0.0
    last_disturbed_pole = None

    colors = [
        (255, 95, 95),
        (95, 190, 255),
        (120, 235, 140),
        (255, 205, 95),
        (205, 120, 255),
    ]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    observation = env.reset()
                    total_reward = 0.0
                    if controller is not None:
                        controller.reset()
                    info = {"balanced_time": 0.0, "max_hold_time": 0.0, "success": False}
                    last_disturbed_pole = None
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_d:
                    last_disturbed_pole = env.disturb()

        action = 0.0

        if not paused:
            if args.manual:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    action -= 1.0
                if keys[pygame.K_RIGHT]:
                    action += 1.0
            else:
                action = controller.action().item()

            observation, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                last_result = f"Last episode: success={info['success']}, hold={info['max_hold_time']:.2f}s"
                observation = env.reset()
                if controller is not None:
                    controller.reset()
                total_reward = 0.0
                info = {"balanced_time": 0.0, "max_hold_time": 0.0, "success": False}

        screen.fill((18, 18, 22))

        ground_y = 390
        left_track_x = world_to_screen_x(-env.track_limit, env.track_limit)
        right_track_x = world_to_screen_x(env.track_limit, env.track_limit)
        pygame.draw.line(
            screen,
            (210, 210, 210),
            (left_track_x, ground_y),
            (right_track_x, ground_y),
            4,
        )

        x = env.state[0]
        angles = env.state[2:2 + env.num_poles]
        cart_x = world_to_screen_x(x, env.track_limit)

        cart_width = 100
        cart_height = 35
        cart_rect = pygame.Rect(
            cart_x - cart_width // 2,
            ground_y - cart_height // 2,
            cart_width,
            cart_height,
        )
        pygame.draw.rect(screen, (235, 235, 235), cart_rect, border_radius=6)

        # Connect each pole to the end of the previous one
        joint_x = float(cart_x)
        joint_y = float(ground_y - cart_height // 2)
        pixels_per_meter = 145

        pygame.draw.circle(screen, (255, 255, 255), (int(joint_x), int(joint_y)), 7)

        for index in range(env.num_poles):
            angle = angles[index]
            length = env.pole_lengths[index]
            end_x = joint_x + math.sin(angle) * length * pixels_per_meter
            end_y = joint_y - math.cos(angle) * length * pixels_per_meter

            pygame.draw.line(
                screen,
                colors[index],
                (int(joint_x), int(joint_y)),
                (int(end_x), int(end_y)),
                8,
            )
            pygame.draw.circle(screen, colors[index], (int(end_x), int(end_y)), 8)

            joint_x = end_x
            joint_y = end_y

        force_length = int(action * 80)
        if force_length != 0:
            start = (cart_x, ground_y + 55)
            end = (cart_x + force_length, ground_y + 55)
            pygame.draw.line(screen, (255, 255, 255), start, end, 4)
            pygame.draw.circle(screen, (255, 255, 255), end, 5)

        if args.manual:
            mode_text = "MANUAL"
        else:
            mode_names = [getattr(swingup, "label", "FROZEN SWING-UP"), "RESIDUAL CATCH"]
            if getattr(swingup, "has_balance", False):
                mode_names.append("IMITATED BALANCE")
            else:
                mode_names.append("LOCAL LQR")
            mode_text = mode_names[int(controller.mode[0])]
        text_lines = [
            f"Mode: {mode_text}",
            f"Hold: {info['balanced_time']:.2f}s / 2.00s",
            f"Action: {action:+.3f}",
            f"Cart x: {env.state[0]:+.3f} m",
            f"Success: {info['success']} | best hold {info['max_hold_time']:.2f}s",
            last_result,
        ]

        for i in range(len(text_lines)):
            text = text_lines[i]
            screen.blit(font.render(text, True, (235, 235, 235)), (25, 25 + i * 30))

        for i in range(len(angles)):
            angle = angles[i]
            screen.blit(
                small_font.render(
                    f"Link {i + 1}: {math.degrees(angle):+6.2f} deg",
                    True,
                    colors[i],
                ),
                (25, 205 + i * 25),
            )

        controls = "R reset   D disturb joint   SPACE pause   ESC quit"
        if args.manual:
            controls += "   LEFT/RIGHT force"
        screen.blit(
            small_font.render(controls, True, (190, 190, 190)),
            (25, HEIGHT - 35),
        )

        if last_disturbed_pole is not None:
            screen.blit(
                small_font.render(
                    f"Last disturbance: link {last_disturbed_pole + 1}",
                    True,
                    (255, 190, 100),
                ),
                (25, 350),
            )

        pygame.display.flip()
        clock.tick(round(1/env.dt))

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/imitation_complete/best.pt")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--mode", choices=['auto', 'hybrid', 'policy', 'lqr'], default='auto')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    play(args)
