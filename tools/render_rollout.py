# Render a video and graphs from a recorded simulator rollout
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace')
    parser.add_argument('--output', default='logs/imitation_demo.mp4')
    args = parser.parse_args()

    data = np.load(args.trace)
    states = data['states'][:, 0]
    actions = data['actions'][:, 0]
    dt = float(data['dt'])

    time = np.arange(len(states)) * dt
    angles = np.unwrap(states[:, 2:7], axis=0)
    angles += 2 * np.pi * np.round((np.pi - angles[0]) / (2 * np.pi))
    colors = ['#ff666e', '#57b8ef', '#80d89c', '#f2c35e', '#bd89ee']

    # Plot the angles, rates, cart position, and force
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i in range(len(colors)):
        axes[0].plot(time, np.degrees(angles[:, i]), color=colors[i], label=f'Link {i + 1}')
        axes[1].plot(time, states[:, 7 + i], color=colors[i])

    axes[0].axhspan(-10, 10, color='green', alpha=0.1)
    axes[0].set_ylabel('Angle (degrees)')
    axes[0].legend(ncol=5, loc='upper right')

    axes[1].axhspan(-0.5, 0.5, color='green', alpha=0.1)
    axes[1].set_ylabel('Rate (rad/s)')

    axes[2].plot(time, states[:, 0], color='#3578ac')
    axes[2].axhline(2.4, color='red', linestyle='--')
    axes[2].axhline(-2.4, color='red', linestyle='--')
    axes[2].set_ylabel('Cart x (m)')

    axes[3].plot(time[:-1], actions * 45, color='#555555')
    axes[3].axhline(45, color='red', linestyle='--')
    axes[3].axhline(-45, color='red', linestyle='--')
    axes[3].set_ylabel('Force (N)')
    axes[3].set_xlabel('Time (s)')

    for axis in axes:
        axis.grid(alpha=0.2)

    fig.suptitle('Learned imitation: hanging start to sustained upright balance')
    fig.tight_layout()
    fig.savefig(Path(args.output).with_suffix('.png'), dpi=150)
    plt.close(fig)

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required for MP4; the diagnostic plot was saved')

    width = 900
    height = 650
    fps = 25
    command = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
               '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.output]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    small_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    lengths = np.array([0.48, 0.44, 0.40, 0.36, 0.32])
    scale = 130.0
    center_x = width / 2
    ground_y = 335

    # Draw every other recorded state
    for index in range(0, len(states), 2):
        state = states[index]
        frame = Image.new('RGB', (width, height), '#15171d')
        draw = ImageDraw.Draw(frame)

        draw.text((25, 22), 'Learned imitation controller', font=font, fill='white')
        draw.text((25, 56), f'Time {index * dt:5.2f} s  |  cart {state[0]:+.2f} m', font=small_font, fill='#dddddd')

        max_angle = float(abs(np.degrees(state[2:7])).max())
        max_speed = float(abs(state[7:]).max())
        draw.text((25, 83), f'Worst link: {max_angle:.2f} degrees  |  fastest link: {max_speed:.3f} rad/s', font=small_font, fill='#dddddd')

        if index * dt < 5:
            phase = 'Swing-up phase'
        else:
            phase = 'Learned balance phase'
        draw.text((25, 111), phase, font=small_font, fill='#83d6b0')

        draw.line((center_x - 2.4 * scale, ground_y, center_x + 2.4 * scale, ground_y), fill='#9b9da2', width=3)
        x = center_x + state[0] * scale
        y = ground_y - 10
        draw.rounded_rectangle((x - 35, ground_y - 12, x + 35, ground_y + 12), radius=5, fill='#eeeeee')
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='white')

        for j in range(len(lengths)):
            length = lengths[j]
            next_x = x + length * scale * np.sin(state[2 + j])
            next_y = y - length * scale * np.cos(state[2 + j])
            draw.line((x, y, next_x, next_y), fill=colors[j], width=7)
            draw.ellipse((next_x - 6, next_y - 6, next_x + 6, next_y + 6), fill=colors[j])
            x = next_x
            y = next_y

        action_index = min(index, len(actions) - 1)
        force = actions[action_index] * 45
        draw.text((25, height - 51), f'Applied cart force: {force:+.2f} N  |  limit: +/-45 N', font=small_font, fill='#dddddd')
        draw.text((25, height - 28), 'Recorded simulator trajectory; all five joints are passive.', font=small_font, fill='#a9acb5')
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')
    print(args.output)


if __name__ == '__main__':
    main()
