"""The equations of motion, kept separate from learning and graphics.

State order: cart position, cart speed, five angles, five angular speeds.
Angles are measured from UP, in radians. Links are massless rods with a
point mass at each tip. Only the cart receives an external control force.
"""

import math
import numpy as np

LINK_COUNT = 5
LENGTHS = [0.48, 0.44, 0.40, 0.36, 0.32]
MASSES = [0.24, 0.22, 0.20, 0.18, 0.16]
CART_MASS = 1.5
GRAVITY = 9.81
CART_FRICTION = 0.10
JOINT_FRICTION = 0.003


def wrap_angle(angle):
    """Use the equivalent angle between -pi and pi."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def mass_at_or_above(link):
    total = 0.0
    for other_link in range(link, LINK_COUNT):
        total += MASSES[other_link]
    return total


def accelerations(states, forces):
    """Solve mass_matrix * acceleration = forces for a batch of carts.

    Each row/column represents the cart or one link. Off-diagonal entries
    express coupling: accelerating one part also moves the other parts.
    A numerical linear-system solver is necessary here; it is not a
    controller or a learned shortcut.
    """
    count = len(states)
    matrix = np.zeros((count, 6, 6))
    right_side = np.zeros((count, 6))
    matrix[:, 0, 0] = CART_MASS + sum(MASSES)
    right_side[:, 0] = forces - CART_FRICTION * states[:, 1]

    for link in range(LINK_COUNT):
        angle = states[:, 2 + link]
        speed = states[:, 7 + link]
        supported_mass = mass_at_or_above(link)
        coupling = supported_mass * LENGTHS[link]

        matrix[:, 0, link + 1] = coupling * np.cos(angle)
        matrix[:, link + 1, 0] = matrix[:, 0, link + 1]
        right_side[:, 0] += coupling * np.sin(angle) * speed**2
        right_side[:, link + 1] = GRAVITY * coupling * np.sin(angle)

        # Joint friction opposes motion relative to the neighbouring link.
        if link == 0:
            previous_speed = 0.0
        else:
            previous_speed = states[:, 7 + link - 1]
        friction = JOINT_FRICTION * (speed - previous_speed)
        if link < LINK_COUNT - 1:
            next_speed = states[:, 7 + link + 1]
            friction += JOINT_FRICTION * (speed - next_speed)
        right_side[:, link + 1] -= friction

        for other_link in range(LINK_COUNT):
            other_angle = states[:, 2 + other_link]
            other_speed = states[:, 7 + other_link]
            shared_mass = mass_at_or_above(max(link, other_link))
            coefficient = shared_mass * LENGTHS[link] * LENGTHS[other_link]
            difference = angle - other_angle
            matrix[:, link + 1, other_link + 1] = coefficient * np.cos(difference)
            right_side[:, link + 1] -= coefficient * np.sin(difference) * other_speed**2

    return np.linalg.solve(matrix, right_side[:, :, None])[:, :, 0]


def derivative(states, forces):
    acceleration = accelerations(states, forces)
    change = np.zeros_like(states)
    change[:, 0] = states[:, 1]
    change[:, 1] = acceleration[:, 0]
    for link in range(LINK_COUNT):
        change[:, 2 + link] = states[:, 7 + link]
        change[:, 7 + link] = acceleration[:, 1 + link]
    return change


def integrate(states, forces, time_step):
    """Runge–Kutta integration: estimate motion four times per step.

    The first estimate uses the start, two use the middle, and the last
    uses the end. Their weighted average is more accurate than Euler.
    """
    start_slope = derivative(states, forces)
    middle_state = states + time_step * start_slope / 2
    first_middle_slope = derivative(middle_state, forces)
    middle_state = states + time_step * first_middle_slope / 2
    second_middle_slope = derivative(middle_state, forces)
    end_state = states + time_step * second_middle_slope
    end_slope = derivative(end_state, forces)
    average_slope = (start_slope + 2 * first_middle_slope + 2 * second_middle_slope + end_slope) / 6
    return states + time_step * average_slope
