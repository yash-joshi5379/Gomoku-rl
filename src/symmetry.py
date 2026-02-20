# src/symmetry.py
import numpy as np
from src.config import Config


def _flip_action_horizontal(action, board_size):
    """Flip an action (flattened index) horizontally (left-right mirror)."""
    row = action // board_size
    col = action % board_size
    new_col = board_size - 1 - col
    return row * board_size + new_col


def _transform_state_and_action(state, action, k_rot, flip, board_size):
    """
    Apply a single symmetry transformation to a state and action.

    Args:
        state:      np.array of shape (C, H, W)
        action: int in [0, board_size^2 - 1]
        k_rot:      number of 90-degree CCW rotations (0, 1, 2, 3)
        flip:       bool, whether to flip horizontally after rotation
        board_size: int

    Returns:
        (transformed_state, transformed_action)
    """
    # np.rot90 rotates the last two axes (H, W) counter-clockwise
    transformed_state = np.rot90(state, k=k_rot, axes=(1, 2)).copy()

    # Apply the same k_rot CCW rotations to the action coordinate
    # One step CCW: (row, col) -> (board_size - 1 - col, row)
    a = action
    for _ in range(k_rot):
        row = a // board_size
        col = a % board_size
        new_row = board_size - 1 - col
        new_col = row
        a = new_row * board_size + new_col

    if flip:
        transformed_state = np.flip(transformed_state, axis=2).copy()
        a = _flip_action_horizontal(a, board_size)

    return transformed_state, a


def get_symmetric_transitions(state, action, reward, next_state, done):
    """
    Generate all 8 symmetry variants of a transition.

    The square has 8 symmetries: 4 rotations (0, 90, 180, 270 CCW)
    each with and without a horizontal flip.

    Args:
        state:      np.array of shape (C, H, W)
        action: int in [0, board_size^2 - 1]
        reward:     float
        next_state: np.array of shape (C, H, W) or None (terminal)
        done:       bool

    Returns:
        List of 8 tuples: (sym_state, sym_action, reward, sym_next_state, done)
        reward and done are identical across all 8 variants.
    """
    board_size = Config.BOARD_SIZE
    symmetric = []

    for k_rot in range(4):
        for flip in (False, True):
            sym_state, sym_action = _transform_state_and_action(
                state, action, k_rot, flip, board_size
            )

            if next_state is not None:
                sym_next_state, _ = _transform_state_and_action(
                    next_state, action, k_rot, flip, board_size
                )
            else:
                sym_next_state = None

            symmetric.append((sym_state, sym_action, reward, sym_next_state, done))

    return symmetric
