# src/rewards.py
from src.config import Config
from src.game import Color


def count_line(game, row, col, dr, dc, color):
    """Count consecutive stones of given color in one direction"""
    count = 0
    r, c = row + dr, col + dc
    while 0 <= r < Config.BOARD_SIZE and 0 <= c < Config.BOARD_SIZE and game.board[r, c] == color:
        count += 1
        r += dr
        c += dc
    return count


def get_pattern_length(game, row, col, color):
    """Get maximum line length for a stone"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_length = 0
    for dr, dc in directions:
        length = (
            1
            + count_line(game, row, col, dr, dc, color)
            + count_line(game, row, col, -dr, -dc, color)
        )
        max_length = max(max_length, length)
    return max_length


def check_blocks_opponent(game, row, col, opponent_color):
    """Check if placing a stone blocks an opponent"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    max_blocked_length = 0
    for dr, dc in directions:
        length = (
            1
            + count_line(game, row, col, dr, dc, opponent_color)
            + count_line(game, row, col, -dr, -dc, opponent_color)
        )
        max_blocked_length = max(max_blocked_length, length)
    return max_blocked_length


def calculate_shaped_reward(game, action, agent_color_val, opponent_color_val):
    """Current shaped reward calculation"""
    row, col = action
    threat_length = get_pattern_length(game, row, col, agent_color_val)
    threat_reward = 0.0
    if threat_length == 4:
        threat_reward = Config.THREAT_REWARD_4
    elif threat_length == 3:
        threat_reward = Config.THREAT_REWARD_3
    elif threat_length == 2:
        threat_reward = Config.THREAT_REWARD_2

    blocked_length = check_blocks_opponent(game, row, col, opponent_color_val)
    block_reward = 0.0
    if blocked_length >= 4:
        block_reward = Config.BLOCK_REWARD_4
    elif blocked_length == 3:
        block_reward = Config.BLOCK_REWARD_3

    return threat_reward + block_reward
