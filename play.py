# play.py
import argparse
import os
import sys
import pygame
from src.game import Game, GameResult, Color
from src.network import DQNAgent
from src.renderer import Renderer
from src.config import Config


def load_agent(checkpoint_path):
    agent = DQNAgent()
    agent.load_model(checkpoint_path)
    agent.epsilon = 0.0  # deterministic opponent
    return agent


def play(checkpoint_index, human_color):
    checkpoint_path = os.path.join(Config.MODEL_DIR, f"checkpoint_{checkpoint_index}.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading: {checkpoint_path}")

    ai = load_agent(checkpoint_path)
    game = Game()
    game.reset()
    renderer = Renderer(game)

    human_is_black = human_color == "black"
    human_col = Color.BLACK if human_is_black else Color.WHITE
    ai_col = Color.WHITE if human_is_black else Color.BLACK

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            # Human move on click
            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and game.result == GameResult.ONGOING
                and game.current_player == human_col
            ):
                cell = renderer.pixel_to_cell(*event.pos)
                if cell is not None and game._is_legal(*cell):
                    game.step(cell)

        # AI move
        if game.result == GameResult.ONGOING and game.current_player == ai_col:
            renderer.render(status="AI thinking...")
            action = ai.select_action(game)
            game.step(action)

        # Render
        if game.result != GameResult.ONGOING:
            if game.result == GameResult.DRAW:
                status = "Draw! Click to exit."
            elif (game.result == GameResult.BLACK_WIN and human_is_black) or (
                game.result == GameResult.WHITE_WIN and not human_is_black
            ):
                status = "You win! Click to exit."
            else:
                status = "AI wins! Click to exit."

            renderer.render(status=status)

            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
                        running = False
            break
        else:
            your_turn = game.current_player == human_col
            status = "Your turn" if your_turn else None
            renderer.render(status=status)

        clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against a Gomoku checkpoint.")
    parser.add_argument(
        "checkpoint",
        type=int,
        help="Checkpoint index to play against (e.g. 0 for checkpoint_0.pth, 2 for checkpoint_2.pth).",
    )
    parser.add_argument(
        "--color",
        choices=["black", "white"],
        default="black",
        help="Your color (default: black, plays first).",
    )
    args = parser.parse_args()

    play(args.checkpoint, args.color)