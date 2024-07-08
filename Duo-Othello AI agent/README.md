# Duo-Othello AI Agent

This repository contains the implementation of an AI agent for the Duo-Othello game, a variant of Reversi/Othello, created for the CSCI-561 (Foundations of Artificial Intelligence) course at USC.

## Project Description

Duo-Othello is a modified version of the classic Reversi/Othello game with the following changes:
- The game is played on a 12x12 board.
- The initial state starts with 8 pieces in a specific configuration.
- The game ends when no more legal moves are possible. The winner is determined by the count of pieces, with the second player receiving a +1 bonus.

The AI agent developed in this project will compete against other agents in the class as well as those implemented by the TAs.

## Files

- `input.txt`: The input file containing the current player, remaining play times, and the board state.
- `output.txt`: The output file where the agent writes its chosen move.

## Getting Started

### Prerequisites

- Python 3.x
- Any required libraries (list them if applicable)

### Running the Agent

1. Place the `input.txt` file in the current directory with the required format.
2. Run the AI agent script.
3. The agent will generate an `output.txt` file with the chosen move.
