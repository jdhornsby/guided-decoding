"""Play a chess game between two chat models, guided or unguided."""

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import chess
import chess.pgn
import chess.svg
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .guides import ChessGuide
from .loop import ThinkConfig, generate
from .models import load_model as load_model_weights, load_pretrained, make_model_forward
from .trace import Tracer
from .vocab import Vocab

THINK_END = "</think>"
MAX_MOVE_TOKENS = 16
MAX_THINK_TOKENS = 2048
HISTORY_MOVES = 10

# Qwen3's documented thinking-budget fallback, injected before forcing </think> on timeout.
# https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md
QWEN3_TIMEOUT_NUDGE = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the "
    "thinking directly now."
)

RESUME_TEXT = "\n\n"  # what every model we've traced reaches for right after </think>

# Qwen's recommended thinking-mode sampling; everything else keeps the harness defaults.
# https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
QWEN3_SAMPLING = (0.6, 0.95, 20)
DEFAULT_SAMPLING = (0.5, 1.0, 0)


def _is_qwen3(model_id: str) -> bool:
    return "qwen3" in model_id.lower()


def build_think_config(tokenizer, model_id: str, max_think_tokens: int) -> ThinkConfig | None:
    """max_think_tokens is the only general knob; the transition text depends on model family."""
    think_end = tokenizer.get_added_vocab().get(THINK_END)
    if think_end is None:
        return None
    resume_ids = tokenizer(RESUME_TEXT, add_special_tokens=False)["input_ids"]
    timeout_nudge_ids = (
        tokenizer(QWEN3_TIMEOUT_NUDGE, add_special_tokens=False)["input_ids"] if _is_qwen3(model_id) else []
    )
    return ThinkConfig(think_end, max_think_tokens, timeout_nudge_ids, resume_ids)


def build_sampling(model_id: str, temperature: float | None, top_p: float | None,
                   top_k: int | None) -> tuple[float, float, int]:
    """Falls back to model-family defaults for whatever wasn't set on the CLI."""
    default_temp, default_top_p, default_top_k = QWEN3_SAMPLING if _is_qwen3(model_id) else DEFAULT_SAMPLING
    return (
        temperature if temperature is not None else default_temp,
        top_p if top_p is not None else default_top_p,
        top_k if top_k is not None else default_top_k,
    )


# Prompts

Prompt = Callable[[chess.Board, list[str], str], list[dict]]

SYSTEM_SAN = (
    "You are playing a game of chess. You receive the opponent's moves one at a time in "
    "standard algebraic notation (SAN). Always reply with only your next move in SAN - for "
    "example 'e4', 'Nf3', 'O-O', or 'exd5' - and nothing else."
)


def san(board: chess.Board, history: list[str], colour: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_SAN}]
    if not history and colour == "white":
        messages.append({"role": "user", "content": "You are White. Play your first move."})
    own = 0 if colour == "white" else 1
    for i, move in enumerate(history):
        role = "assistant" if i % 2 == own else "user"
        messages.append({"role": role, "content": move})
    return messages


SYSTEM_ASCII = (
    "You are playing a game of chess. You are shown the current board as an 8x8 grid, "
    "White's back rank at the bottom. Uppercase letters are White's pieces, lowercase are "
    "Black's (K Q R B N P), '.' is empty. Reply with only your next move in standard "
    "algebraic notation (SAN) - for example 'e4', 'Nf3', 'O-O', or 'exd5' - and nothing else."
)


def ascii_board(board: chess.Board, history: list[str], colour: str) -> list[dict]:
    turn = "White" if board.turn == chess.WHITE else "Black"
    content = f"{board}\n\n{turn} to move."
    return [{"role": "system", "content": SYSTEM_ASCII}, {"role": "user", "content": content}]


SYSTEM_FEN = (
    "You are playing a game of chess. You are shown the current position in FEN "
    "notation: piece placement by rank from 8 down to 1, ranks separated by '/', "
    "uppercase letters are White's pieces, lowercase are Black's (K Q R B N P), and "
    "digits are consecutive empty squares. Reply with only your next move in standard "
    "algebraic notation (SAN) - for example 'e4', 'Nf3', 'O-O', or 'exd5' - and nothing else."
)


def fen(board: chess.Board, history: list[str], colour: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_FEN}, {"role": "user", "content": board.fen()}]


SYSTEM_PGN_FULL = (
    "You are playing a game of chess. You are shown the actual game so far in PGN "
    "movetext notation. A bare trailing move number with nothing after it means the "
    "game hasn't started yet and you play first. Reply with only your next move in "
    "standard algebraic notation (SAN) - for example 'e4', 'Nf3', 'O-O', or 'exd5' - "
    "and nothing else."
)

SYSTEM_PGN_WINDOWED = (
    "You are playing a game of chess. You are shown the actual game so far in PGN "
    "movetext notation. A bare trailing move number with nothing after it means the "
    "game hasn't started yet and you play first. If the game has more moves than are "
    "shown, a '[SetUp \"1\"]' / '[FEN \"...\"]' header gives the starting position for "
    "the moves shown; otherwise the moves start from the normal starting position. "
    "Reply with only your next move in standard algebraic notation (SAN) - for example "
    "'e4', 'Nf3', 'O-O', or 'exd5' - and nothing else."
)


def movetext(anchor: chess.Board, moves: list[str], board: chess.Board) -> str:
    tokens = []
    n = anchor.fullmove_number
    i = 0
    if moves and anchor.turn == chess.BLACK:
        tokens.append(f"{n}... {moves[0]}")
        i, n = 1, n + 1
    while i < len(moves):
        pair = moves[i:i + 2]
        tokens.append(f"{n}. {' '.join(pair)}")
        n += len(pair) // 2
        i += len(pair)
    if board.turn == chess.WHITE:  # bare trailing number cues White; Black's turn is implicit
        tokens.append(f"{n}.")
    return " ".join(tokens)


def pgn_full(board: chess.Board, history: list[str], colour: str) -> list[dict]:
    if not history and colour == "white":
        content = "You are White. Play your first move."
    else:
        content = movetext(chess.Board(), history, board)
    return [{"role": "system", "content": SYSTEM_PGN_FULL}, {"role": "user", "content": content}]


def pgn_windowed(board: chess.Board, history: list[str], colour: str) -> list[dict]:
    if not history and colour == "white":
        content = "You are White. Play your first move."
        return [{"role": "system", "content": SYSTEM_PGN_WINDOWED}, {"role": "user", "content": content}]
    cut = max(0, len(history) - HISTORY_MOVES * 2)
    anchor = chess.Board()
    for move in history[:cut]:
        anchor.push_san(move)
    header = f'[SetUp "1"]\n[FEN "{anchor.fen()}"]\n\n' if cut else ""
    content = header + movetext(anchor, history[cut:], board)
    return [{"role": "system", "content": SYSTEM_PGN_WINDOWED}, {"role": "user", "content": content}]


PROMPTS: dict[str, Prompt] = {
    "san": san, "ascii": ascii_board, "fen": fen, "pgn_full": pgn_full, "pgn_windowed": pgn_windowed,
}


@dataclass
class Player:
    model: object
    tokenizer: object
    vocab: Vocab
    rng: object
    tracer: Tracer
    model_id: str
    colour: str
    temperature: float
    top_p: float
    top_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="model for both players (self-play)")
    parser.add_argument("--white", default=None, dest="white_id", help="override the white model")
    parser.add_argument("--black", default=None, dest="black_id", help="override the black model")
    parser.add_argument("--white-seed", type=int, default=0, dest="white_seed")
    parser.add_argument("--black-seed", type=int, default=1, dest="black_seed")
    parser.add_argument("--prompt", choices=list(PROMPTS), default="pgn_windowed")
    parser.add_argument("--mode", choices=["guided", "unguided"], default="guided")
    parser.add_argument("--max-retries", type=int, default=5, dest="max_retries",
                        help="unguided only: illegal-move resamples before a forfeit")
    parser.add_argument("--temperature", type=float, default=None,
                        help="0.0 takes the top token, so --white-seed/--black-seed have no effect; "
                        "defaults to the model family's recommended value")
    parser.add_argument("--top-p", type=float, default=None, dest="top_p")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k", help="0 disables top-k")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--quant", default="none", choices=["none", "fp4", "nf4", "fp8"],
                        help="fp4/nf4: bitsandbytes 4-bit; fp8: torchao weight-only (memory, not speed, on Ampere)")
    parser.add_argument("--max-plies", type=int, default=200, dest="max_plies")
    parser.add_argument("--max-think-tokens", type=int, default=MAX_THINK_TOKENS, dest="max_think_tokens",
                        help="reasoning models only: thinking is cut off past this many tokens")
    parser.add_argument("--show-thinking", action="store_true", dest="show_thinking",
                        help="print each move's reasoning (reasoning models only)")
    parser.add_argument("--run-id", default="", dest="run_id")
    args = parser.parse_args()
    args.white_id = args.white_id or args.model
    args.black_id = args.black_id or args.model
    return args


def load_model(model_id: str, device: str, dtype: torch.dtype, quant: str = "none"):
    tokenizer = load_pretrained(AutoTokenizer, model_id)
    model = load_model_weights(AutoModelForCausalLM, model_id, device, dtype, quant)
    return tokenizer, model


def make_player(model, tokenizer, model_id: str, seed: int, colour: str,
                args: argparse.Namespace, run_id: str) -> Player:
    rng = np.random.default_rng(seed)
    vocab = Vocab(tokenizer, model.config.vocab_size)
    temperature, top_p, top_k = build_sampling(model_id, args.temperature, args.top_p, args.top_k)
    tracer = Tracer(tokenizer, run_id, path=f"traces/{run_id}-{colour}.jsonl")
    tracer.meta({
        "run_id": run_id,
        "model": model_id,
        "seed": seed,
        "colour": colour,
        "prompt": args.prompt,
        "mode": args.mode,
        "max_retries": args.max_retries,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "device": args.device,
        "dtype": args.dtype,
        "quant": args.quant,
        "max_think_tokens": args.max_think_tokens,
        "vocab_size": len(vocab.token_bytes),
    })
    return Player(model, tokenizer, vocab, rng, tracer, model_id, colour, temperature, top_p, top_k)


def split_reply(tokenizer, ids: list[int], prompt_len: int, think_end: int | None) -> tuple[str, str]:
    """Split a generation into (thinking, answer); thinking is '' for non-reasoning models."""
    reply = ids[prompt_len:]
    if think_end is None or think_end not in reply:
        return "", tokenizer.decode(reply, skip_special_tokens=True)
    split = reply.index(think_end)
    thinking = tokenizer.decode(reply[:split], skip_special_tokens=True).strip()
    answer = tokenizer.decode(reply[split + 1:], skip_special_tokens=True)
    return thinking, answer


def parse_move(text: str) -> str:
    t = re.sub(r"^\d+\.+\s*", "", text.strip().strip('"').lstrip())
    return t.split()[0].rstrip(".,!?") if t.split() else ""


def get_move(player: Player, board: chess.Board, history: list[str], prompt: Prompt,
             args: argparse.Namespace) -> tuple[str | None, int, str]:
    """Propose moves until one is legal or retries run out. Returns
    (legal_move | None, retries_used, last_attempt)."""
    think = build_think_config(player.tokenizer, player.model_id, args.max_think_tokens)
    max_tokens = (think.max_tokens + MAX_MOVE_TOKENS) if think else MAX_MOVE_TOKENS

    base = prompt(board, history, player.colour)
    
    guide = ChessGuide(player.vocab, board) if args.mode == "guided" else None

    scratch: list[dict] = []
    move = ""
    for retries in range(args.max_retries + 1):
        messages = base + scratch
        text = player.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = player.tokenizer(text)["input_ids"]

        forward = make_model_forward(player.model, args.device)
        ids = generate(forward, player.tokenizer, guide, prompt_ids, max_tokens,
                       player.temperature, player.top_p, player.rng, player.tracer, player.top_k, think)
        think_end = think.end_token if think else None
        thinking, answer = split_reply(player.tokenizer, ids, len(prompt_ids), think_end)
        if args.show_thinking and thinking:
            print(f"    {player.colour} thinking: {thinking}")
        move = parse_move(answer)

        try:
            if move:
                board.parse_san(move)
                return move, retries, move
        except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            pass
        scratch += [
            {"role": "assistant", "content": move or "?"},
            {"role": "user", "content": f"'{move}' is not a legal move. Reply with a different legal move in SAN."},
        ]
    return None, args.max_retries, move


def render(board: chess.Board, path: Path) -> None:
    last = board.move_stack[-1] if board.move_stack else None
    path.write_text(chess.svg.board(board, lastmove=last))


def play(white: Player, black: Player, args: argparse.Namespace, run_id: str) -> None:
    board = chess.Board()
    frames = Path("traces") / run_id
    frames.mkdir(parents=True, exist_ok=True)
    render(board, frames / "000-start.svg")

    prompt = PROMPTS[args.prompt]
    tag = f"{args.prompt}, {args.mode}, max-retries {args.max_retries}"
    print(f"Executing game: {white.model_id} vs {black.model_id}  ({tag})")

    forfeit = None
    history: list[str] = []
    while not board.is_game_over() and board.ply() < args.max_plies:
        player = white if board.turn == chess.WHITE else black
        colour = "w" if board.turn == chess.WHITE else "b"
        number = board.fullmove_number

        move, retries, last = get_move(player, board, history, prompt, args)
        if move is None:
            print(f"{number:>3}. {colour}  FAILED after {retries} retries (last: {last!r})")
            forfeit = colour
            break

        board.push_san(move)
        history.append(move)
        render(board, frames / f"{board.ply():03d}-{move}.svg")
        print(f"{number:>3}. {colour}  {move}" + (f"   ({retries} retries)" if retries else ""))

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = white.model_id
    game.headers["Black"] = black.model_id
    pgn_path = Path("traces") / f"{run_id}.pgn"
    pgn_path.write_text(str(game) + "\n")

    if forfeit:
        print(f"result: {forfeit} forfeited after {board.ply()} legal plies")
    else:
        print(f"result: {board.result()} ({board.ply()} plies)")
    print(f"pgn: {pgn_path}")
    print(f"frames: {frames}")


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    dtype = getattr(torch, args.dtype)

    white_tokenizer, white_model = load_model(args.white_id, args.device, dtype, args.quant)
    if args.black_id == args.white_id:
        black_tokenizer, black_model = white_tokenizer, white_model
    else:
        black_tokenizer, black_model = load_model(args.black_id, args.device, dtype, args.quant)

    white = make_player(white_model, white_tokenizer, args.white_id, args.white_seed, "white", args, run_id)
    black = make_player(black_model, black_tokenizer, args.black_id, args.black_seed, "black", args, run_id)

    try:
        play(white, black, args, run_id)
    finally:
        white.tracer.close()
        black.tracer.close()


if __name__ == "__main__":
    main()
