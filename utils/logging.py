"""Terminal logging helpers for CLI scripts."""

G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
B = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> str:
    return f"{G}{BOLD}✔ {msg}{RESET}"


def info(msg: str) -> str:
    return f"{B}{msg}{RESET}"


def warn(msg: str) -> str:
    return f"{Y}{BOLD}⚠ {msg}{RESET}"


def err(msg: str) -> str:
    return f"{R}{BOLD}✘ {msg}{RESET}"


def stage(msg: str) -> None:
    print(f"\n>>> {msg}")
