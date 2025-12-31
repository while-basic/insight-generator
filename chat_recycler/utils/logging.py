from __future__ import annotations

from rich.console import Console

console = Console()


def log_info(message: str) -> None:
    console.print(f"[bold cyan]info[/]: {message}")


def log_warning(message: str) -> None:
    console.print(f"[bold yellow]warn[/]: {message}")


def log_error(message: str) -> None:
    console.print(f"[bold red]error[/]: {message}")
