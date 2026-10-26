# Copyright 2026 mibosuili
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable terminal formatting utilities for examples and applications."""

from collections.abc import Iterable
from typing import Any

from rich.console import Console
from rich.table import Table


def format_duration(seconds: int | float | None) -> str:
    if seconds is None:
        return "-"
    sign = "-" if seconds < 0 else ""
    rest = abs(int(seconds))
    days, rest = divmod(rest, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    parts = [f"{value}{unit}" for value, unit in (
        (days, "d"), (hours, "h"), (minutes, "m"), (seconds, "s")) if value]
    return sign + (" ".join(parts) or "0s")


def print_summary(title: str, rows: Iterable[tuple[str, Any]], *,
                  console: Console | None = None) -> None:
    table = Table(title=title, show_header=False, box=None, pad_edge=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Value")
    for label, value in rows:
        table.add_row(label, "-" if value is None else str(value))
    (console or Console()).print(table)


def print_series(title: str, headers: tuple[str, ...], rows: Iterable[Iterable[Any]], *,
                 console: Console | None = None) -> None:
    table = Table(title=title, header_style="bold cyan")
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*(str(value) for value in row))
    (console or Console()).print(table)
