"""Operational control utilities for managing analysis tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Any, List


class OperationalControlSystem:
    """Schedule and execute callables using a thread pool."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def distribute(self, tasks: Iterable[Callable[[], Any]]) -> List[Any]:
        """Run tasks in parallel and return their results."""
        results: List[Any] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(task): task for task in tasks}
            for future in as_completed(future_map):
                results.append(future.result())
        return results
