from __future__ import annotations
from typing import Callable, Iterable, Iterator, TypeVar, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

T = TypeVar("T")
U = TypeVar("U")

def parallel_map(
    fn: Callable[[T], U],
    iterable: Iterable[T],
    workers: int = 0,
    use_processes: bool = False,
    progress: bool = True,
) -> list[U]:
    """
    Simple parallel map with optional progress bar.
    workers=0 -> auto (#cores).
    Note:  - parallel_map returns results in input order (we store by index internally).
            - For IO-bound tasks (reading images, copying files): use threads.
            - For CPU-bound tasks (heavy numpy ops): use processes.
    """
    items = list(iterable)
    if not items:
        return []
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    max_workers = None if workers in (0, None) else workers

    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(items))
        except Exception:
            pbar = None

    out: list[U] = [None] * len(items)  # type: ignore
    with Executor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, item): idx for idx, item in enumerate(items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            out[idx] = fut.result()
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()
    return out

def imap_progress(iterable: Iterable[T], total: Optional[int] = None):
    """
    Iterator wrapper that yields items with a tqdm progress bar (if available).
    """
    try:
        from tqdm import tqdm  # type: ignore
        yield from tqdm(iterable, total=total)
    except Exception:
        yield from iterable
