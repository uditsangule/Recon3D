import numpy as np

from src.utils import imap_progress, parallel_map


def slow_double(x: int) -> int:
    # pretend work
    ar: np.ndarray = np.random.random((100 ** 3, x))
    y: np.ndarray = np.dot(ar, np.random.random(x))
    return y


max = 50
out = []
for x in imap_progress(range(max)):  # shows a tqdm bar if installed
    out.append(slow_double(5))
# print(out)

# IO-bound → threads (use_processes=False). CPU-bound → processes (use_processes=True).
out = parallel_map(slow_double, range(max), workers=8, use_processes=True, progress=True)
# print(out)  # order matches the input iterable

out = parallel_map(slow_double, range(max), workers=8, use_processes=False, progress=True)

# with nb.jit , its not usefull when np.array is there!
"""
100%|██████████| 50/50 [00:05<00:00,  9.22it/s]
100%|██████████| 50/50 [00:01<00:00, 26.46it/s]
100%|██████████| 50/50 [00:05<00:00,  9.15it/s]
"""
