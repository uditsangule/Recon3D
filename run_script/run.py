import time
import numpy as np
from src.reconstruction.utils.parallel import imap_progress , parallel_map

def slow_double(x: int) -> int:
    # pretend work
    ar = np.random.random((500000,x))
    y = np.dot(ar , np.random.random(x))
    return y

max = 50
out = []
for x in imap_progress(range(max)):      # shows a tqdm bar if installed
    out.append(slow_double(x))
#print(out)

# IO-bound → threads (use_processes=False). CPU-bound → processes (use_processes=True).
out = parallel_map(slow_double, range(max), workers=8, use_processes=True, progress=True)
#print(out)  # order matches the input iterable

out = parallel_map(slow_double, range(max), workers=8, use_processes=False, progress=True)

""" 100%|██████████| 50/50 [00:42<00:00,  1.17it/s]
    100%|██████████| 50/50 [00:19<00:00,  2.55it/s]
    100%|██████████| 50/50 [00:42<00:00,  1.19it/s]"""