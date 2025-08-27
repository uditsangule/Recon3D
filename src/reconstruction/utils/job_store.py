from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from ..core.job import JobSpec, JobRecord

class DirJobStore:
    """
    Simple per-job JSON store:
      root/
        JOBID.json
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def save(self, rec: JobRecord) -> None:
        p = self.path_for(rec.id)
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(rec.to_json_dict(), f, indent=2)
        tmp.replace(p)

    def load(self, job_id: str) -> JobRecord:
        p = self.path_for(job_id)
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return JobRecord.from_json_dict(d)

    def list(self) -> List[JobRecord]:
        out: List[JobRecord] = []
        for jf in sorted(self.root.glob("*.json")):
            try:
                with jf.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                out.append(JobRecord.from_json_dict(d))
            except Exception:
                continue
        return out

    def exists(self, job_id: str) -> bool:
        return self.path_for(job_id).exists()
