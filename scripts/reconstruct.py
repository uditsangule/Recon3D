from src.reconstruction.core.job import JobSpec , JobRecord
from src.utils.job_store import DirJobStore


def job_handler():
    # selects which job to run !
    return

def run_job(cfg, input_path, workdir, outdir, jobs_dir="runs"):
    spec = JobSpec(
        adapter=cfg["adapter"],
        method=cfg["method"],
        pipeline=cfg["pipeline"],
        input_path=input_path, workdir=workdir, outdir=outdir,
        name=cfg.get("name"), config=cfg
    )
    store = DirJobStore(jobs_dir)
    rec = JobRecord.new(spec)
    store.save(rec)

    try:
        rec.start(); store.save(rec)
        # pick the right facade based on spec.pipeline (example uses nerf)
        artifacts = job_handler(cfg, input_path, workdir, outdir)
        rec.complete(artifacts=artifacts); store.save(rec)
    except Exception as e:
        rec.fail(error=str(e)); store.save(rec)
        raise

    return rec