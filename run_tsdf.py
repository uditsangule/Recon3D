from __future__ import annotations
import argparse, yaml
from pathlib import Path
from src.reconstruction.pipelines.run_tsdf_fusion import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=False)   # overrides inputs.root if given
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    if args.input:
        cfg.setdefault("inputs", {})["root"] = args.input

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    artifacts = run(
        cfg,
        input_path=cfg["inputs"]["root"],
        workdir=args.workdir,
        outdir=args.outdir,
    )
    print("Artifacts:", artifacts)

if __name__ == "__main__":
    main()
