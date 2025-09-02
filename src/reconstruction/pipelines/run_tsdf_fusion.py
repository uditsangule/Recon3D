from pathlib import Path
from src.registry import get_adapter , get_method , load_entry_point_plugins
from src.utils import get_logger , configure_logging

log = get_logger(__name__)
configure_logging(rich=False)


def _cfg_adapter_block(cfg, adapter_name):
    adapters = cfg.get("inputs" , {})
    if isinstance(adapters, dict) and adapter_name in adapters:
        return adapters.get(adapter_name, {})
    # universal style
    return (cfg.get("adapters", {})).get(adapter_name, {})

def _cfg_method_block(cfg, method_name):
    if "tsdf" in cfg:
        m = cfg.get("tsdf" , {})
    else:
        m = (cfg.get("methods", {})).get(method_name , {})
    exec_cfg = cfg.get("execution", {})
    return {**m , **exec_cfg}

def run(cfg,input_path, workdir, outdir):
    scan_name = cfg.get("name" , "unknown")
    workdir_p = Path(workdir);
    workdir_p.mkdir(parents=True, exist_ok=True)

    outdir_p = Path(outdir);
    outdir_p.mkdir(parents=True, exist_ok=True)

    load_entry_point_plugins()

    adapter_name = cfg.get("adapter", "rtabmap")
    method_name = cfg.get("method", "tsdf_fusion")

    # 1build adapter
    AdapterCls = get_adapter(adapter_name)
    a_kwargs = _cfg_adapter_block(cfg, adapter_name)
    adapter = AdapterCls(**a_kwargs)

    if not adapter.probe(input_path):
        raise RuntimeError(f"Adapter '{adapter_name}' cannot handle input: {input_path}")

    # load scene
    log.info("loading scene via adapter '%s'...", adapter_name)
    scene = adapter.load(input_path)
    log.info("scene loaded: id=%s frames=%d", scene.id, len(scene.frames))

    # Save normalized scene
    # to do it later

    # build reconstructor (TSDF strategy)
    MethodCls = get_method(method_name)
    m_kwargs = _cfg_method_block(cfg, method_name)
    method = MethodCls(**m_kwargs)

    # Optional: validate needs if your method implements need()

    # run method
    log.info("preparing method '%s'...", method_name)
    method.prepare(scene, workdir_p)

    log.info("running method '%s'...", method_name)
    artifacts = method.run(scene, workdir_p)

    # run other on artifacts
    return artifacts