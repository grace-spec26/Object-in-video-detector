def resolve_default_device(torch_module):
    """Prefer CUDA, then Apple MPS, then CPU for CoTracker demos."""
    if torch_module.cuda.is_available():
        return "cuda"

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"
