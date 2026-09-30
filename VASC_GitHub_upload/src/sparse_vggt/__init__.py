__all__ = ["sparse_aggregator_from_vggt", "sparse_model_from_pi3"]


def __getattr__(name):
    if name == "sparse_aggregator_from_vggt":
        from .models.vggt import sparse_aggregator_from_vggt

        return sparse_aggregator_from_vggt
    if name == "sparse_model_from_pi3":
        from .models.pi3 import sparse_model_from_pi3

        return sparse_model_from_pi3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
