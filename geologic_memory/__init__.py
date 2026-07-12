"""GeoLogic Memory runtime package."""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"GeoLogicMemoryService", "create_service", "runtime_options"}:
        from .api import GeoLogicMemoryService, create_service, runtime_options

        return {
            "GeoLogicMemoryService": GeoLogicMemoryService,
            "create_service": create_service,
            "runtime_options": runtime_options,
        }[name]
    raise AttributeError(name)


__all__ = ["GeoLogicMemoryService", "create_service", "runtime_options"]

