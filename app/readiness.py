from .types import ReadinessStatus


def service_readiness(service) -> ReadinessStatus:
    if service is None:
        return ReadinessStatus(ready=False, detail="service unavailable")
    return ReadinessStatus(ready=True)
