try:
    from .settings import parse_settings
except ImportError:
    from settings import parse_settings


def main(argv=None, env=None, runner=None, app_factory=None, service_cls=None) -> None:
    if runner is None:
        import uvicorn

        runner = uvicorn.run
    if app_factory is None:
        try:
            from .app import create_app
        except ImportError:
            from app import create_app

        app_factory = create_app
    if service_cls is None:
        try:
            from .service import ChatCompletionService
        except ImportError:
            from service import ChatCompletionService

        service_cls = ChatCompletionService

    settings = parse_settings(argv=argv, env=env) if env is not None else parse_settings(argv=argv)
    app = app_factory(service_factory=lambda: service_cls(config=settings.runtime))
    runner(
        app,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        log_level=settings.server.log_level,
    )


if __name__ == "__main__":
    main()
