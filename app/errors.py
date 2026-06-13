class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_type: str = "server_error",
        code: str = "runtime_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.code = code
