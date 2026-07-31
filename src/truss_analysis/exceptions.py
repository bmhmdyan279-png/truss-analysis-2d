"""مدیریت استثناها و کدهای خطای مستند (Masterplan v6.0)"""


class TrussAnalysisError(Exception):
    def __init__(self, message: str, error_code: str = "TRUSS-0000"):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class InputDataError(TrussAnalysisError):
    def __init__(self, message: str, error_code: str = "TRUSS-1001"):
        super().__init__(message, error_code)


class SingularMatrixError(TrussAnalysisError):
    def __init__(self, message: str, error_code: str = "TRUSS-2001"):
        super().__init__(message, error_code)


class SolverError(TrussAnalysisError):
    def __init__(self, message: str, error_code: str = "TRUSS-2002"):
        super().__init__(message, error_code)


class OutputError(TrussAnalysisError):
    def __init__(self, message: str, error_code: str = "TRUSS-3001"):
        super().__init__(message, error_code)
