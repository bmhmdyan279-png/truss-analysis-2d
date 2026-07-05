class TrussAnalysisError(Exception):
    """خطای پایه تحلیل خرپا"""

    pass


class SingularMatrixError(TrussAnalysisError):
    """ماتریس سختی منفرد است"""

    pass


class InputDataError(TrussAnalysisError):
    """داده‌های ورودی نامعتبر است"""

    pass
