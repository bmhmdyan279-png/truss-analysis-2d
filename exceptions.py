class TrussAnalysisError(Exception): """خطای پایه تحلیل خرپا"""
class SingularMatrixError(TrussAnalysisError): """ماتریس سختی منفرد است"""
class InputDataError(TrussAnalysisError): """داده‌های ورودی نامعتبر است"""