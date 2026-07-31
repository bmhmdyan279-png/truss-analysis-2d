import logging

logger = logging.getLogger(__name__)


def safe_bidi_format(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        logger.warning("arabic_reshaper or bidi missing. Skipping bidi format.")
        return text
    except Exception as e:
        logger.warning(f"Bidi formatting failed: {e}")
        return text
