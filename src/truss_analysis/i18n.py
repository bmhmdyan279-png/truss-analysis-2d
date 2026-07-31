"""ماژول مدیریت چندزبانه (i18n) با پشتیبانی از Graceful Degradation"""

import logging

logger = logging.getLogger(__name__)
TRANSLATIONS = {
    "fa": {
        "force": "نیرو",
        "displacement": "جابجایی",
        "node": "گره",
        "element": "عضو",
        "status": "وضعیت",
    },
    "en": {
        "force": "Force",
        "displacement": "Displacement",
        "node": "Node",
        "element": "Element",
        "status": "Status",
    },
}
CURRENT_LANG = "fa"


def get_text(key: str, lang: str = CURRENT_LANG) -> str:
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS.get("fa", {}))
    if key in lang_dict:
        return lang_dict[key]
    logger.warning(f"Translation key not found: '{key}' in language '{lang}'")
    return f"!!{key}!!"
