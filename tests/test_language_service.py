from app.services.language_service import (
    detect_language,
)


def test_detects_persian():

    text = (
        "کاهش مصرف آب در کشاورزی می‌تواند "
        "به حفظ منابع طبیعی کمک کند."
    )

    assert detect_language(text) == "fa"


def test_detects_english():

    text = (
        "Reliable public transportation can "
        "reduce traffic congestion in cities."
    )

    assert detect_language(text) == "en"


def test_detects_russian():

    text = (
        "Развитие общественного транспорта "
        "может уменьшить загруженность дорог."
    )

    assert detect_language(text) == "ru"


def test_empty_text_returns_none():

    assert detect_language("   ") is None


def test_detects_persian_technical_document():

    text = (
        "آموزش نتورک پالس\n\n"
        "Network Basics\n\n"
        "مفاهیم پایه ای شبکه\n"
        "تعریف شبکه\n"
        "ارتباط حداقل 2 دستگاه بر روی یک بستر ارتباطی "
        "با هدف به اشتراک گذاری منابع و تبادل اطلاعات.\n"
        "Uptime\n"
        "SLA level of 99.999 % uptime/availability.\n"
        "Network Components\n"
        "اجزای تشکیل دهنده شبکه."
    )

    assert detect_language(text) == "fa"


def test_detects_russian_technical_document():

    text = (
        "Основы сетей\n"
        "Network Basics\n"
        "Основные понятия компьютерных сетей.\n"
        "Server, Client, TCP/IP и DNS."
    )

    assert detect_language(text) == "ru"