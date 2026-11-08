from zoneinfo import ZoneInfo

from domain.enums import Exchange

HK_TZ = ZoneInfo("Asia/Hong_Kong")
KR_TZ = ZoneInfo("Asia/Seoul")
UTC_TZ = ZoneInfo("UTC")

# 5 Major tickers per exchange
TICKERS_BY_EXCHANGE: dict[Exchange, list[dict[str, object]]] = {
    Exchange.HKEX: [
        {
            "symbol": "0700.HK",
            "name_en": "Tencent Holdings",
            "name_local": "騰訊控股",
            "code": "0700",
            "search_keywords": ["0700", "0700.HK", "騰訊", "腾讯", "Tencent"],
        },
        {
            "symbol": "9988.HK",
            "name_en": "Alibaba Group",
            "name_local": "阿里巴巴",
            "code": "9988",
            "search_keywords": ["9988", "9988.HK", "阿里巴巴", "阿里", "Alibaba"],
        },
        {
            "symbol": "3690.HK",
            "name_en": "Meituan",
            "name_local": "美團",
            "code": "3690",
            "search_keywords": ["3690", "3690.HK", "美團", "美团", "Meituan"],
        },
        {
            "symbol": "1810.HK",
            "name_en": "Xiaomi Corporation",
            "name_local": "小米集團",
            "code": "1810",
            "search_keywords": ["1810", "1810.HK", "小米", "Xiaomi"],
        },
        {
            "symbol": "9618.HK",
            "name_en": "JD.com",
            "name_local": "京東集團",
            "code": "9618",
            "search_keywords": ["9618", "9618.HK", "京東", "京东", "JD"],
        },
    ],
    Exchange.KRX: [
        {
            "symbol": "005930.KS",
            "name_en": "Samsung Electronics",
            "name_local": "삼성전자",
            "code": "005930",
            "search_keywords": ["005930", "005930.KS", "삼성전자", "삼전", "Samsung Electronics"],
        },
        {
            "symbol": "000660.KS",
            "name_en": "SK Hynix",
            "name_local": "SK하이닉스",
            "code": "000660",
            "search_keywords": ["000660", "000660.KS", "SK하이닉스", "하이닉스", "SK Hynix"],
        },
        {
            "symbol": "035420.KS",
            "name_en": "NAVER Corporation",
            "name_local": "네이버",
            "code": "035420",
            "search_keywords": ["035420", "035420.KS", "네이버", "NAVER"],
        },
        {
            "symbol": "035720.KS",
            "name_en": "Kakao",
            "name_local": "카카오",
            "code": "035720",
            "search_keywords": ["035720", "035720.KS", "카카오", "Kakao"],
        },
        {
            "symbol": "005380.KS",
            "name_en": "Hyundai Motor",
            "name_local": "현대자동차",
            "code": "005380",
            "search_keywords": ["005380", "005380.KS", "현대자동차", "현대차", "Hyundai Motor"],
        },
    ],
}
