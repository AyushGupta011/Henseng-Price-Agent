from enum import Enum


class Exchange(str, Enum):
    HKEX = "HKEX"
    KRX = "KRX"


class Direction(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class PositionClaim(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class LevelType(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT_PP = "pivot_pp"
    PIVOT_R1 = "pivot_r1"
    PIVOT_S1 = "pivot_s1"
    PIVOT_R2 = "pivot_r2"
    PIVOT_S2 = "pivot_s2"
