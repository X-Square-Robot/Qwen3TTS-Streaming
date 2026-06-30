LEVEL1_PUNCTIONS = ("。", "！", "？", ".", "!", "?")
LEVEL2_PUNCTIONS = LEVEL1_PUNCTIONS + ("，", "、", "；", "：", ",", ";", ":")
LEVEL3_PUNCTIONS = LEVEL2_PUNCTIONS + (
    "\n",
    "\r",
    "\t",
    "\f",
    "\v",
    "…",
    "……",
    "—",
    "——",
)

__all__ = ("LEVEL1_PUNCTIONS", "LEVEL2_PUNCTIONS", "LEVEL3_PUNCTIONS")
