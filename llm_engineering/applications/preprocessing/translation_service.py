import re
from typing import Dict, List, Tuple
from loguru import logger


class GeometryTranslator:
    """Service for translating geometry text from English to Vietnamese"""

    # Translation rules ordered by priority (process first to last)
    TRANSLATION_RULES: List[Tuple[str, str]] = [
        # 1. FIXED PHRASES & QUANTIFIERS
        # Must be processed first to avoid grammatical conflicts
        ("two concentric circles", "hai đường tròn đồng tâm"),
        ("three concentric circles", "ba đường tròn đồng tâm"),
        ("concentric circles", "các đường tròn đồng tâm"),
        ("perpendicular bisector", "đường trung trực"),
        ("is perpendicular to", "vuông góc với"),
        ("is parallel to", "song song với"),
        ("is tangent to", "tiếp xúc với"),
        ("is similar to", "đồng dạng với"),
        ("is congruent to", "bằng"),
        ("is inside", "nằm trong"),

        # 2. LONG GEOMETRIC TERMS
        ("equilateral triangle", "tam giác đều"),
        ("isosceles triangle", "tam giác cân"),
        ("right triangle", "tam giác vuông"),
        ("circumcircle", "đường tròn ngoại tiếp"),
        ("circumcenter", "tâm đường tròn ngoại tiếp"),
        ("incircle", "đường tròn nội tiếp"),
        ("incenter", "tâm đường tròn nội tiếp"),
        ("excircle", "đường tròn bàng tiếp"),
        ("excenter", "tâm đường tròn bàng tiếp"),
        ("parallelogram", "hình bình hành"),
        ("quadrilateral", "tứ giác"),
        ("rectangle", "hình chữ nhật"),
        ("trapezoid", "hình thang"),
        ("pentagon", "ngũ giác"),

        # 3. COMMON ERROR FIXES
        ("extension of", "phần kéo dài của"),
        ("extension", "phần kéo dài"),
        ("centroid of", "trọng tâm của"),
        ("centroid", "trọng tâm"),
        ("diagonal", "đường chéo"),

        # 4. PLURAL NOUNS
        # Must be handled before singulars to avoid malformed words
        ("circles", "các đường tròn"),
        ("angles", "các góc"),
        ("segments", "các đoạn thẳng"),
        ("lines", "các đường thẳng"),
        ("rays", "các tia"),
        ("points", "các điểm"),

        # 5. SINGULAR NOUNS
        ("circle", "đường tròn"),
        ("angle", "góc"),
        ("segment", "đoạn thẳng"),
        ("line", "đường thẳng"),
        ("ray", "tia"),
        ("point", "điểm"),
        ("diameter", "đường kính"),
        ("radius", "bán kính"),
        ("triangle", "tam giác"),
        ("square", "hình vuông"),

        # 6. CONNECTORS & PREPOSITIONS
        ("is the", "là"),
        ("is on", "nằm trên"),
        ("intersects", "cắt"),
        ("intersect", "cắt"),
        ("through", "đi qua"),
        ("midpoint of", "trung điểm của"),
        ("with", "với"),
        ("and", "và"),
        ("of", "của"),
        ("at", "tại"),

        # 7. NUMBERS
        ("two", "hai"),
        ("three", "ba"),
        ("one", "một"),
        
        # 8. OTHERS
        ("is", "là"),
        ("are", "là")
    ]

    # Special geometry symbols
    SYMBOL_MAP: Dict[str, str] = {
        "⊥": "vuông góc với",
        "//": "song song với"
    }

    @classmethod
    def translate(cls, text: str) -> str:
        """
        Translate geometry text from English to Vietnamese.

        Args:
            text: English geometry description

        Returns:
            Vietnamese translation
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid input text: {text}")
            return ""

        # STEP 1: REMOVE INDEFINITE ARTICLE "a"
        # Prevent incorrect translations like "điểm một"
        # Only remove lowercase 'a' not followed by uppercase letter (Point A)
        text = re.sub(r'\ba\s+(?![A-Z])', '', text)

        # Handle sentence starting with "A line ..." → "Line ..."
        text = re.sub(r'^A\s+(?=[a-z])', '', text)

        # STEP 2: APPLY TRANSLATION RULES SEQUENTIALLY
        # Order matters – rules are applied top-down
        for en_term, vn_term in cls.TRANSLATION_RULES:
            pattern = re.compile(
                r'\b' + re.escape(en_term) + r'\b',
                re.IGNORECASE
            )
            text = pattern.sub(vn_term, text)

        # STEP 3: REPLACE SPECIAL SYMBOLS
        for sym, val in cls.SYMBOL_MAP.items():
            text = text.replace(sym, val)

        # STEP 4: POST-PROCESSING GRAMMAR FIXES
        # Remove "các" after numbers: "ba các đường tròn" → "ba đường tròn"
        text = re.sub(
            r'(hai|ba|bốn|năm)\s+các\s+',
            r'\1 ',
            text,
            flags=re.IGNORECASE
        )

        # Remove unnecessary determiners
        text = re.sub(r'\b(the|that)\b', '', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        return text

    @classmethod
    def translate_batch(cls, texts: List[str]) -> List[str]:
        """
        Translate a batch of geometry texts.

        Args:
            texts: List of English geometry descriptions

        Returns:
            List of Vietnamese translations
        """
        results = []
        for text in texts:
            try:
                translated = cls.translate(text)
                results.append(translated)
            except Exception as e:
                logger.error(f"Error translating text '{text}': {e}")
                results.append("")

        logger.info(f"Translated {len(results)} texts")
        return results
