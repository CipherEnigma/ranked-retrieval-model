import re


# Vocabularies extracted directly from data/corpus_100.txt (titles + text),
# not guessed - see the facet-extraction pass used to build this file.
# All values are lowercase; multi-word values are matched as phrases.

GENDER_VALUES = {"men", "men's", "mens", "women", "women's", "womens", "unisex"}

COLOR_VALUES = {
    "teal", "purple", "green", "grey", "gray", "blue", "lavender", "wine",
    "beige", "maroon", "black", "red", "mustard", "white", "yellow",
    "olive", "pink",
}

# Longer/multi-word fabrics listed first so they're matched before their
# shorter substrings (e.g. "comfort stretch denim" before "denim").
FABRIC_VALUES = [
    "comfort stretch denim", "cotton denim", "cotton stretch",
    "cotton slub", "cotton blend", "poly cotton", "viscose blend",
    "linen blend", "nylon spandex", "quilted polyester", "stretch denim",
    "100% cotton", "denim", "chiffon", "viscose", "cotton", "fleece",
]

SIZE_VALUES = {"s", "m", "l", "xl", "xxl"}

OCCASION_VALUES = {"casual", "office", "travel", "festive"}

CATEGORY_VALUES = {
    "dress", "hoodie", "jacket", "jeans", "kurta", "leggings", "saree",
    "shirt", "sweatshirt", "t-shirt", "tshirt",
}

GENDER_CANONICAL = {
    "men": "men", "men's": "men", "mens": "men",
    "women": "women", "women's": "women", "womens": "women",
    "unisex": "unisex",
}

CATEGORY_CANONICAL = {"tshirt": "t-shirt"}


def _extract_phrase_facet(query_lower, values):
    """
    Find and remove every occurrence of any value (single or multi-word
    phrase) from query_lower. Longer values are tried first so multi-word
    fabrics aren't partially matched by a shorter substring.

    Returns (matches, remaining_query_lower).
    """

    matches = []
    remaining = query_lower

    for value in sorted(values, key=len, reverse=True):
        pattern = r"\b" + re.escape(value) + r"\b"
        if re.search(pattern, remaining):
            matches.append(value)
            remaining = re.sub(pattern, " ", remaining)

    return matches, remaining


def parse_query(query):
    """
    Split a free-text query into structured facets + leftover free text.

    Facets: gender, color, fabric, size, occasion, category.
    Matching is done on the raw lowercased query (not stemmed), since facet
    values like sizes ("XL") and colors need exact surface-form matching,
    not stems.

    Returns:
        {
            "gender": [...],      # canonicalized, e.g. "men's" -> "men"
            "color": [...],
            "fabric": [...],
            "size": [...],        # uppercase, e.g. "XL"
            "occasion": [...],
            "category": [...],
            "free_text": "..."    # leftover query text, for the base model
        }
    """

    query_lower = query.lower()

    gender_matches, query_lower = _extract_phrase_facet(query_lower, GENDER_VALUES)
    category_matches, query_lower = _extract_phrase_facet(query_lower, CATEGORY_VALUES)
    fabric_matches, query_lower = _extract_phrase_facet(query_lower, FABRIC_VALUES)
    color_matches, query_lower = _extract_phrase_facet(query_lower, COLOR_VALUES)
    occasion_matches, query_lower = _extract_phrase_facet(query_lower, OCCASION_VALUES)
    size_matches, query_lower = _extract_phrase_facet(query_lower, SIZE_VALUES)

    free_text = " ".join(query_lower.split())

    return {
        "gender": sorted({GENDER_CANONICAL[g] for g in gender_matches}),
        "color": sorted(set(color_matches)),
        "fabric": sorted(set(fabric_matches)),
        "size": sorted({s.upper() for s in size_matches}),
        "occasion": sorted(set(occasion_matches)),
        "category": sorted({CATEGORY_CANONICAL.get(c, c) for c in category_matches}),
        "free_text": free_text,
    }
