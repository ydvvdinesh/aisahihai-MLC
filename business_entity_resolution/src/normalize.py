"""Text normalisation for business names and addresses.

Country-agnostic by design: every rule is either script-generic (Indic
transliteration via Unicode character names, accent stripping) or a
token-level abbreviation canonicalisation that covers common English,
Indian and French address/legal vocabulary. No country label is used.
"""
import re
import unicodedata
from functools import lru_cache

# --------------------------------------------------------------------------
# Indic (Brahmic) transliteration, driven by Unicode character names so one
# routine covers Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
# Telugu, Kannada and Malayalam.
# --------------------------------------------------------------------------
_INDIC_SCRIPTS = ("DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA",
                  "TAMIL", "TELUGU", "KANNADA", "MALAYALAM")
_VOWELS = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u",
           "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li",
           "E": "e", "EE": "e", "SHORT E": "e", "AI": "ai", "O": "o",
           "OO": "o", "SHORT O": "o", "AU": "au", "CANDRA E": "e", "CANDRA O": "o"}
_CONS_FIX = {"nn": "n", "tt": "t", "dd": "d", "ss": "sh", "ll": "l", "rr": "r",
             "ny": "n", "nng": "n", "ng": "n", "lll": "l", "ddh": "dh", "tth": "th"}


def _build_indic_table():
    table = {}
    for cp in list(range(0x0900, 0x0D80)):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        script = name.split(" ")[0]
        if script not in _INDIC_SCRIPTS:
            continue
        rest = name[len(script) + 1:]
        if rest.startswith("LETTER "):
            l = rest[7:]
            if l in _VOWELS:
                table[ch] = ("V", _VOWELS[l])
            elif l.endswith("A"):
                c = l[:-1].lower()
                table[ch] = ("C", _CONS_FIX.get(c, c))
        elif rest.startswith("VOWEL SIGN "):
            l = rest[11:]
            table[ch] = ("M", _VOWELS.get(l, ""))
        elif rest.startswith("SIGN VIRAMA"):
            table[ch] = ("X", "")
        elif rest.startswith(("SIGN ANUSVARA", "SIGN CANDRABINDU", "TIPPI", "SIGN BINDI")):
            table[ch] = ("N", "n")
        elif rest.startswith("SIGN VISARGA"):
            table[ch] = ("N", "h")
        elif rest.startswith("DIGIT "):
            table[ch] = ("D", str(unicodedata.digit(ch)))
        elif rest.startswith(("AU LENGTH MARK", "SIGN NUKTA", "ADDAK")):
            table[ch] = ("X0", "")
    return table


_INDIC = _build_indic_table()
_INDIC_RE = re.compile("[ऀ-ൿ]")


def translit_indic(s):
    if not _INDIC_RE.search(s):
        return s
    out = []
    pending = False  # consonant waiting for its inherent vowel
    for ch in s:
        t = _INDIC.get(ch)
        if t is None:
            if pending:
                out.append("a")
                pending = False
            out.append(ch)
            continue
        kind, val = t
        if kind == "C":
            if pending:
                out.append("a")
            out.append(val)
            pending = True
        elif kind == "M":
            out.append(val)
            pending = False
        elif kind == "X":
            pending = False
        elif kind == "X0":
            pass
        else:
            if pending:
                out.append("a")
                pending = False
            out.append(val)
    if pending:
        out.append("a")
    res = "".join(out).replace("rr", "t")
    # schwa deletion at word end ("raama" -> "raam")
    res = re.sub(r"(?<=[b-df-hj-np-tv-z])a\b", "", res)
    return res


# --------------------------------------------------------------------------
# Generic cleanup
# --------------------------------------------------------------------------
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "@": "a", "$": "s"})
_NONALNUM = re.compile(r"[^a-z0-9]+")


def base_clean(s):
    s = translit_indic(s)
    s = strip_accents(s).lower()
    s = s.replace("&", " and ").replace("+", " and ")
    return s


def fix_leet(tok):
    # "5ecure" -> "secure", "c0mpany" -> "company"; leave pure numbers alone
    if tok.isdigit() or not any(c.isalpha() for c in tok):
        return tok
    return tok.translate(_LEET)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
LEGAL = {
    "inc", "incorporated", "llc", "l", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "lp", "llp", "pc", "plc", "pllc", "lc", "the",
    "sarl", "sas", "sa", "eurl", "sci", "ei", "sasu", "snc", "scop", "gie", "selarl",
    "and", "of", "de", "du", "des", "la", "le", "les", "et", "pty", "gmbh", "ag", "opc",
}
FILLER = {"services", "service", "center", "centre", "group", "groupe", "partners",
          "holdings", "holding", "solutions", "enterprises", "enterprise", "international",
          "global", "compagnie", "cie", "etablissements", "ets", "shri", "sri", "shree", "m", "s", "ms"}
_ALIAS_RE = re.compile(
    r"\b(?:d\s*/\s*b\s*/\s*a|dba|doing business as|formerly known as|formerly|"
    r"f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|also known as|trading as|t\s*/\s*a)\b")
_WEB_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co|fr|biz|info|us|io)\b")


def name_tokens(s):
    return [fix_leet(t) for t in _NONALNUM.split(s) if t]


def normalize_name(raw):
    """Returns dict with several views of a business name."""
    s = base_clean(raw)
    is_web = 0
    m = _WEB_RE.search(s)
    if m:
        is_web = 1
        s = _WEB_RE.sub(lambda mm: " " + mm.group(1) + " ", s)
    parts = [p for p in _ALIAS_RE.split(s)]
    has_alias = int(len(parts) > 1)
    part_toks = [name_tokens(p) for p in parts]
    part_toks = [p for p in part_toks if p] or [[]]
    full = [t for p in part_toks for t in p]
    cores = []
    for p in part_toks:
        c = [t for t in p if t not in LEGAL and t not in FILLER and skeleton_word(t) not in LEGAL_SKEL]
        cores.append(" ".join(c) if c else " ".join(p))
    return {
        "n_full": " ".join(full),
        "n_core": cores[-1] if has_alias else cores[0],
        "n_alias": "|".join(cores),
        "n_nospace": "".join(cores[0].split()) if not has_alias else "|".join("".join(c.split()) for c in cores),
        "is_web": is_web,
        "has_alias": has_alias,
    }


_SKEL_SUBS = [("ph", "f"), ("w", "v"), ("q", "k"), ("x", "ks"), ("c", "k"), ("z", "j"), ("g", "j"), ("h", "")]
_VOW = re.compile(r"[aeiouy]")


@lru_cache(maxsize=1_000_000)
def skeleton_word(w):
    if w.isdigit():
        return w
    for a, b in _SKEL_SUBS:
        w = w.replace(a, b)
    w = _VOW.sub("", w)
    return re.sub(r"(.)\1+", r"\1", w)


LEGAL_SKEL = {"prvt", "lmtd", "lmtt", "prvtd"}


def skeleton(s):
    return " ".join(k for k in (skeleton_word(w) for w in s.split()) if k)


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------
ADDR_CANON = {}
for canon, variants in {
    "st": ["street", "st", "str", "saint", "sainte", "stree", "streeet"],
    "rd": ["road", "rd"], "av": ["avenue", "ave", "av", "aven", "avn"],
    "dr": ["drive", "dr", "drv"], "ln": ["lane", "ln"], "ct": ["court", "ct", "crt"],
    "cir": ["circle", "cir", "circ"], "bd": ["boulevard", "blvd", "bd", "boul", "bvd"],
    "pl": ["place", "pl"], "pkwy": ["parkway", "pkwy", "pky"], "hwy": ["highway", "hwy"],
    "ter": ["terrace", "ter", "terr"], "trl": ["trail", "trl"], "sq": ["square", "sq"],
    "n": ["north", "n", "nord"], "s": ["south", "s", "sud"], "e": ["east", "e", "est"],
    "w": ["west", "w", "ouest"], "rue": ["rue", "r"], "ch": ["chemin", "ch", "chem"],
    "imp": ["impasse", "imp"], "all": ["allee", "all"], "rte": ["route", "rte"],
    "qu": ["quai", "qu"], "mt": ["mount", "mt", "mont"], "ft": ["fort", "ft"],
    "pt": ["point", "pt"], "ctr": ["center", "ctr", "centre", "cntr"],
    "hts": ["heights", "hts"], "ext": ["extension", "extn", "ext"],
    "bldg": ["building", "bldg", "bldng"], "opp": ["opposite", "opp"],
    "sec": ["sector", "sec", "sect"], "col": ["colony", "col"], "ngr": ["nagar", "ngr"],
    "mkt": ["market", "mkt"], "cross": ["cross", "crs"], "stn": ["station", "stn"],
    "way": ["way", "wy"], "pk": ["park", "pk"], "vlg": ["village", "vlg", "vill"],
    "jn": ["junction", "jn", "jct"], "fac": ["faubourg", "fg", "fbg"],
    "ml": ["mill", "ml"], "cv": ["cove", "cv"], "pass": ["passage", "pass"],
}.items():
    for v in variants:
        ADDR_CANON[v] = canon

_ORD_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
              "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
              "seventeenth", "eighteenth", "nineteenth", "twentieth"]
_SUF = {1: "st", 2: "nd", 3: "rd"}
ORDINALS = {w: f"{i}{_SUF.get(i if i < 4 else 0, 'th')}" for i, w in enumerate(_ORD_WORDS, 1)}
ORDINALS.update({"thirtieth": "30th", "fortieth": "40th", "fiftieth": "50th"})

ADDR_STOP = {"de", "du", "des", "la", "le", "les", "d", "l", "of", "the", "and", "near", "nr",
             "no", "door", "house", "h", "hno", "flat", "unit", "apt", "apartment", "ste",
             "suite", "fl", "floor", "flr", "pmb", "po", "box", "plot", "cdp", "number", "num",
             "shop", "room", "rm", "dist", "district", "tq", "taluk", "tal", "na", "n/a",
             "et", "au", "aux", "en", "sur", "null", "none", "nan"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py",
    # transliterated native-script spellings
    "dilli": "dl", "tamilnatu": "tn", "tamilnadu": "tn", "pashcimabang": "wb", "pashcimbang": "wb",
    "pashchimbang": "wb", "keralam": "kl", "karnatak": "ka", "gujarat": "gj", "odisa": "od",
    "telanganaa": "tg", "panjab": "pb", "asam": "as", "hariyan": "hr", "hariyana": "hr",
    "keralan": "kl", "pnjab": "pb", "punjab": "pb", "odish": "od",
}
# French administrative areas (regions + departments): like US/Indian states,
# sources disagree on which level they write, so they are dropped.
FR_ADMIN = [
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire", "corse",
    "grand est", "hauts de france", "ile de france", "normandie", "nouvelle aquitaine", "occitanie",
    "pays de la loire", "provence alpes cote d azur", "paca", "guadeloupe", "martinique", "guyane",
    "la reunion", "mayotte",
    "ain", "aisne", "allier", "alpes de haute provence", "hautes alpes", "alpes maritimes", "ardeche",
    "ardennes", "ariege", "aube", "aude", "aveyron", "bouches du rhone", "calvados", "cantal", "charente",
    "charente maritime", "cher", "correze", "corse du sud", "haute corse", "cote d or", "cotes d armor",
    "creuse", "dordogne", "doubs", "drome", "eure", "eure et loir", "finistere", "gard", "haute garonne",
    "gers", "gironde", "herault", "ille et vilaine", "indre", "indre et loire", "isere", "jura", "landes",
    "loir et cher", "loire", "haute loire", "loire atlantique", "loiret", "lot", "lot et garonne", "lozere",
    "maine et loire", "manche", "marne", "haute marne", "mayenne", "meurthe et moselle", "meuse", "morbihan",
    "moselle", "nievre", "nord", "oise", "orne", "pas de calais", "puy de dome", "pyrenees atlantiques",
    "hautes pyrenees", "pyrenees orientales", "bas rhin", "haut rhin", "rhone", "haute saone",
    "saone et loire", "sarthe", "savoie", "haute savoie", "paris", "seine maritime", "seine et marne",
    "yvelines", "deux sevres", "somme", "tarn", "tarn et garonne", "var", "vaucluse", "vendee", "vienne",
    "haute vienne", "vosges", "yonne", "territoire de belfort", "essonne", "hauts de seine",
    "seine saint denis", "val de marne", "val d oise",
]
_NUM_RE = re.compile(r"\d+")
_FR_ADMIN = set(FR_ADMIN)
_STATE_SKEL = None  # filled below
_STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())


def normalize_address(raw):
    s = base_clean(raw)
    # "#123", "h.no.12" etc -> spaced
    s = s.replace("#", " ").replace("n°", " ").replace("no°", " ")
    comps = [c.strip() for c in s.split(",") if c.strip()]
    toks = []
    for c in comps:
        c2 = " ".join(_NONALNUM.split(c)).strip()
        # whole component that is a state name -> drop (low information, noisy format)
        if c2 in US_STATES or c2 in IN_STATES or c2 in _STATE_CODES or c2 in _FR_ADMIN or c2.replace(" ", "") in IN_STATES or skeleton(c2).replace(" ", "") in _STATE_SKEL:
            continue
        for t in c2.split():
            t = fix_leet(t) if not any(ch.isdigit() for ch in t[:1]) else t
            t = ORDINALS.get(t, t)
            t = ADDR_CANON.get(t, t)
            if t in ADDR_STOP:
                continue
            toks.append(t)
    nums = []
    for n in _NUM_RE.findall(" ".join(toks)):
        n2 = n.lstrip("0") or "0"
        nums.append(n2)
    # first number in the address is usually the house/door number
    return {
        "a_norm": " ".join(toks),
        "a_nums": " ".join(dict.fromkeys(nums)),
        "a_words": " ".join(t for t in toks if not any(ch.isdigit() for ch in t) and len(t) > 1),
    }


_STATE_SKEL = {skeleton(k).replace(" ", "") for k in list(US_STATES) + list(IN_STATES) if len(skeleton(k)) >= 3}


def normalize_record(name, addr):
    d = normalize_name(name or "")
    d.update(normalize_address(addr or ""))
    d["n_skel"] = skeleton(d["n_core"])
    return d
