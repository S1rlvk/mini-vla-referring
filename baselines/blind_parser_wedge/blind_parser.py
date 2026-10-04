"""Rule-based referring-expression parser for a tabletop scene. Stdlib only.

select(text, objects) -> id or None.
Image coords: x right, y DOWN. behind = smaller y, in front of = larger y.
"""
import re

COLORS = ["red", "green", "blue", "yellow", "purple"]
SHAPES = ["circle", "square", "triangle"]
SIZES = ["small", "large"]
THRESH = 0.06

_SYN = {
    "violet": "purple", "magenta": "purple", "lilac": "purple",
    "crimson": "red", "scarlet": "red", "lime": "green", "gold": "yellow",
    "golden": "yellow", "grn": "green",
    "circles": "circle", "circular": "circle", "round": "circle", "ball": "circle",
    "balls": "circle", "disc": "circle", "disk": "circle", "dot": "circle",
    "squares": "square", "box": "square", "boxes": "square", "cube": "square",
    "cubes": "square", "block": "square", "blocks": "square",
    "triangles": "triangle", "triangular": "triangle", "pyramid": "triangle", "wedge": "triangle", "wedges": "triangle",
    "little": "small", "tiny": "small", "mini": "small", "smaller": "small",
    "big": "large", "huge": "large", "giant": "large", "bigger": "large",
    "larger": "large",
}
_COL = "(?:%s)" % "|".join(COLORS)
_NEG = re.compile(
    r"\b(?:not|non|isnt|arent|never|other than|except|excluding|but not|anything but|"
    r"any colou?r but|but|besides|neither|nor)\s+(?:(?:an?|the|colou?red|colou?r|of|being)\s+)*"
    r"(" + _COL + r"(?:\s*(?:,|or|nor)\s*" + _COL + r")*)\b")

_REL = re.compile(
    r"\b(?:"
    r"(?P<between>in the middle of|in between|midway between|halfway between|between|amid|flanked by|sandwiched between)"
    r"|(?P<left>(?:(?:to|on|at|towards) the )?(?:far )?(?:left|leftward|leftwards|west)(?: hand)?(?: side)?\s+(?:of|from|to)"
    r"|(?:further|farther|more) (?:to the )?left (?:than|of))"
    r"|(?P<right>(?:(?:to|on|at|towards) the )?(?:far )?(?:right|rightward|rightwards|east)(?: hand)?(?: side)?\s+(?:of|from|to)"
    r"|(?:further|farther|more) (?:to the )?right (?:than|of))"
    r"|(?P<behind>(?:in )?behind|above|(?:in|at) (?:the )?back of|beyond|north of|higher than"
    r"|(?:further|farther) (?:back|away|up)(?: from (?:the )?(?:camera|viewer|front))? than)"
    r"|(?P<front>(?:in )?front of|below|beneath|underneath|under|south of|lower than)"
    r")\b")

_EDGES = [
    ("top", re.compile(r"\b(?:top|topmost|uppermost|upper|highest)\b")),
    ("bottom", re.compile(r"\b(?:bottom|bottommost|lowermost|lower|lowest)\b")),
    ("left", re.compile(r"\bleftmost\b|\bleft (?:edge|border|boundary)\b|\bfar left\b")),
    ("right", re.compile(r"\brightmost\b|\bright (?:edge|border|boundary)\b|\bfar right\b")),
]
_FILL = {"the", "a", "an", "just", "directly", "immediately", "only", "and", ",", "it", "is"}
_IMAGE = {"image", "table", "scene", "picture", "frame", "screen", "photo", "view"}
_VERBS = re.compile(r",|\b(?:is|are|pick|select|choose|take|grab|find|get|touch|point|click|move|push|lift|identify|which|lies|sits|stands)\b")


def _norm(text):
    t = text.lower().replace("'", "").replace("’", "")
    t = t.replace("-", " ").replace("&", " and ")
    t = re.sub(r"[^a-z0-9,]+", " ", t)
    t = t.replace(",", " , ")
    t = " ".join(_SYN.get(w, w) for w in t.split())
    t = t.replace("on top of", "above")
    # "X with Y to its left" -> "X right of Y"
    m = re.search(r"\b(?:with|that has|which has|having|has)\s+(.*?)\s+(?:to|on|at) its (left|right)(?: side)?\b", t)
    if m:
        inv = "right" if m.group(2) == "left" else "left"
        t = t[:m.start()] + " %s of %s " % (inv, m.group(1)) + t[m.end():]
    return " " + t + " "


class _Spec:
    def __init__(self, text):
        self.colors, self.shapes, self.sizes, self.neg, self.edge = set(), set(), set(), set(), None
        for m in _NEG.finditer(text):
            self.neg |= set(re.findall(_COL, m.group(1)))
        text = _NEG.sub(" ", text)
        for name, rx in _EDGES:
            if rx.search(text):
                self.edge = name
                text = rx.sub(" ", text)
                break
        for w in text.split():
            if w in COLORS:
                self.colors.add(w)
            elif w in SHAPES:
                self.shapes.add(w)
            elif w in SIZES:
                self.sizes.add(w)
        self.words = set(text.split())

    @property
    def has(self):
        return bool(self.colors or self.shapes or self.sizes or self.neg or self.edge)

    def match(self, o):
        return ((not self.colors or o["color"] in self.colors)
                and (not self.shapes or o["shape"] in self.shapes)
                and (not self.sizes or o["size"] in self.sizes)
                and o["color"] not in self.neg)


def _extreme(objs, edge):
    if not objs:
        return []
    key = {"top": lambda o: o["y"], "bottom": lambda o: -o["y"],
           "left": lambda o: o["x"], "right": lambda o: -o["x"]}[edge]
    best = min(key(o) for o in objs)
    return [o for o in objs if abs(key(o) - best) < 1e-9]


def _resolve(spec, objs):
    c = [o for o in objs if spec.match(o)]
    return _extreme(c, spec.edge) if spec.edge else c


def _rel_ok(kind, c, a):
    if kind == "left":
        return a["x"] - c["x"] > THRESH
    if kind == "right":
        return c["x"] - a["x"] > THRESH
    if kind == "behind":
        return a["y"] - c["y"] > THRESH
    return c["y"] - a["y"] > THRESH


def _between_ok(c, a, b):
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    L2 = dx * dx + dy * dy
    if L2 <= 0:
        return False
    px, py = c["x"] - a["x"], c["y"] - a["y"]
    t = (px * dx + py * dy) / L2
    perp = abs(px * dy - py * dx) / (L2 ** 0.5)
    return 0.15 < t < 0.85 and perp < 0.10


def _parts(text):
    out = []
    for p in re.split(r"\band\b", text.replace(",", " ")):
        s = _Spec(p)
        if s.has:
            out.append(s)
    return out


def _solve(text, objs):
    t = _norm(text)
    ms = list(_REL.finditer(t))
    if not ms:
        head, segs = t, []
    else:
        head = t[:ms[0].start()]
        segs = []
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
            segs.append([m.lastgroup, t[m.end():end]])
        if set(head.split()) <= _FILL:  # sentence starts with the relation
            s = _VERBS.search(segs[0][1])
            if s:
                head, segs[0][1] = segs[0][1][s.end():], segs[0][1][:s.start()]
            else:
                head = ""
        for sg in segs:
            if "," in sg[1]:
                a, extra = sg[1].split(",", 1)
                sg[1] = a
                head += " " + extra
    hs = _Spec(head)
    cands = [o for o in objs if hs.match(o)]
    for kind, txt in segs:
        parts = _parts(txt)
        if kind in ("left", "right") and not parts and set(txt.split()) & _IMAGE:
            hs.edge = kind
            continue
        if not parts:
            continue
        if kind == "between":
            if len(parts) < 2:
                return None
            A, B = _resolve(parts[0], objs), _resolve(parts[1], objs)
            cands = [c for c in cands
                     if any(_between_ok(c, a, b) for a in A for b in B
                            if a is not c and b is not c and a is not b)]
        else:
            for p in parts:
                A = _resolve(p, objs)
                cands = [c for c in cands if any(_rel_ok(kind, c, a) for a in A if a is not c)]
    if hs.edge:
        cands = _extreme(cands, hs.edge)
    return cands[0]["id"] if len(cands) == 1 else None


def select(text, objects):
    try:
        objs = [dict(o, x=float(o["x"]), y=float(o["y"])) for o in objects]
        return _solve(str(text), objs)
    except Exception:
        return None
