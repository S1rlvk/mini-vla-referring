from blind_parser import select

def O(i, c, s, z, x, y):
    return {"id": i, "color": c, "shape": s, "size": z, "x": x, "y": y}

S = [O(0, "red", "circle", "small", .2, .5), O(1, "blue", "square", "large", .8, .5),
     O(2, "green", "triangle", "small", .5, .52), O(3, "yellow", "circle", "large", .5, .9)]
CASES = [
    ("the red circle", 0), ("the large circle", 3), ("small green triangle", 2),
    ("the circle that is not red", 3), ("the shape that isn't red or blue or green", 3),
    ("the object to the left of the blue square, not green or yellow", 0),
    ("the thing right of the red circle and above the yellow circle, not blue", 2),
    ("the shape between the red circle and the blue square", 2),
    ("Between the red circle and the blue square is the green triangle", 2),
    ("the circle in front of the green triangle", 3),
    ("the object that is not red or blue, behind the yellow circle", 2),
    ("the circle near the top edge", 0 ) if False else ("the large shape at the bottom", 3),
    ("To the left of the blue square, pick the red circle", 0),
    ("the circle with the blue square to its right and the green triangle to its left", 0), ("the small circle with the blue square to its right", 0),
    ("gibberish", None),
]
bad = 0
for s, e in CASES:
    r = select(s, S)
    if r != e:
        bad += 1
        print("FAIL", s, r, e)
print("failures:", bad, "of", len(CASES))
assert select(None, S) is None and select("x", None) is None
