# all task instructions live here — import this everywhere
TASK_INSTRUCTIONS = {
    "reach-v3":                  "reach the target",
    "push-v3":                   "push the puck to the goal",
    "pick-place-v3":             "pick up the object and place it at the goal",
    "door-open-v3":              "open the door",
    "drawer-close-v3":           "close the drawer",
    "drawer-open-v3":            "open the drawer",
    "button-press-topdown-v3":   "press the button",
    "peg-insert-side-v3":        "insert the peg into the hole",
    "window-open-v3":            "open the window",
    "window-close-v3":           "close the window",
}

# difficulty tiers — useful for curriculum or weighted sampling
TASK_DIFFICULTY = {
    "reach-v3":                  "easy",
    "drawer-close-v3":           "easy",
    "button-press-topdown-v3":   "easy",
    "push-v3":                   "medium",
    "door-open-v3":              "medium",
    "window-open-v3":            "medium",
    "window-close-v3":           "medium",
    "drawer-open-v3":            "medium",
    "pick-place-v3":             "hard",
    "peg-insert-side-v3":        "hard",
}