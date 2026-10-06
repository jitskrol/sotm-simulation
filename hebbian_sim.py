#!/usr/bin/env python3
"""
Hebbian association network — a 2D simulation.

Concepts are nodes. Every ordered pair of concepts (A, B) has two association
strengths: forward A→B and backward B→A. Through Hebbian learning each strength
drifts toward the experienced conditional correlation:

    w[A→B]  →  P(B active | A active)

so "coffee → cup" and "cup → coffee" can end up with different strengths.

Each tick every concept:
  1. receives input from every other concept:  max(0, activity(A) × w[A→B] − resistance)
     plus sensory input from the current experience, clicks and a little noise,
  2. combines all input, scales it by its sensitivity and settles toward a new
     activity level (0 … 1),
  3. tires when active for long (fatigue), which lets thoughts wander.

Homeostasis: the mean activity of the whole network is compared with the
setpoint. Above it, all concepts instantly become less sensitive; below it,
their sensitivity slowly creeps up.

Emotions: the mood has three opposing axes (sadness ↔ happiness, fear ↔ calm,
anger ↔ affection), each from −1 to +1. Most concepts are tied to one or more
emotions with a graded weight (crying: strongly sad; computer: slightly
fearful; campfire: happy, calm and a little affectionate). Active concepts push
the mood toward their poles ("reactivity"), and the mood relaxes back to
neutral on its own ("half-life"). The mood in turn makes fitting concepts more
sensitive and concepts of the opposite pole less sensitive ("emotionality"), so
it steers where activation flows.

STDP: when concept B starts firing shortly after concept A was active, the
outgoing A→B strengthens and the incoming B→A weakens. Routines (sequences of
scenes such as running → shower → getting dressed) give it order to learn, so
free association starts to run forward through them.

While an experience is happening, associative spread is muted ("attention to
the world"), so learning reflects what was actually experienced rather than
what the network imagined alongside it.

Run:  ./hebbian_sim.py      (needs python3, numpy and GTK4 via PyGObject)
"""

import math
import os
import random
import time
from collections import deque

import cairo
import numpy as np
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gtk, Gdk, Gio, GLib  # noqa: E402

# ---------------------------------------------------------------------------
# The "real world": recurring situations in which concepts are experienced
# together. (name, relative frequency, concepts)
# ---------------------------------------------------------------------------
SCENES = [
    # home and daily life
    ("breakfast",         5, ["morning", "coffee", "bread", "kitchen", "cup", "sun", "toast"]),
    ("waking up",         4, ["alarm", "bed", "morning", "tired", "shower", "coffee"]),
    ("cooking dinner",    3, ["kitchen", "stove", "pan", "onion", "knife", "smell", "evening"]),
    ("family dinner",     3, ["mother", "father", "home", "table", "talk", "child", "soup"]),
    ("doing the dishes",  2, ["kitchen", "sink", "water", "plate", "tired", "evening"]),
    ("cleaning",          2, ["home", "broom", "dust", "window", "music", "saturday"]),
    ("laundry",           1, ["laundry", "soap", "water", "shirt", "sunday"]),
    ("quiet night",       3, ["home", "evening", "book", "tea", "cup", "fireplace", "blanket"]),
    ("hot bath",          1, ["bath", "water", "soap", "candle", "quiet", "towel"]),
    ("bedtime story",     3, ["bed", "child", "book", "night", "mother", "blanket", "dream"]),
    ("grocery shopping",  3, ["supermarket", "bread", "cheese", "milk", "apple", "queue", "money"]),
    ("paying bills",      2, ["money", "letter", "bills", "desk", "worry", "bank"]),
    ("moving house",      1, ["box", "home", "truck", "friend", "tired", "key"]),
    ("lost keys",         1, ["key", "door", "pocket", "rush", "late", "morning"]),
    ("cat at home",       2, ["cat", "sofa", "sun", "sleep", "window", "quiet"]),
    ("reading a novel",   2, ["book", "sofa", "rain", "tea", "story", "quiet"]),
    ("writing",           2, ["pen", "notebook", "idea", "desk", "coffee", "window", "quiet"]),
    ("newborn",           1, ["baby", "crying", "night", "tired", "milk", "smile", "mother"]),
    ("sick in bed",       2, ["fever", "bed", "tea", "blanket", "tired", "soup", "pain"]),
    # work
    ("office",            4, ["desk", "computer", "coffee", "deadline", "stress", "colleague", "email"]),
    ("meeting",           3, ["colleague", "boss", "table", "talk", "presentation", "coffee"]),
    ("deadline crunch",   2, ["deadline", "night", "computer", "stress", "boss", "pizza"]),
    ("lunch break",       3, ["colleague", "bread", "cheese", "talk", "table", "coffee", "sandwich"]),
    ("job interview",     1, ["suit", "boss", "nervous", "question", "handshake", "smile"]),
    ("performance review", 1, ["boss", "feedback", "praise", "desk", "nervous"]),
    ("being laid off",    1, ["boss", "letter", "money", "worry", "box", "silence"]),
    ("promotion",         1, ["boss", "praise", "champagne", "money", "colleague", "success"]),
    ("unfair blame",      1, ["boss", "lie", "shouting", "colleague", "injustice"]),
    ("commute",           4, ["train", "street", "morning", "rain", "phone", "crowd", "late"]),
    ("traffic jam",       2, ["car", "road", "radio", "late", "rush", "horn"]),
    ("video call",        2, ["computer", "phone", "friend", "talk", "evening", "screen"]),
    ("inbox",             3, ["email", "computer", "screen", "message", "coffee"]),
    ("broken computer",   1, ["computer", "error", "screen", "broken", "deadline"]),
    # school and learning
    ("school",            2, ["child", "book", "desk", "teacher", "learn", "pencil", "classroom"]),
    ("exam",              1, ["exam", "nervous", "pencil", "question", "teacher", "silence", "classroom"]),
    ("studying",          2, ["book", "desk", "learn", "tea", "stress", "deadline", "library"]),
    ("library",           1, ["library", "book", "silence", "quiet", "rain", "window"]),
    ("schoolyard",        2, ["child", "ball", "friend", "laughter", "bell", "bully"]),
    ("bullied",           1, ["bully", "child", "crying", "alone", "bell", "shouting"]),
    ("music lesson",      1, ["piano", "song", "learn", "teacher", "child", "practice"]),
    ("graduation",        1, ["diploma", "mother", "father", "success", "photo", "champagne", "speech"]),
    # nature and weather
    ("forest",            2, ["tree", "green", "bird", "walk", "autumn", "leaf", "moss"]),
    ("autumn walk",       2, ["leaf", "autumn", "walk", "rain", "tree", "cold", "mushroom"]),
    ("garden",            2, ["tree", "green", "flower", "sun", "bird", "summer", "bee"]),
    ("gardening",         2, ["soil", "seed", "flower", "spade", "sun", "grandmother"]),
    ("mountain hike",     1, ["mountain", "path", "view", "wind", "backpack", "tired", "sun"]),
    ("thunderstorm",      2, ["storm", "thunder", "lightning", "rain", "dark", "window", "dog"]),
    ("rainy day",         3, ["rain", "cloud", "umbrella", "grey", "autumn", "street", "puddle"]),
    ("snow day",          2, ["snow", "cold", "winter", "child", "sled", "laughter", "scarf"]),
    ("winter evening",    2, ["snow", "cold", "winter", "grey", "home", "fireplace", "scarf"]),
    ("lake at dawn",      1, ["lake", "mist", "morning", "bird", "quiet", "boat"]),
    ("night sky",         1, ["night", "stars", "moon", "quiet", "cold", "dream"]),
    ("spring",            2, ["spring", "flower", "bird", "sun", "green", "bee", "blossom"]),
    ("heatwave",          1, ["summer", "sun", "sweat", "ice cream", "shade", "tired"]),
    # sea and travel
    ("beach",             2, ["sun", "sea", "sand", "summer", "holiday", "swim", "shell"]),
    ("morning swim",      1, ["swim", "cold", "morning", "sea", "laughter"]),
    ("summer trip",       1, ["train", "holiday", "sea", "suitcase", "summer", "phone", "map"]),
    ("airport",           1, ["airport", "plane", "suitcase", "queue", "passport", "late"]),
    ("flight",            1, ["plane", "window", "cloud", "nervous", "turbulence", "passport"]),
    ("foreign city",      1, ["map", "street", "stranger", "language", "photo", "holiday"]),
    ("camping",           1, ["tent", "campfire", "stars", "night", "friend", "mosquito", "tree"]),
    ("sailing",           1, ["boat", "wind", "sea", "sun", "rope", "laughter"]),
    ("storm at sea",      1, ["boat", "storm", "wind", "sea", "dark", "rope"]),
    # town and friends
    ("market",            2, ["street", "crowd", "cheese", "flower", "bread", "talk", "apple"]),
    ("café",              3, ["coffee", "cup", "friend", "talk", "cake", "window"]),
    ("concert",           1, ["piano", "song", "evening", "friend", "crowd", "stage", "applause"]),
    ("dinner party",      2, ["friend", "wine", "table", "talk", "evening", "cheese", "laughter"]),
    ("party",             2, ["music", "dance", "friend", "wine", "crowd", "laughter", "night"]),
    ("bar",               2, ["beer", "friend", "laughter", "music", "night", "talk"]),
    ("festival",          1, ["music", "stage", "crowd", "summer", "dance", "beer", "sun"]),
    ("birthday",          2, ["cake", "candle", "gift", "child", "song", "laughter", "balloon"]),
    ("christmas",         1, ["christmas", "tree", "candle", "gift", "mother", "father", "snow", "song"]),
    ("new year",          1, ["fireworks", "champagne", "night", "kiss", "crowd", "clock"]),
    ("barbecue",          1, ["grill", "sausage", "beer", "summer", "friend", "smoke"]),
    ("cinema",            1, ["film", "popcorn", "dark", "screen", "friend", "laughter"]),
    ("museum",            1, ["painting", "museum", "silence", "art", "stranger"]),
    ("painting at home",  1, ["painting", "brush", "colour", "window", "quiet", "art"]),
    # love and family
    ("wedding",           1, ["wedding", "dress", "kiss", "champagne", "dance", "mother", "speech"]),
    ("date",              2, ["partner", "restaurant", "wine", "candle", "kiss", "evening", "nervous"]),
    ("sofa evening",      2, ["partner", "sofa", "blanket", "film", "evening", "kiss"]),
    ("argument",          2, ["partner", "shouting", "door", "slam", "silence", "kitchen"]),
    ("breakup",           1, ["partner", "tears", "alone", "message", "phone", "night", "rain"]),
    ("visiting grandma",  2, ["grandmother", "cake", "tea", "photo", "story", "clock"]),
    ("old photos",        1, ["photo", "grandmother", "childhood", "attic", "dust", "story"]),
    ("fishing with dad",  1, ["father", "lake", "boat", "quiet", "fish", "morning"]),
    ("learning to cycle", 1, ["father", "bike", "child", "fall", "tears", "praise"]),
    ("fixing the bike",   1, ["bike", "tool", "garage", "oil", "father"]),
    ("dog walk",          3, ["dog", "walk", "park", "green", "friend", "morning", "ball"]),
    ("playground",        2, ["child", "park", "sun", "laughter", "dog", "swing"]),
    # body and sport
    ("football match",    2, ["ball", "stadium", "crowd", "shouting", "goal", "friend"]),
    ("football practice", 1, ["ball", "grass", "sweat", "coach", "child", "rain"]),
    ("victory",           1, ["goal", "success", "applause", "crowd", "champagne"]),
    ("gym",               2, ["gym", "sweat", "music", "tired", "mirror", "shower"]),
    ("running",           2, ["running", "park", "morning", "sweat", "music", "breath"]),
    ("bike ride",         2, ["bike", "road", "wind", "sun", "park", "helmet"]),
    ("yoga",              1, ["yoga", "breath", "quiet", "mat", "candle", "morning"]),
    ("meditation",        1, ["breath", "silence", "candle", "quiet", "mat"]),
    # illness, danger and loss
    ("hospital visit",    1, ["mother", "worry", "doctor", "phone", "grey", "hospital"]),
    ("grandma in hospital", 1, ["grandmother", "hospital", "doctor", "worry", "flower", "clock"]),
    ("funeral",           1, ["funeral", "grandmother", "tears", "black", "flower", "rain", "silence"]),
    ("dentist",           1, ["dentist", "pain", "nervous", "waiting", "magazine"]),
    ("accident",          1, ["car", "road", "siren", "ambulance", "blood", "pain", "hospital"]),
    ("fire alarm",        1, ["smoke", "alarm", "stairs", "siren", "crowd", "running"]),
    ("dark street",       1, ["dark", "street", "stranger", "footsteps", "night", "running"]),
    ("burglary",          1, ["door", "window", "police", "stranger", "broken", "night"]),
    ("nightmare",         1, ["nightmare", "dark", "running", "night", "footsteps", "bed"]),
    ("basement",          1, ["spider", "basement", "dark", "dust", "stairs"]),
    ("morning news",      3, ["news", "screen", "politics", "worry", "coffee", "morning"]),
    ("war report",        1, ["news", "war", "smoke", "siren", "worry", "tears"]),
    ("sleepless night",   2, ["night", "worry", "bed", "stress", "deadline", "phone", "clock"]),
    ("bureaucracy",       1, ["queue", "form", "waiting", "clock", "letter"]),
    ("phone scam",        1, ["stranger", "phone", "money", "bank", "lie"]),
    # steps of daily routines (see ROUTINES)
    ("shower",            4, ["shower", "water", "soap", "towel", "steam", "mirror"]),
    ("getting dressed",   4, ["shirt", "mirror", "shoes", "wardrobe", "morning", "suit"]),
    ("commute home",      3, ["train", "evening", "tired", "crowd", "phone", "street"]),
    ("brushing teeth",    3, ["toothbrush", "mirror", "water", "night", "tired"]),
    ("going to bed",      3, ["bed", "night", "blanket", "alarm", "sleep", "dark"]),
    ("bath time",         2, ["child", "bath", "water", "soap", "towel", "laughter"]),
    ("homework",          2, ["child", "pencil", "desk", "book", "tired", "question"]),
    ("packing",           1, ["suitcase", "shirt", "passport", "map", "shoes", "rush"]),
    ("hotel room",        1, ["hotel", "bed", "window", "view", "towel", "tired"]),
    ("sunburn",           1, ["sun", "pain", "summer", "shade", "sea", "water"]),
    ("sunset walk",       1, ["sunset", "sea", "sand", "partner", "quiet", "sun"]),
    ("party preparation", 1, ["balloon", "cake", "kitchen", "child", "rush", "gift"]),
    ("coming home",       2, ["key", "door", "home", "evening", "tired", "silence"]),
    ("christmas shopping", 1, ["crowd", "gift", "queue", "money", "street", "cold"]),
    ("taxi home",         1, ["car", "night", "street", "tired", "stranger", "phone"]),
    ("hangover",          1, ["bed", "pain", "tired", "water", "coffee", "silence"]),
    ("late for work",     1, ["late", "rush", "boss", "desk", "email", "nervous"]),
    # emotionally charged experiences
    ("ambulance ride",    1, ["ambulance", "siren", "blood", "pain", "heartbeat", "stranger"]),
    ("emergency room",    1, ["hospital", "doctor", "waiting", "worry", "clock", "blood"]),
    ("bad news call",     1, ["phone", "tears", "silence", "worry", "night", "mother"]),
    ("empty house",       1, ["empty chair", "silence", "alone", "photo", "home", "tears"]),
    ("feeling ill",       2, ["fever", "tired", "bed", "pain", "tea", "worry"]),
    ("doctor's visit",    2, ["doctor", "waiting", "magazine", "worry", "question", "nervous"]),
    ("diagnosis",         1, ["doctor", "diagnosis", "silence", "worry", "tears", "partner"]),
    ("recovery",          1, ["hospital", "flower", "smile", "sun", "friend", "walk"]),
    ("silent treatment",  1, ["partner", "silence", "door", "alone", "kitchen", "phone"]),
    ("apology",           1, ["partner", "apology", "tears", "hug", "talk"]),
    ("making up",         1, ["partner", "hug", "kiss", "laughter", "smile", "evening"]),
    ("crying alone",      2, ["tears", "alone", "bed", "night", "crying", "rain"]),
    ("calling a friend",  2, ["phone", "friend", "talk", "tears", "night", "tea"]),
    ("night before the exam", 1, ["book", "nervous", "night", "coffee", "heartbeat", "desk"]),
    ("passed exam",       1, ["exam", "success", "praise", "smile", "laughter", "friend"]),
    ("failed exam",       1, ["exam", "letter", "tears", "alone", "silence", "teacher"]),
    ("celebration",       2, ["champagne", "friend", "laughter", "dance", "music", "success"]),
    ("job application",   1, ["computer", "letter", "email", "question", "nervous", "coffee"]),
    ("waiting for the call", 1, ["phone", "clock", "nervous", "waiting", "heartbeat", "tea"]),
    ("getting the job",   1, ["phone", "success", "smile", "laughter", "partner", "champagne"]),
    ("rejection letter",  1, ["email", "letter", "silence", "tears", "alone"]),
    ("telling your partner", 1, ["partner", "kitchen", "tears", "hug", "worry", "money"]),
    ("walking home at night", 2, ["street", "night", "dark", "moon", "footsteps", "phone"]),
    ("footsteps behind",  1, ["footsteps", "stranger", "dark", "heartbeat", "street", "running"]),
    ("running home",      1, ["running", "heartbeat", "key", "door", "breath", "street"]),
    ("locking the door",  1, ["door", "key", "heartbeat", "silence", "phone", "window"]),
    ("power outage",      1, ["dark", "candle", "silence", "storm", "phone", "cold"]),
    ("candlelight",       1, ["candle", "partner", "quiet", "blanket", "laughter", "dark"]),
    ("defeat",            1, ["ball", "stadium", "crowd", "grey", "silence", "rain"]),
    ("first kiss",        1, ["kiss", "partner", "heartbeat", "night", "smile", "nervous"]),
    ("texting",           2, ["phone", "message", "smile", "partner", "bed", "night"]),
    ("pregnancy test",    1, ["partner", "toothbrush", "heartbeat", "smile", "tears", "hug"]),
    ("hospital birth",    1, ["hospital", "baby", "tears", "partner", "doctor", "smile", "pain"]),
    ("sleepless baby night", 1, ["baby", "crying", "night", "tired", "milk", "partner"]),
    ("painful evening",   1, ["pain", "tea", "blanket", "sofa", "tired"]),
    ("police report",     1, ["police", "question", "form", "stranger", "broken", "worry"]),
    ("talk with mother",  1, ["mother", "hug", "tears", "kitchen", "tea", "talk"]),
    ("lost child",        1, ["park", "child", "crowd", "heartbeat", "shouting", "police"]),
    ("found again",       1, ["child", "hug", "tears", "mother", "smile", "police"]),
    ("sick dog",          1, ["dog", "worry", "blanket", "tired", "silence"]),
    ("vet",               1, ["vet", "dog", "worry", "waiting", "tears"]),
    ("burying the dog",   1, ["dog", "spade", "soil", "tears", "child", "silence"]),
    ("wedding party",     1, ["wedding", "dance", "champagne", "music", "laughter", "speech", "cake"]),
    ("honeymoon",         1, ["partner", "sea", "sun", "holiday", "hotel", "kiss"]),
    ("reunion",           1, ["airport", "hug", "tears", "smile", "mother", "laughter"]),
    ("surprise party",    1, ["friend", "surprise", "laughter", "cake", "balloon", "gift"]),
    ("panic attack",      1, ["heartbeat", "breath", "crowd", "nervous", "alone", "train"]),
    ("giving a speech",   1, ["speech", "crowd", "nervous", "heartbeat", "applause", "stage"]),
    ("road rage",         1, ["car", "horn", "shouting", "fist", "late", "road"]),
    ("cheated on",        1, ["partner", "lie", "message", "phone", "tears", "shouting"]),
    ("comforting a friend", 2, ["friend", "hug", "tears", "tea", "talk", "sofa"]),
    ("first steps",       1, ["baby", "laughter", "smile", "mother", "father", "applause"]),
    ("stargazing",        1, ["stars", "partner", "night", "quiet", "kiss", "cold"]),
    ("night in hospital", 1, ["hospital", "silence", "clock", "worry", "heartbeat", "doctor"]),
    ("scary film",        1, ["film", "dark", "scream", "popcorn", "partner", "laughter"]),
    ("alone at night",    1, ["creak", "dark", "alone", "night", "heartbeat", "stairs"]),
    ("missed train",      1, ["train", "late", "rush", "shouting", "queue", "rain"]),
    ("anniversary",       1, ["partner", "restaurant", "gift", "kiss", "candle", "wine"]),
    ("betrayed by colleague", 1, ["colleague", "lie", "boss", "injustice", "email", "shouting"]),
    ("protest",           1, ["crowd", "politics", "shouting", "street", "police", "injustice"]),
    ("bee sting",         1, ["bee", "pain", "child", "crying", "summer", "flower"]),
    ("dog bite",          1, ["dog", "stranger", "blood", "pain", "crying", "park"]),
    ("helping a stranger", 1, ["stranger", "smile", "street", "praise", "rain", "umbrella"]),
    ("lonely evening",    2, ["alone", "silence", "phone", "rain", "window", "tea"]),
    ("home from school",  2, ["child", "mother", "hug", "cake", "smile", "story"]),
    ("playing with the cat", 1, ["cat", "laughter", "sofa", "sun", "child"]),
    ("yelled at by the boss", 1, ["boss", "shouting", "desk", "nervous", "tears", "injustice"]),
    ("overdue bills",     1, ["bills", "letter", "money", "worry", "bank", "phone"]),
    ("insulted",          1, ["stranger", "insult", "shouting", "fist", "street", "tears"]),
    ("misty morning",     1, ["mist", "morning", "walk", "dog", "moss", "quiet"]),
    ("steamy kitchen",    1, ["steam", "soup", "kitchen", "window", "rain", "tea"]),
    ("teenage tantrum",   1, ["child", "slam", "door", "shouting", "tears", "mother"]),
    ("summer fireworks",  1, ["fireworks", "summer", "crowd", "night", "ice cream", "laughter"]),
    ("computer crash",    1, ["computer", "error", "deadline", "shouting", "fist", "coffee"]),
    ("haunted by a dream", 1, ["nightmare", "bed", "heartbeat", "dark", "creak", "alone"]),
]

# Routines: scenes that usually follow one another in this order. When a scene
# that belongs to a routine is experienced, the next step follows (with a
# probability set by the "routine predictability" slider) after a short gap.
ROUTINES = [
    ("morning routine",   ["waking up", "shower", "getting dressed", "breakfast", "commute", "office"]),
    ("morning run",       ["running", "shower", "getting dressed", "breakfast"]),
    ("gym session",       ["gym", "shower", "getting dressed"]),
    ("workday",           ["office", "meeting", "lunch break", "inbox", "commute home"]),
    ("evening at home",   ["commute home", "grocery shopping", "cooking dinner", "family dinner",
                           "doing the dishes", "quiet night", "brushing teeth", "going to bed"]),
    ("kids' bedtime",     ["family dinner", "bath time", "bedtime story", "sofa evening"]),
    ("school day",        ["waking up", "breakfast", "school", "schoolyard", "home from school", "homework"]),
    ("crunch",            ["deadline crunch", "sleepless night", "waking up", "late for work"]),
    ("hosting dinner",    ["grocery shopping", "cooking dinner", "dinner party", "doing the dishes"]),
    ("trip abroad",       ["packing", "airport", "flight", "foreign city", "hotel room"]),
    ("beach day",         ["summer trip", "beach", "sunburn", "sunset walk"]),
    ("car accident",      ["traffic jam", "accident", "ambulance ride", "emergency room", "bad news call"]),
    ("losing grandma",    ["grandma in hospital", "bad news call", "funeral", "old photos", "empty house"]),
    ("illness",           ["feeling ill", "doctor's visit", "diagnosis", "night in hospital", "recovery"]),
    ("fight and make up", ["argument", "silent treatment", "apology", "making up", "sofa evening"]),
    ("breaking up",       ["argument", "breakup", "crying alone", "calling a friend", "café"]),
    ("exam season",       ["studying", "night before the exam", "exam", "passed exam", "celebration"]),
    ("failing",           ["exam", "failed exam", "crying alone", "talk with mother"]),
    ("job hunt",          ["job application", "job interview", "waiting for the call", "getting the job", "celebration"]),
    ("rejection",         ["job interview", "waiting for the call", "rejection letter", "lonely evening"]),
    ("laid off",          ["being laid off", "telling your partner", "overdue bills", "job application"]),
    ("followed at night", ["walking home at night", "footsteps behind", "running home", "locking the door"]),
    ("stormy night",      ["thunderstorm", "power outage", "candlelight", "going to bed"]),
    ("match day",         ["football practice", "football match", "victory", "bar"]),
    ("losing the match",  ["football match", "defeat", "bar"]),
    ("first date",        ["getting dressed", "date", "first kiss", "walking home at night", "texting"]),
    ("new baby",          ["pregnancy test", "hospital birth", "newborn", "sleepless baby night", "first steps"]),
    ("birthday",          ["party preparation", "birthday", "cleaning"]),
    ("late again",        ["lost keys", "traffic jam", "late for work", "yelled at by the boss"]),
    ("toothache",         ["dentist", "painful evening"]),
    ("break-in",          ["coming home", "burglary", "police report", "nightmare"]),
    ("holidays",          ["christmas shopping", "christmas", "new year"]),
    ("gardening year",    ["spring", "gardening", "garden", "autumn walk"]),
    ("bullying",          ["schoolyard", "bullied", "crying alone", "talk with mother"]),
    ("night out",         ["getting dressed", "bar", "party", "taxi home", "hangover"]),
    ("worrying news",     ["morning news", "war report", "sleepless night"]),
    ("lost in the park",  ["playground", "lost child", "found again"]),
    ("losing the dog",    ["sick dog", "vet", "burying the dog", "empty house"]),
    ("wedding day",       ["getting dressed", "wedding", "wedding party", "honeymoon"]),
]

# Emotions are three opposing axes, each running from −1 to +1.
# (negative pole, colour, positive pole, colour)
AXES = [
    ("sadness", "#5b9bf0", "happiness", "#f2c14e"),
    ("fear",    "#a07cff", "calm",      "#4ecdc4"),
    ("anger",   "#ef5350", "affection", "#ff6fa5"),
]
# pole code → (axis, sign)
POLES = {"S": (0, -1), "H": (0, +1), "F": (1, -1), "C": (1, +1), "R": (2, -1), "A": (2, +1)}
POLE_NAMES = {"S": "sad", "H": "happy", "F": "fear", "C": "calm", "R": "anger", "A": "affection"}

# How strongly each concept is tied to each emotion (0.1 … 1).
# H happiness  S sadness  C calm  F fear/stress  A affection  R anger
EMOTION_WEIGHTS = """
morning C.3 H.2 | coffee H.2 C.2 | bread C.2 H.1 | kitchen C.2 A.2 | cup C.3 | sun H.8 C.3
toast H.1 | alarm F.3 R.2 | bed C.5 | tired S.3 | shower C.4 | stove A.1 | knife F.3 | smell H.2
evening C.4 | mother A1 C.2 | father A.8 | home C.6 A.4 | table A.2 | talk A.3 | child A.6 H.4
soup C.3 A.2 | water C.3 | dust S.1 | window C.2 | music H.6 C.2 | saturday H.4 C.2
soap C.2 | sunday C.4 H.2 | book C.6 | tea C.8 | fireplace C.7 A.3 | blanket C.8 A.2 | bath C.9
candle C.6 A.2 | quiet C1 | towel C.3 | night F.3 C.2 | dream C.3 H.2 | cheese H.1 | milk C.1
apple H.1 | queue R.4 | money F.3 | letter F.2 | bills F.4 R.4 | desk F.2 | worry F.8 S.5
bank F.2 | box S.1 | friend A.8 H.5 | rush F.4 R.3 | late F.4 R.5 | cat C.5 A.5 | sofa C.6
sleep C1 | rain S.4 C.2 | story A.4 C.4 | notebook C.2 | idea H.4 | baby A1 H.4 | crying S1
smile H.9 A.5 | fever S.3 F.2 | pain S.5 F.4 | computer F.2 | deadline F.6 S.3 | stress F.8 R.3
colleague A.2 | email F.2 | boss F.4 R.3 | presentation F.5 | pizza H.3 | suit F.2 | nervous F1
question F.2 | handshake A.2 | feedback F.4 | praise H.9 A.3 | silence S.4 C.3 | champagne H.9
success H1 | lie R.8 S.3 | shouting R1 F.4 | injustice R1 S.3 | street F.1 | crowd F.3
radio H.2 | horn R.8 F.2 | message A.2 | error R.6 F.2 | broken R.5 S.4 | learn H.2
classroom F.2 | exam F.8 | library C.6 | ball H.4 | laughter H1 A.3 | bully R.8 F.7 S.4
alone S.9 F.2 | piano C.4 H.3 | song H.6 | diploma H.8 | photo A.3 S.2 | speech F.5 H.2
tree C.5 | green C.5 | bird C.5 H.3 | walk C.5 H.3 | autumn S.3 C.3 | leaf C.3 S.1 | moss C.8
cold S.3 F.1 | mushroom C.2 | flower H.6 C.3 | summer H.7 | bee H.2 F.2 | soil C.3 | seed H.3
grandmother A.9 S.3 C.3 | mountain C.5 H.4 | path C.4 | view H.6 C.6 | wind C.2 F.1
backpack H.2 | storm F.8 | thunder F.9 | lightning F.8 | dark F.9 | dog A.6 H.4 | cloud S.2
grey S.6 | puddle H.2 | snow C.4 H.4 | winter S.2 C.3 | sled H.7 | scarf C.3 | lake C.9
mist C.7 S.2 | boat C.4 H.3 | stars C.8 H.3 | moon C.6 | spring H.7 | blossom H.7 C.4
ice cream H.9 | shade C.4 | sea C.6 H.5 | sand H.4 C.3 | holiday H.9 C.4 | swim H.5 | shell H.3
suitcase H.2 | airport F.2 H.2 | plane F.3 | turbulence F1 | stranger F.8 | tent H.3
campfire H.6 C.5 A.3 | mosquito R.4 | cake H.8 | stage F.4 H.3 | applause H.9 | wine H.4 C.3
dance H.9 | beer H.5 | gift H.8 A.4 | balloon H.8 | christmas H.6 A.6 | fireworks H.9
kiss A1 H.5 | clock F.3 | grill H.3 | smoke F.5 | film H.3 | popcorn H.4 | painting C.5
museum C.5 | art C.4 H.3 | brush C.2 | colour H.4 | wedding A1 H.8 | dress H.3 | partner A1
restaurant H.4 A.3 | slam R1 | tears S1 | childhood A.4 S.3 H.3 | attic S.3 | fish C.3
bike H.4 | fall F.5 S.3 | park C.5 H.4 | swing H.6 | stadium H.4 | goal H1 | grass C.4
running F.2 H.2 | breath C1 | yoga C1 | mat C.7 | doctor F.5 | hospital F.6 S.6 | funeral S1
black S.7 | dentist F.7 | waiting F.3 R.2 | siren F1 | ambulance F1 | blood F1 | footsteps F.9
police F.5 | nightmare F1 | spider F.9 | basement F.6 | news F.3 | politics R.6 | war F1 S.7 R.4
form R.3 | steam C.5 | hotel H.4 | sunset C.9 H.6 | heartbeat F.9 | empty chair S1
diagnosis F.8 S.7 | apology A.6 S.3 | hug A1 H.5 | vet F.4 S.3 | surprise H.8 | fist R1
scream F1 | creak F.9 | insult R1 S.3
"""


def parse_emotion_weights(text):
    table = {}
    for entry in text.replace("\n", "|").split("|"):
        words = entry.split()
        if not words:
            continue
        codes = [w for w in words if w[0] in POLES and w[1:].replace(".", "").isdigit()]
        name = " ".join(w for w in words if w not in codes)
        table[name] = [(c[0], float(c[1:]) if c[1] != "." else float("0" + c[1:])) for c in codes]
    return table


EXPERIENCE_INPUT = 2.0   # sensory drive of concepts in an experience
EXPERIENCE_LEN = 25      # ticks an experience lasts
MEMBER_PROB = 0.85       # chance each concept of a scene is actually present
NOISE_PROB = 0.000       # chance per tick that a concept gets a spontaneous kick (was 0.003)
NOISE_KICK = 0.6         # size of that kick
THRESHOLD = 0.15         # input needed before a concept starts firing
ACT_RATE = 0.25          # how fast activity follows its input
FATIGUE_BUILD = 0.04     # how fast fatigue builds while active (was 0.02)
FATIGUE_RECOVER = 0.001  # how fast it recovers (was 0.003)
HOMEO_RATE = 3.0         # how fast sensitivity adapts (slow part) (was 0.005)
FAST_HOMEO = 5.0         # instant desensitisation when above the setpoint
SENS_MIN, SENS_MAX = 0.05, 40.0
EMO_GAIN = 2.5          # log-sensitivity change of a fully tied concept at full emotion
EMO_RISE = 0.008         # how fast an emotion builds while its concepts are active
EMO_SATURATE = 2.0       # this much tied activity drives an emotion at full speed
EDGE_MIN = 0.05          # associations weaker than this count as absent


class Network:
    def __init__(self):
        names = []
        for _, _, members in SCENES:
            for m in members:
                if m not in names:
                    names.append(m)
        self.names = names
        self.n = len(names)
        self.idx = {m: i for i, m in enumerate(names)}
        self.scenes = [(s, f, np.array([self.idx[m] for m in members]))
                       for s, f, members in SCENES]
        self.scene_freq = [f for _, f, _ in SCENES]
        scene_idx = {s: k for k, (s, _, _) in enumerate(SCENES)}
        self.routines = [(r, [scene_idx[s] for s in steps]) for r, steps in ROUTINES]
        self.scene_routines = {}             # scene → [(routine, position), …]
        for r, (_, steps) in enumerate(self.routines):
            for pos, k in enumerate(steps):
                self.scene_routines.setdefault(k, []).append((r, pos))

        # emotional ties: pos[k, i] / neg[k, i] = how strongly concept i belongs to
        # the positive / negative pole of axis k
        self.axes = AXES
        self.pos = np.zeros((len(AXES), self.n))
        self.neg = np.zeros((len(AXES), self.n))
        self.profile = [[] for _ in range(self.n)]       # [(pole code, weight), …]
        for name, ties in parse_emotion_weights(EMOTION_WEIGHTS).items():
            if name not in self.idx:
                continue
            i = self.idx[name]
            for code, weight in ties:
                k, sign = POLES[code]
                (self.pos if sign > 0 else self.neg)[k, i] = weight
                self.profile[i].append((code, weight))
        self.align = self.pos - self.neg                  # +: fits the positive pole

        # parameters (controlled by the UI)
        self.setpoint = 0.02
        self.resistance = 0.05
        self.learn_rate = 0.005
        self.fatigue_strength = 4.0
        self.experience_gap = 50
        self.experiences_on = True
        self.learning_on = True
        self.learn_free = 0.0       # learning rate factor outside experiences
        self.encode_gain = 0.1      # associative spread while experiencing the world
        self.emotionality = 1.0      # how strongly the mood changes sensitivity
        self.emo_reactivity = 1.0    # how strongly active concepts drive their emotion
        self.emo_halflife = 230      # ticks for an emotion to fade to half on its own
        self.predictability = 0.8    # chance the next step of a routine follows
        self.routine_gap = 10        # ticks between steps of a routine
        self.stdp_rate = 0.2         # spike-timing-dependent plasticity strength
        self.stdp_window = 40        # ticks over which "before" still counts

        self.reset_weights()
        self.reset_activity()

    # -- state ------------------------------------------------------------
    def reset_weights(self):
        n = self.n
        self.w = np.zeros((n, n))            # w[i, j] = strength of i → j
        self.seen = np.zeros(n)              # times each concept was experienced
        self.co_seen = np.zeros((n, n))      # times i and j were experienced together
        self.n_experiences = 0

    def reset_activity(self):
        n = self.n
        self.a = np.zeros(n)
        self.fatigue = np.zeros(n)
        self.ext = np.zeros(n)
        self.pulse = np.zeros(n)
        self.flow = np.zeros((n, n))
        self.mood = np.zeros(len(AXES))       # −1 … +1 per axis
        self.trace = np.zeros(n)             # recent activity, for STDP
        self.routine = None                  # (routine, step) currently being lived through
        self.pending = None                  # next scene of the routine
        self.scene_k = None
        self.sens = 1.0
        self.sens_eff = 1.0
        self.mean = 0.0
        self.t = 0
        self.scene_name = None
        self.phase_left = 10
        self.history = deque(maxlen=500)

    def save(self, path, layout_pos):
        """Store everything that was learned, plus the layout, in a .npz file."""
        np.savez_compressed(path, version=1, names=np.array(self.names), w=self.w,
                            seen=self.seen, co_seen=self.co_seen, trace=self.trace,
                            n_experiences=self.n_experiences, t=self.t, mood=self.mood,
                            pos=layout_pos)

    def load(self, path):
        """Load a saved network. Concepts are matched by name, so a save still loads
        after the scenes were edited: new concepts start untrained, removed ones are
        dropped. Returns (layout positions, matched, total saved)."""
        with np.load(path, allow_pickle=False) as d:
            names = [str(s) for s in d["names"]]
            src = [k for k, m in enumerate(names) if m in self.idx]
            dst = [self.idx[names[k]] for k in src]
            self.reset_weights()
            self.reset_activity()
            ix_src, ix_dst = np.ix_(src, src), np.ix_(dst, dst)
            self.w[ix_dst] = d["w"][ix_src]
            self.co_seen[ix_dst] = d["co_seen"][ix_src]
            self.seen[dst] = d["seen"][src]
            self.trace[dst] = d["trace"][src]
            self.n_experiences = int(d["n_experiences"])
            self.t = int(d["t"])
            if d["mood"].shape == self.mood.shape:
                self.mood[:] = d["mood"]
            pos = np.random.normal(0.0, 0.3, (self.n, 2))   # unmatched concepts start near the middle
            pos[dst] = d["pos"][src]
            return pos, len(dst), len(names)

    def stimulate(self, i, amount=2.5):
        self.pulse[i] = max(self.pulse[i], amount)

    def experienced(self, i, j):
        """Real-world experienced P(j | i), or None if i was never experienced."""
        if self.seen[i] == 0:
            return None
        return self.co_seen[i, j] / self.seen[i]

    def emotion_boost(self):
        """Per-concept sensitivity multiplier from the current mood: concepts that
        fit the mood become more sensitive, those of the opposite pole less."""
        return np.exp(EMO_GAIN * self.emotionality * (self.mood @ self.align))

    def pole_levels(self):
        """Current strength (0 … 1) of each pole, by pole code."""
        return {c: max(0.0, s * self.mood[k]) for c, (k, s) in POLES.items()}

    # -- simulation -------------------------------------------------------
    def _schedule(self):
        self.phase_left -= 1
        if self.phase_left > 0:
            return
        if self.scene_name is not None:            # experience ends
            self.scene_name = None
            self.ext[:] = 0
            if self.routine is not None:           # does the routine continue?
                r, pos = self.routine
                steps = self.routines[r][1]
                if pos + 1 < len(steps) and random.random() < self.predictability:
                    self.routine = (r, pos + 1)
                    self.pending = steps[pos + 1]
                    self.phase_left = max(1, int(self.routine_gap * random.uniform(0.5, 1.5)))
                    return
            self.routine = None
            self.phase_left = max(1, int(self.experience_gap * random.uniform(0.5, 1.5)))
        elif self.experiences_on:                   # new experience starts
            if self.pending is not None:
                k, self.pending = self.pending, None
            else:
                k = random.choices(range(len(self.scenes)), weights=self.scene_freq)[0]
                options = self.scene_routines.get(k)
                self.routine = random.choice(options) if options else None
            name, _, members = self.scenes[k]
            present = members[np.random.random(len(members)) < MEMBER_PROB]
            if len(present) < 2:
                present = members
            self.scene_name = name
            self.scene_k = k
            self.ext[:] = 0
            self.ext[present] = EXPERIENCE_INPUT
            self.seen[present] += 1
            self.co_seen[np.ix_(present, present)] += 1
            self.n_experiences += 1
            self.phase_left = EXPERIENCE_LEN
        else:
            self.phase_left = 10

    def routine_label(self):
        if self.routine is None:
            return ""
        r, pos = self.routine
        name, steps = self.routines[r]
        return f"{name} {pos + 1}/{len(steps)}"

    def step(self):
        self._schedule()
        # every concept sends activity × strength along each of its associations;
        # the resistance of an association is subtracted, so weak signals are blocked
        self.flow = np.maximum(0.0, self.a[:, None] * self.w - self.resistance)
        assoc_in = self.flow.sum(0)
        if self.scene_name:
            assoc_in *= self.encode_gain
        # spontaneous activity: now and then a random concept gets a small kick
        noise = NOISE_KICK * (np.random.random(self.n) < NOISE_PROB)
        internal = assoc_in + self.pulse + noise

        # homeostasis, fast part: the moment total activity exceeds the setpoint,
        # every concept becomes less sensitive (the slow part is below)
        over = max(0.0, self.mean - self.setpoint) / max(self.setpoint, 0.005)
        self.sens_eff = self.sens * math.exp(-FAST_HOMEO * over)

        # combine all inputs into a new activity level; sensitivity (global, times
        # the boost from emotions) scales how strongly a concept responds to its
        # associations (sensory input from the world always gets through)
        sens_i = self.sens_eff * self.emotion_boost()
        net = self.ext + sens_i * internal * (1.0 - self.fatigue)
        target = np.tanh(np.maximum(0.0, net - THRESHOLD))
        rise = ACT_RATE * (target - self.a)
        self.a += rise
        onset = np.maximum(0.0, rise)                 # a concept "fires" as its activity rises
        # fatigue (0…1) builds while a concept is active and recovers slowly
        self.fatigue += (FATIGUE_BUILD * self.fatigue_strength * self.a * (1.0 - self.fatigue)
                         - FATIGUE_RECOVER * self.fatigue)
        self.pulse *= 0.9

        # emotions slowly build while their concepts are active, and slowly fade
        # active concepts push each axis toward the pole they are tied to; the
        # mood also relaxes back toward neutral on its own
        up = np.minimum(1.0, (self.pos @ self.a) / EMO_SATURATE)
        down = np.minimum(1.0, (self.neg @ self.a) / EMO_SATURATE)
        decay = math.log(2) / max(1.0, self.emo_halflife)
        rate = EMO_RISE * self.emo_reactivity
        self.mood += rate * (up * (1.0 - self.mood) - down * (1.0 + self.mood)) - decay * self.mood
        self.mood = np.clip(self.mood, -1.0, 1.0)

        # homeostasis, slow part: sensitivity drifts up while activity is below
        # the setpoint and down while it is above
        self.mean = float(self.a.mean())
        self.sens *= math.exp(HOMEO_RATE * (self.setpoint - self.mean) / max(self.setpoint, 0.005))
        self.sens = min(SENS_MAX, max(SENS_MIN, self.sens))

        # Hebbian learning: w[i→j] drifts toward j's activity whenever i is active,
        # so it converges on P(j active | i active).
        if self.learning_on:
            g = 1.0 if self.scene_name else self.learn_free
            active = np.nonzero(self.a > 0.005)[0]      # only rows that change
            if len(active):
                self.w[active] += (g * self.learn_rate * self.a[active, None]
                                   * (self.a[None, :] - self.w[active]))
                self.w[active, active] = 0.0

            # STDP: when j starts firing while i was active shortly before,
            # i → j (outgoing from the earlier concept) strengthens and
            # j → i (incoming to it) weakens. Simultaneous onsets cancel out.
            post = np.nonzero(onset > 1e-4)[0]
            if self.stdp_rate > 0 and len(post):
                dw = g * self.stdp_rate * np.outer(self.trace, onset[post])  # [i, j∈post]
                self.w[:, post] += dw
                self.w[post, :] -= dw.T
                self.w[:, post] = np.clip(self.w[:, post], 0.0, 1.0)
                self.w[post, :] = np.clip(self.w[post, :], 0.0, 1.0)
                self.w[post, post] = 0.0
        self.trace += (self.a - self.trace) / max(1.0, self.stdp_window)

        self.t += 1
        self.history.append((self.mean, self.sens_eff, self.mood.copy()))


# ---------------------------------------------------------------------------
# Layout: associated concepts attract, all concepts repel each other.
# ---------------------------------------------------------------------------
class Layout:
    def __init__(self, n):
        k = np.arange(n) + 0.5                        # even "sunflower" spread
        r = np.sqrt(k / n) * 0.9
        ang = k * math.pi * (3 - math.sqrt(5))
        self.pos = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
        self.vel = np.zeros_like(self.pos)

    def step(self, w):
        p = self.pos
        n = len(p)
        d = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
        np.fill_diagonal(dist, 1.0)
        strength = w + w.T
        # damp hubs: a link counts less when both ends have many strong links,
        # so richly connected concepts don't pull everything into one ball
        deg = strength.sum(1) + 1e-3
        strength = strength / np.sqrt(np.outer(deg, deg)) * deg.mean()
        # scaled so any network size fits; softened at close range so crowded
        # nodes don't kick each other away
        repulse = (0.1 / n) / np.maximum(dist, 0.03) ** 2
        attract = -strength * 0.1 * (dist - 0.08)
        f = (((repulse + attract) / dist)[..., None] * d).sum(1)
        f -= p * 0.05                                  # gravity toward centre
        # nodes held by many strong springs move more carefully; without this,
        # well-trained networks overshoot and never settle
        f /= (1.0 + strength.sum(1) * 0.1)[:, None]
        self.vel = (self.vel + f * 0.2) * 0.7
        step = np.clip(self.vel, -0.03, 0.03)
        self.pos = np.clip(p + step, -1.05, 1.05)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
def hex_rgb(h):
    return tuple(int(h[k:k + 2], 16) / 255 for k in (1, 3, 5))


BG = (0.07, 0.08, 0.10)
EDGE_COL = (0.51, 0.57, 0.65)
HOT_STOPS = [(0.0, (0.16, 0.22, 0.30)), (0.35, (0.85, 0.35, 0.15)),
             (0.7, (1.0, 0.75, 0.2)), (1.0, (1.0, 0.98, 0.85))]


def heat(x):
    x = min(1.0, max(0.0, x))
    for (x0, c0), (x1, c1) in zip(HOT_STOPS, HOT_STOPS[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return HOT_STOPS[-1][1]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CSS = b"""
.side { font-size: 9pt; }
.side scale { padding-top: 0; padding-bottom: 2px; min-height: 0; }
.side scale trough { min-height: 4px; }
.side scale slider { min-width: 12px; min-height: 12px; margin: -5px; }
.side button { padding: 2px 6px; min-height: 0; }
.side checkbutton { padding: 0; min-height: 0; }
.side expander title { font-weight: bold; }
.side scale.axis trough { min-height: 6px; }
""" + b"".join(
    b"#axis%d trough { background-image: linear-gradient(to right, %s, #3a3f47 50%%, %s); }\n"
    % (k, neg_c.encode(), pos_c.encode()) for k, (_, neg_c, _, pos_c) in enumerate(AXES))

STRUCT_BUCKETS = [(0.12, 0.25, 0.09, 0.6), (0.25, 0.45, 0.16, 1.0), (0.45, 1.01, 0.28, 1.6)]
MAX_FLOW_EDGES = 160
MAX_STRUCT_EDGES = 900
RESISTANCE_STEP = 0.005                  # motivation slider: one step of resistance
MOTIVATION_STEPS = 50                    # … so the slider covers resistance 0.25 … 0
ZOOM_MIN, ZOOM_MAX = 0.5, 12.0


class SimWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="Symphony of the Mind - by Jits Krol")
        self.set_default_size(1500, 950)

        self.net = Network()
        self.layout = Layout(self.net.n)
        self.selected = None
        self.running = True
        self.all_labels = True
        self.steps_per_sec = 30.0
        self.step_acc = 0.0
        self.last_tick = time.monotonic()
        self.frame = 0
        self.screen_pos = np.zeros((self.net.n, 2))
        self.radius = np.full(self.net.n, 5.0)
        self.iu = np.triu_indices(self.net.n, 1)
        self.pole_colors = {}
        for neg, neg_c, pos, pos_c in AXES:
            self.pole_colors[neg] = hex_rgb(neg_c)
            self.pole_colors[pos] = hex_rgb(pos_c)
        code_axis = {c: AXES[k][0 if s < 0 else 2] for c, (k, s) in POLES.items()}
        self.code_color = {c: self.pole_colors[name] for c, name in code_axis.items()}
        self.syncing = False
        self.view = None                     # smoothed bounding box of the layout
        self.struct_cache = None             # (surface, size, frame) of the faint background lines
        self.notice = None                   # (message, until) shown on the canvas
        self.zoom = 1.0                      # camera: screen = fitted position × zoom + pan
        self.pan = np.zeros(2)
        self.pointer = None                  # last mouse position over the canvas
        self.overlays = True                 # status text, graphs and inspector (H toggles)
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")

        css = Gtk.CssProvider()
        css.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        # header bar: run/step and the sidebar toggle stay reachable when the sidebar is hidden
        header = Gtk.HeaderBar()
        self.set_titlebar(header)
        self.run_btn = Gtk.Button(label="Pause")
        self.run_btn.connect("clicked", lambda _b: self.toggle_run())
        step_btn = Gtk.Button(label="Step")
        step_btn.connect("clicked", lambda _b: self._do_steps(1))
        header.pack_start(self.run_btn)
        header.pack_start(step_btn)
        save_btn = Gtk.Button(label="Save", tooltip_text="Save the trained network (Ctrl+S)")
        save_btn.connect("clicked", lambda _b: self.save_dialog())
        load_btn = Gtk.Button(label="Load", tooltip_text="Load a saved network (Ctrl+O)")
        load_btn.connect("clicked", lambda _b: self.load_dialog())
        header.pack_start(save_btn)
        header.pack_start(load_btn)
        self.side_btn = Gtk.ToggleButton(active=True, tooltip_text="Show/hide controls (F9)")
        self.side_btn.set_icon_name("sidebar-show-right-symbolic")
        self.side_btn.connect("toggled", lambda b: self.revealer.set_reveal_child(b.get_active()))
        header.pack_end(self.side_btn)

        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.set_child(root)

        self.area = Gtk.DrawingArea(hexpand=True, vexpand=True)
        self.area.set_draw_func(self.on_draw)
        click = Gtk.GestureClick()
        click.connect("pressed", self.on_click)
        self.area.add_controller(click)
        # camera: mouse wheel zooms toward the pointer, middle button drags to pan
        motion = Gtk.EventControllerMotion()
        motion.connect("motion", lambda _c, x, y: setattr(self, "pointer", (x, y)))
        self.area.add_controller(motion)
        scroll = Gtk.EventControllerScroll(flags=Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect("scroll", self.on_scroll)
        self.area.add_controller(scroll)
        drag = Gtk.GestureDrag(button=Gdk.BUTTON_MIDDLE)
        drag.connect("drag-begin", self.on_pan_begin)
        drag.connect("drag-update", self.on_pan_update)
        drag.connect("drag-end", lambda *_: self.area.set_cursor_from_name(None))
        self.area.add_controller(drag)
        # for trackpads without a middle button (e.g. on a Mac): Shift + drag pans,
        # pinching zooms
        shift_drag = Gtk.GestureDrag(button=Gdk.BUTTON_PRIMARY)
        shift_drag.connect("drag-begin", self.on_shift_pan_begin)
        shift_drag.connect("drag-update", self.on_pan_update)
        shift_drag.connect("drag-end", lambda *_: self.area.set_cursor_from_name(None))
        self.area.add_controller(shift_drag)
        pinch = Gtk.GestureZoom()
        pinch.connect("begin", self.on_pinch_begin)
        pinch.connect("scale-changed", self.on_pinch)
        self.area.add_controller(pinch)
        root.append(self.area)

        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self.on_key)
        self.add_controller(keys)

        self.revealer = Gtk.Revealer(reveal_child=True,
                                     transition_type=Gtk.RevealerTransitionType.SLIDE_LEFT)
        self.revealer.set_hexpand(False)     # stop expanding labels inside from widening it
        scroller = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, hexpand=False)
        scroller.set_size_request(235, -1)
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        panel.add_css_class("side")
        for side in ("start", "end", "top", "bottom"):
            getattr(panel, f"set_margin_{side}")(8)
        scroller.set_child(panel)
        self.revealer.set_child(scroller)
        root.append(self.revealer)

        net = self.net
        self._slider(panel, "Simulation speed (ticks/s)", 0, 100, 45,
                     self._set_speed, lambda v: f"{self._speed_of(v):.0f}")

        box = self._section(panel, "Network", True)
        self._slider(box, "Homeostasis setpoint", 0, 150, net.setpoint * 1000,
                     lambda v: setattr(net, "setpoint", v / 1000), lambda v: f"{v / 10:.1f} %")
        # motivation is inverted resistance, in steps of 0.005 from 0.25 (0 %) to 0 (100 %)
        self._slider(box, "Motivation (resistance)", 0, MOTIVATION_STEPS,
                     MOTIVATION_STEPS - round(net.resistance / RESISTANCE_STEP),
                     lambda v: setattr(net, "resistance", (MOTIVATION_STEPS - v) * RESISTANCE_STEP),
                     lambda v: f"{v / MOTIVATION_STEPS * 100:.0f}% ({(MOTIVATION_STEPS - v) * RESISTANCE_STEP:.3f})",
                     tip="High motivation = low resistance: activation passes through associations "
                         "more easily. The number in brackets is the resistance subtracted from "
                         "every signal.")
        self._slider(box, "Fatigue", 0, 600, net.fatigue_strength * 100,
                     lambda v: setattr(net, "fatigue_strength", v / 100), lambda v: f"{v / 100:.1f}")

        box = self._section(panel, "Emotions", True)
        self._slider(box, "Emotionality", 0, 300, net.emotionality * 100,
                     lambda v: setattr(net, "emotionality", v / 100), lambda v: f"{v:.0f} %",
                     tip="How strongly the mood makes fitting concepts more sensitive "
                         "and concepts of the opposite pole less sensitive")
        self._slider(box, "Emotional reactivity", 0, 300, net.emo_reactivity * 100,
                     lambda v: setattr(net, "emo_reactivity", v / 100), lambda v: f"{v:.0f} %",
                     tip="How strongly the activity of emotionally charged concepts "
                         "builds up their emotion")
        self._slider(box, "Emotion half-life", 20, 2000, net.emo_halflife,
                     lambda v: setattr(net, "emo_halflife", v), lambda v: f"{v:.0f}",
                     tip="Ticks for an emotion to fade to half on its own")
        self.emo_scales = [self._axis_slider(box, k) for k in range(len(AXES))]

        box = self._section(panel, "Learning", False)
        self._slider(box, "Learning rate", 0, 100, net.learn_rate * 1000,
                     lambda v: setattr(net, "learn_rate", v / 1000), lambda v: f"{v / 1000:.3f}")
        self._slider(box, "Learn from own thoughts", 0, 100, net.learn_free * 100,
                     lambda v: setattr(net, "learn_free", v / 100), lambda v: f"{v:.0f} %")
        self._slider(box, "Attention to the world", 0, 100, (1 - net.encode_gain) * 100,
                     lambda v: setattr(net, "encode_gain", 1 - v / 100), lambda v: f"{v:.0f} %")
        self._slider(box, "Pause between experiences", 5, 400, net.experience_gap,
                     lambda v: setattr(net, "experience_gap", int(v)), lambda v: f"{v:.0f}")
        self._check(box, "Feed real-world experiences", True,
                    lambda on: setattr(net, "experiences_on", on))
        self._check(box, "Hebbian learning on", True, lambda on: setattr(net, "learning_on", on))

        box = self._section(panel, "Routines & STDP", True)
        self._slider(box, "Routine predictability", 0, 100, net.predictability * 100,
                     lambda v: setattr(net, "predictability", v / 100), lambda v: f"{v:.0f} %",
                     tip="Chance that the next step of a routine follows the current one")
        self._slider(box, "Gap within routines", 1, 100, net.routine_gap,
                     lambda v: setattr(net, "routine_gap", int(v)), lambda v: f"{v:.0f}")
        self._slider(box, "STDP strength", 0, 100, net.stdp_rate * 100,
                     lambda v: setattr(net, "stdp_rate", v / 100), lambda v: f"{v / 100:.2f}",
                     tip="When B fires after A: A→B strengthens, B→A weakens. 0 = off")
        self._slider(box, "STDP window", 2, 200, net.stdp_window,
                     lambda v: setattr(net, "stdp_window", v), lambda v: f"{v:.0f}",
                     tip="Ticks over which an earlier activation still counts as 'before'")

        box = self._section(panel, "View & reset", False)
        self._check(box, "Show all labels", True, lambda on: setattr(self, "all_labels", on))
        row = Gtk.Box(spacing=4)
        rw = Gtk.Button(label="Forget")
        rw.set_tooltip_text("Forget all associations")
        rw.connect("clicked", lambda _b: net.reset_weights())
        ra = Gtk.Button(label="Calm")
        ra.set_tooltip_text("Reset activity, fatigue and emotions")
        ra.connect("clicked", lambda _b: net.reset_activity())
        rl = Gtk.Button(label="Re-layout")
        rl.connect("clicked", lambda _b: self.relayout())
        for b in (rw, ra, rl):
            row.append(b)
        box.append(row)

        help_lbl = Gtk.Label(xalign=0, wrap=True, max_width_chars=30)
        help_lbl.set_markup(
            "<small>After training, turn off Feed Real-World Experiences to simulate associative trains of thought. "
            "Click a concept to stimulate it and inspect it; click empty space to "
            "deselect. Coloured rings show emotion links. Hover a slider for details. "
            "Mouse wheel or pinch = zoom, middle button or Shift + drag = pan, "
            "Home = reset view, H = hide "
            "overlays. Space = pause, F9 = hide controls.</small>")
        help_lbl.set_margin_top(8)
        panel.append(help_lbl)

        GLib.timeout_add(16, self.on_tick)

    # -- widgets ----------------------------------------------------------
    def _section(self, panel, title, expanded):
        exp = Gtk.Expander(label=title, expanded=expanded)
        exp.set_margin_top(4)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        box.set_margin_start(4)
        exp.set_child(box)
        panel.append(exp)
        return box

    def _slider(self, box, title, lo, hi, init, setter, fmt, markup=False, tip=None):
        head = Gtk.Box(spacing=4)
        if tip:
            head.set_tooltip_text(tip)
        name = Gtk.Label(xalign=0, hexpand=True, ellipsize=3)
        if markup:
            name.set_markup(title)
        else:
            name.set_label(title)
        val = Gtk.Label(label=fmt(init), xalign=1)
        head.append(name)
        head.append(val)
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, lo, hi, 1)
        scale.set_value(init)
        scale.set_draw_value(False)

        def changed(s):
            v = s.get_value()
            val.set_label(fmt(v))
            if not self.syncing:
                setter(v)
        scale.connect("value-changed", changed)
        if tip:
            scale.set_tooltip_text(tip)
        box.append(head)
        box.append(scale)
        return scale

    def _check(self, box, label, init, setter):
        chk = Gtk.CheckButton(label=label, active=init)
        chk.connect("toggled", lambda b: setter(b.get_active()))
        box.append(chk)

    def _axis_slider(self, box, k):
        """A mood axis: negative pole on the left, positive pole on the right."""
        neg, neg_c, pos, pos_c = AXES[k]
        head = Gtk.Box(spacing=4)
        left = Gtk.Label(xalign=0)
        left.set_markup(f"<span foreground='{neg_c}'>●</span> {neg.capitalize()}")
        val = Gtk.Label(label="neutral", hexpand=True)
        val.set_opacity(0.7)
        right = Gtk.Label(xalign=1)
        right.set_markup(f"{pos.capitalize()} <span foreground='{pos_c}'>●</span>")
        for lbl in (left, val, right):
            head.append(lbl)
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -100, 100, 1)
        scale.set_name(f"axis{k}")
        scale.add_css_class("axis")
        scale.set_draw_value(False)
        scale.set_has_origin(False)
        scale.add_mark(0, Gtk.PositionType.BOTTOM, None)
        scale.set_value(0)
        tip = (f"Current mood between {neg} and {pos}. It moves on its own as concepts are "
               f"active; drag to set it.")
        head.set_tooltip_text(tip)
        scale.set_tooltip_text(tip)

        def changed(s):
            v = s.get_value()
            val.set_label("neutral" if abs(v) < 3 else
                          f"◀ {abs(v):.0f}%" if v < 0 else f"{v:.0f}% ▶")
            if not self.syncing:
                self.net.mood[k] = v / 100
        scale.connect("value-changed", changed)
        box.append(head)
        box.append(scale)
        return scale

    @staticmethod
    def _speed_of(v):
        return 0.0 if v <= 0 else 0.5 * 1.087 ** v

    def _set_speed(self, v):
        self.steps_per_sec = self._speed_of(v)

    def relayout(self):
        self.layout = Layout(self.net.n)
        self.view = None
        self.reset_camera()

    def toggle_run(self):
        self.running = not self.running
        self.run_btn.set_label("Pause" if self.running else "Run")

    # -- saving -----------------------------------------------------------
    def _dialog(self, title):
        os.makedirs(self.save_dir, exist_ok=True)
        dialog = Gtk.FileDialog(title=title, modal=True)
        dialog.set_initial_folder(Gio.File.new_for_path(self.save_dir))
        flt = Gtk.FileFilter(name="Saved networks (*.npz)")
        flt.add_pattern("*.npz")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(flt)
        dialog.set_filters(filters)
        return dialog

    def save_dialog(self):
        dialog = self._dialog("Save trained network")
        dialog.set_initial_name(f"network-tick{self.net.t}.npz")
        dialog.save(self, None, self._on_save_chosen)

    def _on_save_chosen(self, dialog, result):
        try:
            path = dialog.save_finish(result).get_path()
        except GLib.Error:
            return                                  # cancelled
        if not path.endswith(".npz"):
            path += ".npz"
        try:
            self.net.save(path, self.layout.pos)
            self.show_notice(f"Saved to {os.path.basename(path)}")
        except OSError as e:
            self.show_notice(f"Could not save: {e}")

    def load_dialog(self):
        self._dialog("Load trained network").open(self, None, self._on_load_chosen)

    def _on_load_chosen(self, dialog, result):
        try:
            path = dialog.open_finish(result).get_path()
        except GLib.Error:
            return                                  # cancelled
        try:
            pos, matched, total = self.net.load(path)
        except (OSError, KeyError, ValueError) as e:
            self.show_notice(f"Could not load: {e}")
            return
        self.relayout()
        self.layout.pos = pos
        self.struct_cache = None
        self.selected = None
        self.sync_emotion_sliders()
        msg = f"Loaded {os.path.basename(path)}"
        if matched < self.net.n or matched < total:
            msg += f" — {matched} of {self.net.n} concepts matched"
        self.show_notice(msg)

    def show_notice(self, text, seconds=4):
        self.notice = (text, time.monotonic() + seconds)

    def on_key(self, _ctl, keyval, _code, state):
        command = (Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.META_MASK
                   | Gdk.ModifierType.SUPER_MASK)      # Ctrl, or Cmd on a Mac
        if state & command:
            if keyval in (Gdk.KEY_s, Gdk.KEY_S):
                self.save_dialog()
                return True
            if keyval in (Gdk.KEY_o, Gdk.KEY_O):
                self.load_dialog()
                return True
        if keyval in (Gdk.KEY_h, Gdk.KEY_H) and not state & command:
            self.overlays = not self.overlays
            return True
        if keyval in (Gdk.KEY_Home, Gdk.KEY_0, Gdk.KEY_KP_0):
            self.reset_camera()
            return True
        if keyval == Gdk.KEY_F9:
            self.side_btn.set_active(not self.side_btn.get_active())
            return True
        if keyval == Gdk.KEY_space and not isinstance(self.get_focus(), Gtk.Button):
            self.toggle_run()
            return True
        return False

    # -- loop -------------------------------------------------------------
    def _do_steps(self, k):
        for _ in range(k):
            self.net.step()

    def on_tick(self):
        now = time.monotonic()
        elapsed = min(0.25, now - self.last_tick)   # real time, so speed stays true
        self.last_tick = now                        # even when drawing is slow
        if self.running:
            self.step_acc += self.steps_per_sec * elapsed
            k = int(self.step_acc)
            self.step_acc -= k
            self._do_steps(min(k, 300))
        self.layout.step(self.net.w)
        self.area.queue_draw()
        self.frame += 1
        if self.frame % 6 == 0:
            self.sync_emotion_sliders()
        return True

    def sync_emotion_sliders(self):
        self.syncing = True
        for k, scale in enumerate(self.emo_scales):
            scale.set_value(round(self.net.mood[k] * 100))
        self.syncing = False

    # -- input ------------------------------------------------------------
    def on_scroll(self, _ctl, _dx, dy):
        mx, my = self.pointer if self.pointer else (self.area.get_width() / 2,
                                                    self.area.get_height() / 2)
        self.zoom_at(mx, my, self.zoom * 1.15 ** (-dy))
        return True

    def on_pan_begin(self, _gesture, _x, _y):
        self.pan_start = self.pan.copy()
        self.area.set_cursor_from_name("grabbing")

    def on_shift_pan_begin(self, gesture, x, y):
        if gesture.get_current_event_state() & Gdk.ModifierType.SHIFT_MASK:
            self.on_pan_begin(gesture, x, y)
        else:
            gesture.set_state(Gtk.EventSequenceState.DENIED)    # a plain click, not a pan

    def on_pan_update(self, _gesture, dx, dy):
        self.pan = self.pan_start + np.array([dx, dy])

    def zoom_at(self, mx, my, new):
        """Zoom to `new`, keeping the screen point (mx, my) where it is."""
        new = min(ZOOM_MAX, max(ZOOM_MIN, new))
        self.pan = np.array([mx, my]) - (np.array([mx, my]) - self.pan) * (new / self.zoom)
        self.zoom = new

    def on_pinch_begin(self, gesture, _seq):
        self.pinch_start = self.zoom
        ok, cx, cy = gesture.get_bounding_box_center()
        self.pinch_center = (cx, cy) if ok else (self.area.get_width() / 2,
                                                 self.area.get_height() / 2)

    def on_pinch(self, _gesture, scale):
        self.zoom_at(*self.pinch_center, self.pinch_start * scale)

    def reset_camera(self):
        self.zoom = 1.0
        self.pan = np.zeros(2)

    def on_click(self, gesture, _n, x, y):
        if gesture.get_current_event_state() & Gdk.ModifierType.SHIFT_MASK:
            return                                   # Shift + drag pans the view
        # undo the camera to find what was clicked in layout (fitted) coordinates
        x, y = (x - self.pan[0]) / self.zoom, (y - self.pan[1]) / self.zoom
        d = np.hypot(self.screen_pos[:, 0] - x, self.screen_pos[:, 1] - y) - self.radius
        i = int(np.argmin(d))
        if d[i] < 8 / self.zoom:
            self.selected = i
            self.net.stimulate(i)
        else:
            self.selected = None

    # -- drawing ----------------------------------------------------------
    def on_draw(self, _area, cr, width, height):
        n = self.net
        cr.set_source_rgb(*BG)
        cr.paint()

        graph_h = 100
        margin = 36
        # fit the layout to the window (smoothed, so the view doesn't jitter)
        pos = self.layout.pos
        # robust bounds: a few outlying concepts shouldn't squeeze everything else
        box = np.concatenate([np.percentile(pos, 2, axis=0), np.percentile(pos, 98, axis=0)])
        self.view = box if self.view is None else self.view * 0.9 + box * 0.1
        lo, hi = self.view[:2], np.maximum(self.view[2:], self.view[:2] + 1e-3)
        top = 70
        avail_w, avail_h = width - 2 * margin, height - graph_h - top - margin
        sp = np.empty_like(pos)
        sp[:, 0] = margin + (pos[:, 0] - lo[0]) / (hi[0] - lo[0]) * avail_w
        sp[:, 1] = top + (pos[:, 1] - lo[1]) / (hi[1] - lo[1]) * avail_h
        sp[:, 0] = np.clip(sp[:, 0], margin, width - margin)
        sp[:, 1] = np.clip(sp[:, 1], top * 0.6, height - graph_h - 12)
        self.screen_pos = sp
        self.boost = n.emotion_boost()
        tilt = np.clip(np.log(self.boost), -1.0, 1.0)      # how open/closed the mood makes it
        self.radius = radius = (5 + 9 * n.a) * (1 + 0.25 * tilt)
        spl, rl = sp.tolist(), radius.tolist()

        # the faint background lines are costly, so they are redrawn only every
        # few frames and reused in between; while zooming or panning the cached
        # picture is stretched to fit and redrawn sharp shortly after
        z, (px, py) = self.zoom, self.pan
        c = self.struct_cache
        moved = c is not None and (c[3], c[4], c[5]) != (z, px, py)
        if (c is None or c[1] != (width, height) or self.frame - c[2] >= 15
                or (moved and self.frame - c[2] >= 4)):
            surf = cr.get_target().create_similar(cairo.CONTENT_COLOR_ALPHA, width, height)
            sc = cairo.Context(surf)
            sc.translate(px, py)
            sc.scale(z, z)
            self._draw_structure(sc, n, spl)
            self.struct_cache = c = (surf, (width, height), self.frame, z, px, py)
        cr.save()
        s = z / c[3]
        cr.translate(px - c[4] * s, py - c[5] * s)
        cr.scale(s, s)
        cr.set_source_surface(c[0], 0, 0)
        cr.paint()
        cr.restore()

        cr.save()                                   # network drawn through the camera
        cr.translate(px, py)
        cr.scale(z, z)
        self._draw_flows(cr, n, spl, rl)
        self._draw_nodes(cr, n, spl, rl)
        cr.restore()
        if not self.overlays:
            return
        self._draw_status(cr, n)
        if self.selected is not None:
            self._draw_selected(cr, n, width)
        self._draw_graphs(cr, n, 12, height - graph_h + 4, width - 24, graph_h - 14)

    def _draw_structure(self, cr, n, spl):
        """All learned associations as faint straight lines (both directions merged)."""
        s = np.maximum(n.w, n.w.T)[self.iu]
        ii, jj = self.iu
        if (s >= STRUCT_BUCKETS[0][0]).sum() > MAX_STRUCT_EDGES:   # keep only the strongest
            s = np.where(s >= np.partition(s, -MAX_STRUCT_EDGES)[-MAX_STRUCT_EDGES], s, 0.0)
        cr.set_antialias(cairo.ANTIALIAS_FAST)
        for lo, hi, alpha, width in STRUCT_BUCKETS:
            sel = np.nonzero((s >= lo) & (s < hi))[0]
            if not len(sel):
                continue
            cr.set_source_rgba(*EDGE_COL, alpha)
            cr.set_line_width(width)
            for i, j in zip(ii[sel].tolist(), jj[sel].tolist()):
                cr.move_to(*spl[i])
                cr.line_to(*spl[j])
            cr.stroke()

    def _draw_flows(self, cr, n, spl, rl):
        """Directed curves for associations carrying activation now, and for the selection."""
        edges = {}
        ii, jj = np.nonzero(n.flow > 0.02)
        if len(ii) > MAX_FLOW_EDGES:
            top = np.argsort(n.flow[ii, jj])[-MAX_FLOW_EDGES:]
            ii, jj = ii[top], jj[top]
        for i, j in zip(ii.tolist(), jj.tolist()):
            edges[(i, j)] = False
        if self.selected is not None:
            s = self.selected
            for j in np.nonzero(n.w[s] > 0.08)[0].tolist():
                edges[(s, j)] = True
            for i in np.nonzero(n.w[:, s] > 0.08)[0].tolist():
                edges[(i, s)] = True
        order = sorted(edges, key=lambda e: n.flow[e] + n.w[e] * 0.01)
        for i, j in order:
            self._draw_edge(cr, spl[i], spl[j], rl[i], rl[j], float(n.w[i, j]),
                            float(n.flow[i, j]), edges[(i, j)])

    def _draw_edge(self, cr, p, q, rp, rq, w, flow, highlight):
        # plain floats instead of numpy: this runs for hundreds of edges per frame
        px, py = p
        qx, qy = q
        dx, dy = qx - px, qy - py
        dist = math.hypot(dx, dy)
        if dist < 1:
            return
        # control point bulges to the right-hand side of the travel direction
        cx = (px + qx) / 2 - dy * 0.12
        cy = (py + qy) / 2 + dx * 0.12
        # trim to circle edges along the curve's end tangents
        l1 = math.hypot(cx - px, cy - py)
        ax, ay = px + (cx - px) / l1 * rp, py + (cy - py) / l1 * rp
        l2 = math.hypot(qx - cx, qy - cy)
        tx, ty = (qx - cx) / l2, (qy - cy) / l2
        bx, by = qx - tx * (rq + 3), qy - ty * (rq + 3)

        f = min(1.0, flow * 3)
        g = 1 - f
        alpha = min(1.0, 0.25 + 0.6 * w + 0.6 * f + (0.3 if highlight else 0.0))
        if highlight and f < 0.1:
            cr.set_source_rgba(0.85, 0.9, 1.0, alpha)
        else:
            cr.set_source_rgba(EDGE_COL[0] * g + f, EDGE_COL[1] * g + 0.7 * f,
                               EDGE_COL[2] * g + 0.25 * f, alpha)
        cr.set_line_width(0.6 + 3.5 * w)
        cr.move_to(ax, ay)
        cr.curve_to(ax + (cx - ax) * 2 / 3, ay + (cy - ay) * 2 / 3,
                    bx + (cx - bx) * 2 / 3, by + (cy - by) * 2 / 3, bx, by)
        cr.stroke()
        size = 4 + 5 * w
        sx, sy = -ty * size * 0.5, tx * size * 0.5
        kx, ky = bx - tx * size, by - ty * size
        cr.move_to(bx, by)
        cr.line_to(kx + sx, ky + sy)
        cr.line_to(kx - sx, ky - sy)
        cr.close_path()
        cr.fill()

    def _draw_nodes(self, cr, n, spl, rl):
        acts = n.a.tolist()
        ext = (n.ext > 0).tolist()
        levels = n.pole_levels()
        tilt = np.clip(np.log(self.boost), -1.5, 1.5).tolist()
        for i in range(n.n):
            x, y = spl[i]
            r = rl[i]
            act = acts[i]
            if act > 0.05:                                  # glow
                cr.set_source_rgba(1.0, 0.6, 0.2, 0.18 * act)
                cr.arc(x, y, r * 2.0, 0, 2 * math.pi)
                cr.fill()
            # resting colour shows the mood's effect: brighter when the mood
            # opens a concept up, sunk into the background when it closes it
            col = heat(act)
            t = tilt[i]
            if t > 0:
                col = tuple(c + (h - c) * min(1, t / 1.2) * (1 - act)
                            for c, h in zip(col, (0.45, 0.52, 0.62)))
            elif t < 0:
                col = tuple(c + (b - c) * min(0.85, -t / 1.2) * (1 - act)
                            for c, b in zip(col, BG))
            cr.set_source_rgb(*col)
            cr.arc(x, y, r, 0, 2 * math.pi)
            cr.fill()
            if ext[i]:                                      # currently experienced
                cr.set_source_rgb(0.45, 0.8, 1.0)
                cr.set_line_width(2)
                cr.arc(x, y, r, 0, 2 * math.pi)
                cr.stroke()
            ties = n.profile[i]
            if ties:                                        # emotion ring segments
                seg = 2 * math.pi / len(ties)
                gap = 0.15 if len(ties) > 1 else 0.0
                for k, (code, wt) in enumerate(ties):
                    lvl = levels[code]
                    cr.set_source_rgba(*self.code_color[code], 0.2 + 0.3 * wt + 0.5 * lvl)
                    cr.set_line_width(0.5 + 1.2 * wt + 3.0 * wt * lvl)
                    cr.new_sub_path()
                    cr.arc(x, y, r + 2.5, -math.pi / 2 + k * seg + gap,
                           -math.pi / 2 + (k + 1) * seg - gap)
                    cr.stroke()
            if i == self.selected:
                cr.set_source_rgb(1, 1, 1)
                cr.set_line_width(2)
                cr.arc(x, y, r + 6, 0, 2 * math.pi)
                cr.stroke()

        for i in range(n.n):                                # labels on top
            act = acts[i]
            big = act > 0.15 or i == self.selected
            if not (big or self.all_labels):
                continue
            x, y = spl[i]
            cr.select_font_face("Sans", 0, 1 if big else 0)
            cr.set_font_size(12 if big else 9)
            label = n.names[i]
            ext_ = cr.text_extents(label)
            alpha = 0.95 if big else min(0.85, max(0.12, 0.4 + 0.3 * tilt[i]))
            cr.set_source_rgba(0.9, 0.92, 0.95, alpha)
            cr.move_to(x - ext_.width / 2 - ext_.x_bearing, y + rl[i] + (13 if big else 10))
            cr.show_text(label)

    def _draw_status(self, cr, n):
        cr.set_source_rgba(*BG, 0.75)                # keeps the text readable over a zoomed network
        cr.rectangle(6, 6, 300, 92)
        cr.fill()
        cr.select_font_face("Sans", 0, 1)
        cr.set_font_size(15)
        cr.set_source_rgb(0.88, 0.9, 0.94)
        cr.move_to(14, 24)
        cr.show_text(f"Experiencing: {n.scene_name}" if n.scene_name else "Free association")
        routine = n.routine_label()
        if routine:
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(12)
            cr.set_source_rgba(0.6, 0.8, 1.0, 0.9)
            cr.show_text(f"   ·  {routine}" + ("  (next step coming)" if n.pending is not None else ""))
        cr.select_font_face("Monospace", 0, 0)
        cr.set_font_size(11)
        cr.set_source_rgba(0.75, 0.78, 0.82, 0.9)
        lines = [f"tick {n.t}   experiences {n.n_experiences}",
                 f"activity {n.mean * 100:4.1f}% (target {n.setpoint * 100:.1f}%)",
                 f"sensitivity {n.sens_eff:.2f}"]
        if not self.running or self.steps_per_sec == 0:
            lines.append("paused")
        if abs(self.zoom - 1) > 1e-3 or self.pan.any():
            lines.append(f"zoom {self.zoom:.1f}×  (Home to reset)")
        if self.notice and time.monotonic() < self.notice[1]:
            lines.append(self.notice[0])
        for k, line in enumerate(lines):
            cr.move_to(14, 42 + 14 * k)
            cr.show_text(line)

    def _draw_selected(self, cr, n, width):
        i = self.selected
        others = [j for j in range(n.n) if j != i]
        others.sort(key=lambda j: -(n.w[i, j] + n.w[j, i]
                                    + (n.experienced(i, j) or 0) + (n.experienced(j, i) or 0)))
        boost = self.boost[i]
        levels = n.pole_levels()
        lines = [(f"{n.names[i]}", None),
                 (f"activity {n.a[i]:.2f}  fatigue {n.fatigue[i]:.2f}", None),
                 (f"mood effect ×{boost:.2f} sensitivity  seen {int(n.seen[i])}×", None)]
        for code, wt in n.profile[i]:
            lines.append((f"  {POLE_NAMES[code]:<9} tie {wt:.1f}   mood now {levels[code] * 100:3.0f}%",
                          self.code_color[code]))
        if not n.profile[i]:
            lines.append(("  emotionally neutral", None))
        lines.append(("", None))
        lines.append(("learned (experienced together)", None))
        lines.append((f"{'':<11}{'→ out':>12}{'← in':>12}", None))

        def fmt(v):
            return " -- " if v is None else f"{v:.2f}"
        for j in others[:14]:
            fwd, back = n.w[i, j], n.w[j, i]
            if fwd < 0.005 and back < 0.005 and not n.co_seen[i, j]:
                continue
            lines.append((f"{n.names[j][:10]:<10} {fwd:.2f}({fmt(n.experienced(i, j))})"
                          f" {back:.2f}({fmt(n.experienced(j, i))})", None))

        cr.select_font_face("Monospace", 0, 0)
        cr.set_font_size(11)
        w, lh = 300, 14
        x0, y0 = width - w - 12, 12
        cr.set_source_rgba(0.05, 0.06, 0.08, 0.85)
        cr.rectangle(x0, y0, w, lh * len(lines) + 12)
        cr.fill()
        for k, (text, col) in enumerate(lines):
            if k == 0:
                cr.select_font_face("Monospace", 0, 1)
                cr.set_source_rgb(1, 1, 1)
            else:
                cr.select_font_face("Monospace", 0, 0)
                cr.set_source_rgb(*(col or (0.8, 0.83, 0.87)))
            cr.move_to(x0 + 8, y0 + 16 + k * lh)
            cr.show_text(text)

    def _draw_graphs(self, cr, n, x, y, w, h):
        hist = list(n.history)
        gap = 12
        w1 = w * 0.55
        w2 = w - w1 - gap
        for gx, gw in ((x, w1), (x + w1 + gap, w2)):
            cr.set_source_rgb(*BG)                   # solid, so a zoomed network doesn't show through
            cr.rectangle(gx, y, gw, h)
            cr.fill()
            cr.set_source_rgba(1, 1, 1, 0.04)
            cr.rectangle(gx, y, gw, h)
            cr.fill()
        cr.select_font_face("Sans", 0, 0)
        cr.set_font_size(10)
        cr.set_source_rgb(0.6, 0.65, 0.7)
        cr.move_to(x + 6, y + 12)
        cr.show_text("activity (orange) vs setpoint (dashed) · sensitivity, log (blue)")
        cr.move_to(x + w1 + gap + 6, y + 12)
        cr.show_text("mood: happy / calm / affection ↑   sad / fear / anger ↓")
        mid = y + 8 + (h - 8) / 2
        cr.set_source_rgba(1, 1, 1, 0.25)
        cr.set_line_width(1)
        cr.move_to(x + w1 + gap, mid)
        cr.line_to(x + w, mid)
        cr.stroke()

        top = max(0.05, n.setpoint * 3)
        sy = y + h - min(1, n.setpoint / top) * h
        cr.set_source_rgba(1, 1, 1, 0.5)
        cr.set_dash([4, 4])
        cr.set_line_width(1)
        cr.move_to(x, sy)
        cr.line_to(x + w1, sy)
        cr.stroke()
        cr.set_dash([])
        if len(hist) < 2:
            return
        maxlen = n.history.maxlen

        def plot(gx, gw, values, to_y):
            dx = gw / (maxlen - 1)
            x0 = gx + gw - (len(values) - 1) * dx
            for k, v in enumerate(values):
                (cr.move_to if k == 0 else cr.line_to)(x0 + k * dx, to_y(v))
            cr.stroke()

        lo, hi = math.log(SENS_MIN), math.log(SENS_MAX)
        cr.set_source_rgba(0.45, 0.7, 1.0, 0.8)
        cr.set_line_width(1.2)
        plot(x, w1, [s for _, s, _ in hist], lambda s: y + h - (math.log(s) - lo) / (hi - lo) * h)
        cr.set_source_rgb(1.0, 0.6, 0.2)
        cr.set_line_width(1.6)
        plot(x, w1, [m for m, _, _ in hist], lambda m: y + h - min(1, m / top) * h)

        moods = np.array([m for _, _, m in hist])
        dx = w2 / (maxlen - 1)
        x0 = x + w1 + gap + w2 - (len(hist) - 1) * dx
        half = (h - 12) / 2
        cr.set_line_width(1.6)
        for k, (neg, _, pos, _) in enumerate(AXES):      # colour follows the pole it leans to
            vals = moods[:, k].tolist()
            for sign, name in ((1, pos), (-1, neg)):
                cr.set_source_rgb(*self.pole_colors[name])
                drawing = False
                for j, v in enumerate(vals):
                    px, py = x0 + j * dx, mid - v * half
                    if v * sign > 0 or (j and vals[j - 1] * sign > 0):
                        (cr.line_to if drawing else cr.move_to)(px, py)
                        drawing = True
                    else:
                        drawing = False
                cr.stroke()


class App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.symphony.HebbianSim")

    def do_activate(self):
        win = SimWindow(self)
        win.present()


if __name__ == "__main__":
    App().run(None)
