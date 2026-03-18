#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import textwrap


@dataclass(frozen=True)
class Topic:
    slug: str
    title: str
    subject: str
    issue: str
    mechanism: str
    cue: str
    mistake: str
    fix: str
    context: str
    product: str
    analogy: str


@dataclass(frozen=True)
class Scene:
    slug: str
    place: str
    routine: str
    disruption: str
    image: str
    shift: str
    closing: str


TOPICS = [
    Topic(
        slug="battery-heat",
        title="Why fast charging warms a battery",
        subject="battery charging",
        issue="a device warms noticeably during aggressive charging",
        mechanism="higher current magnifies small internal resistance, so part of the incoming energy leaves as heat instead of stored chemical potential",
        cue="mild warmth rises gradually while dangerous heat feels abrupt, sharp, or paired with swelling",
        mistake="treating every warm phone as a failure or, in the opposite direction, ignoring obvious danger signs",
        fix="slowing the charge rate, improving airflow, and checking the cable and adapter before blaming the cell itself",
        context="someone notices the back of a phone warming near the top of the charge curve",
        product="power management",
        analogy="The system behaves more like a crowded hallway than a magic bucket: the harder traffic is pushed, the more friction shows up.",
    ),
    Topic(
        slug="photosynthesis",
        title="Photosynthesis without the textbook fog",
        subject="photosynthesis",
        issue="plant energy explanations drift into vague language about plants making food from light",
        mechanism="light excites electrons, cells build energy-rich molecules from that motion, and those molecules help assemble sugars from carbon dioxide and water",
        cue="healthy leaves turn sunlight into stored chemical work rather than into a mystical green force",
        mistake="talking about chlorophyll as if it does every step by itself",
        fix="tracking what light starts, what molecules carry the energy forward, and where the stored sugar is spent later",
        context="a student wants a clearer way to describe how leaves fuel growth",
        product="plant science",
        analogy="It is closer to a quiet factory shift than to a magic trick. Inputs arrive, energy is routed, and outputs leave in a usable form.",
    ),
    Topic(
        slug="basil-roots",
        title="Why basil droops in the afternoon",
        subject="container basil care",
        issue="a basil plant looks wilted by midafternoon even though it is watered every morning",
        mechanism="heat increases demand on the leaves while heavy soil or weak drainage can keep roots too wet to breathe",
        cue="the surface may look dry while the lower root zone is still carrying yesterday's water",
        mistake="adding more water on schedule without checking the potting mix below the surface",
        fix="checking moisture an inch down, improving drainage, and letting the roots guide the timing instead of the clock",
        context="a kitchen gardener is trying to separate underwatering from overwatering",
        product="home gardening",
        analogy="The plant is asking for balance, not constant attention. Roots need both water and air in the same small container.",
    ),
    Topic(
        slug="router-state",
        title="When only one device loses Wi-Fi",
        subject="home networking",
        issue="a laptop cannot reach the internet while a phone on the same Wi-Fi still works",
        mechanism="the router may be healthy while the laptop is stuck with stale credentials, a bad lease, or confused interface state",
        cue="mixed device behavior points to local state on the failing machine before it points to a whole-network outage",
        mistake="rebooting every box at once and losing the chance to see where the fault actually lives",
        fix="reconnecting the laptop, checking the assigned address, then restarting components one at a time if needed",
        context="someone wants a calmer troubleshooting order for a flaky connection",
        product="network support",
        analogy="It is like tracing a hallway light: if the other rooms still work, the wall switch in one room becomes the first suspect.",
    ),
    Topic(
        slug="starter-rhythm",
        title="A sourdough starter that changes mood",
        subject="sourdough maintenance",
        issue="a starter doubles one day and barely moves the next",
        mechanism="fermentation speed shifts with temperature, feeding ratio, flour minerals, and the time window used for observation",
        cue="bubbles, aroma, and texture often tell the truth before the jar height does",
        mistake="changing flour, feeding time, hydration, and room temperature all at once",
        fix="holding one condition steady for a few feedings so the actual signal can emerge from the noise",
        context="a baker wants to know whether an uneven starter is unhealthy or simply reacting to conditions",
        product="kitchen fermentation",
        analogy="The culture behaves like a choir warming up: the room, the timing, and the pacing all change the volume you hear.",
    ),
    Topic(
        slug="dry-room",
        title="Why heated rooms feel harsher overnight",
        subject="winter indoor air",
        issue="a bedroom feels dry and scratchy after the heater runs all night",
        mechanism="cold outdoor air holds little moisture, and warming it indoors lowers relative humidity unless water is added back",
        cue="warmth alone does not guarantee comfort when the air starts stealing moisture from skin and throat",
        mistake="treating temperature and humidity as if they move together",
        fix="measuring humidity, adding moisture gradually, and reducing airflow that is stronger than the room needs",
        context="someone wakes up with a dry throat during heating season",
        product="home comfort",
        analogy="The room can feel like a towel left by a radiator: warm to the touch but eager to pull moisture from anything nearby.",
    ),
    Topic(
        slug="rain-radar",
        title="Why rain radar uses colors",
        subject="weather radar interpretation",
        issue="a radar map uses green, yellow, and red instead of one neutral scale",
        mechanism="a color ramp compresses intensity differences into a pattern the eye can compare almost instantly",
        cue="people notice severe pockets faster when color changes map to reflectivity or rainfall intensity",
        mistake="assuming every provider uses the exact same legend or threshold",
        fix="checking the legend once and then using the colors as a quick relative guide rather than a universal law",
        context="a weather watcher wants to read a storm map at a glance",
        product="forecasting",
        analogy="The color scale works like a highlighted margin note. It tells the eye where to pause before the brain has finished reading the numbers.",
    ),
    Topic(
        slug="charge-limit",
        title="Why laptops stop charging at eighty percent",
        subject="battery longevity settings",
        issue="a laptop offers a mode that stops charging before one hundred percent",
        mechanism="lithium-ion cells age faster when they sit completely full and warm for long stretches",
        cue="limiting the top of the charge window can trade a little immediate runtime for slower long-term wear",
        mistake="reading the feature as proof that the battery is defective",
        fix="matching the limit to real use, especially for machines that spend most days plugged into a desk setup",
        context="a user is deciding whether an eighty percent cap is helping or hurting",
        product="device maintenance",
        analogy="It is similar to leaving a little space in a packed bag. The bag may hold slightly less, but the seams strain less over time.",
    ),
    Topic(
        slug="kettle-scale",
        title="Why kettles get louder before they fail",
        subject="scale buildup in kettles",
        issue="a kettle sounds harsher and takes longer to boil after months of hard-water use",
        mechanism="mineral scale coats the heating surface, which reduces efficient heat transfer and changes the way bubbles form and collapse",
        cue="the noise often appears before the extra boil time becomes obvious",
        mistake="assuming the only explanation is a failing electrical element",
        fix="descaling the vessel, checking the water source, and comparing the boil cycle again before replacing the appliance",
        context="a kitchen appliance begins to hiss and rumble more than it used to",
        product="appliance care",
        analogy="The mineral layer behaves like a winter coat on the heater: some heat still gets through, but not with the same clean pace.",
    ),
    Topic(
        slug="compost-heat",
        title="Why a compost pile cools down",
        subject="compost management",
        issue="a compost heap heats quickly, then loses momentum a few days later",
        mechanism="microbes burn through easy food first, and the pile cools when moisture, oxygen, or the mix of browns and greens falls out of balance",
        cue="a cooling pile is often asking for structure, water, or turning instead of for a dramatic rescue",
        mistake="adding more scraps without restoring airflow through the mass",
        fix="adjusting moisture, fluffing the pile, and pairing fresh greens with dry carbon-rich material",
        context="someone is trying to keep a backyard compost pile active",
        product="garden composting",
        analogy="The pile acts less like a bonfire than like a breathing workshop. It needs food, air, and spacing to stay lively.",
    ),
    Topic(
        slug="bike-brakes",
        title="When bicycle brakes feel soft",
        subject="bicycle braking feel",
        issue="a brake lever pulls farther than usual before the bike slows confidently",
        mechanism="pad wear, cable stretch, rotor contamination, or trapped air can all widen the gap between hand motion and braking force",
        cue="the first clue is often a vague lever long before outright stopping power disappears",
        mistake="tightening one adjustment blindly and hoping every other variable follows",
        fix="checking pad thickness, cable or hose condition, and rotor cleanliness in a deliberate order",
        context="a commuter wants to know whether a soft brake is minor adjustment or a safety issue",
        product="bike maintenance",
        analogy="The system is like a conversation with too much static. The message still arrives, but more effort is needed before it lands cleanly.",
    ),
    Topic(
        slug="greenhouse-vent",
        title="Why a greenhouse overheats fast",
        subject="greenhouse ventilation",
        issue="a small greenhouse becomes much hotter than the outdoor air after only a short burst of sun",
        mechanism="clear panels admit sunlight easily while trapping heat unless warm air can escape and fresh air can replace it",
        cue="plants may wilt at noon even when the soil is still moist because leaf temperature outruns root supply",
        mistake="focusing only on watering when the air itself is the main stressor",
        fix="opening vents earlier, increasing crossflow, and using shade or thermal mass before the interior spikes",
        context="someone is surprised by how quickly protected plants can overheat",
        product="protected cultivation",
        analogy="The structure acts like a parked car with a better purpose. Light enters quickly, but heat leaves only if someone gives it a route.",
    ),
]

SCENES = [
    Scene(
        slug="harbor-clock",
        place="an old harbor where the public clock never quite keeps up with the day",
        routine="a local resident walks there each week to measure her mood against the water and the noise of the gulls",
        disruption="one small detail in the harbor shifts and refuses to fit the story she has told herself",
        image="the place feels older than the schedule board beside it and kinder than any timetable deserves",
        shift="the character realizes that the useful kind of certainty is not the same as precision",
        closing="By the time she leaves the quay, the delay has turned from insult into permission.",
    ),
    Scene(
        slug="night-train",
        place="a late train that turns the windows into moving mirrors",
        routine="a commuter keeps choosing the same seat because the reflection and the city make a softer picture together",
        disruption="an ordinary sound or passenger gesture jolts the ride out of routine and asks for attention",
        image="lights slide across the glass in bands that feel less like data and more like weather passing indoors",
        shift="the trip stops being a commute and becomes a place where the day can loosen its grip",
        closing="When the carriage reaches the river, the water finally looks willing to hold what she cannot keep carrying.",
    ),
    Scene(
        slug="bakery-window",
        place="a neighborhood bakery that opens before the street has decided to wake up",
        routine="the first employee arrives while the ovens still sound larger than the whole room",
        disruption="a familiar task slips out of sequence and exposes how much trust has been built into the place",
        image="flour hangs in the early light like stage dust waiting for the first line",
        shift="the character notices that care is often just repetition performed with full attention",
        closing="By sunrise, the trays in the window look less like inventory and more like a public apology for the dark hour before dawn.",
    ),
    Scene(
        slug="greenhouse-evening",
        place="a greenhouse after the visitors have gone home and only the fans still speak",
        routine="the caretaker makes a final loop to check leaves, vents, and the stubborn corner that traps heat",
        disruption="one plant responds differently than expected and turns the inspection into a conversation with memory",
        image="the glass holds the last light long enough to make every stem seem taller than it really is",
        shift="the room becomes a lesson in how quietly a place can ask for correction",
        closing="When the latch clicks behind him, the air outside feels simpler but not wiser.",
    ),
    Scene(
        slug="library-stairwell",
        place="a public library stairwell where footsteps arrive before faces do",
        routine="a shelving clerk uses the echo of the stairs to guess who is coming down before the turn of the landing",
        disruption="a stranger pauses in the middle of the stairwell and changes the sound of the whole morning",
        image="dust on the banister catches the light like a patient record of other hands",
        shift="the building reveals that silence is rarely empty; it is only distributed differently from room to room",
        closing="At closing time the echo remains, but it sounds less like loneliness and more like proof that the walls were listening.",
    ),
    Scene(
        slug="repair-bench",
        place="a small repair bench behind a hardware shop",
        routine="the technician spends each afternoon turning damaged objects slowly enough for them to explain themselves",
        disruption="one repair arrives that should be simple but somehow keeps resisting the usual order of steps",
        image="the bench lamp makes every scratch look like a sentence that almost wanted to be read",
        shift="the work stops being about the object and becomes about the patience required to meet it honestly",
        closing="When the light clicks off, the unfinished job no longer feels like failure, only like tomorrow arriving early.",
    ),
]

ADJECTIVES = [
    "amber", "brisk", "cedar", "clear", "copper", "gentle", "glossed", "granite",
    "ink-dark", "maple", "measured", "quiet", "river", "satin", "silver", "slate",
    "soft", "steady", "stone", "weathered",
]
NOUNS = [
    "arc", "bench", "bridge", "curve", "field", "frame", "horizon", "index",
    "lantern", "notebook", "pattern", "pivot", "relay", "signal", "thread", "valve",
    "window", "workshop", "marker", "harbor",
]
PLACES = [
    "the harbor clinic", "a north-facing balcony", "the tram workshop", "the station office",
    "a riverside kitchen", "the seed room", "the weather desk", "a quiet greenhouse",
    "the tool shed", "the ferry terminal", "a repair counter", "the west stairwell",
    "the library annex", "a market stall", "the campus lab", "a warm pantry",
]
TOOLS = [
    "checklist", "field note", "flashlight", "glass jar", "guide rail", "logbook",
    "meter", "mug", "notepad", "sample tray", "sensor", "spool", "thermometer", "timer",
]
RHYTHMS = [
    "before breakfast", "by late afternoon", "during the first heat of the day", "near closing time",
    "once the room goes quiet", "right after the air changes", "when the crowd thins out", "while the glass is still cool",
]
VERBS = [
    "drifts", "gathers", "holds", "leans", "lingers", "opens", "settles", "slides",
    "softens", "steadies", "turns", "waits",
]
MATERIALS = [
    "canvas", "clay", "glass", "linen", "oak", "paper", "salt", "steel",
]
OBSERVERS = [
    "Mara", "Lena", "Theo", "Asha", "Jonah", "Clara", "Noah", "Mira",
    "Felix", "Daria", "Leah", "Owen",
]
REFLECTIVE_OPENS = [
    "The cleanest explanation starts by naming the physical tradeoff.",
    "A useful description gets better as soon as it stops reaching for drama.",
    "People usually understand the problem faster when the mechanism is kept plain.",
    "The practical version of the story begins with what changes first and why.",
]
REFLECTIVE_CLOSES = [
    "That is why a small, patient check often teaches more than a dramatic intervention.",
    "The lesson is less about one object than about the order in which attention should move.",
    "Once that pattern is visible, the next decision becomes noticeably calmer.",
    "From there the problem turns from a vague worry into a sequence of testable steps.",
]
CHECKLIST_LEADS = [
    "A short field check helps more than another argument about theory.",
    "The next useful move is rarely bigger; it is usually clearer.",
    "Most confusion fades once the checkpoints are arranged in the right order.",
    "The situation improves when the signs are sorted into normal, useful, and urgent.",
]
STORY_NAMES = [
    "Mara", "Lena", "Ivo", "Sana", "Theo", "Nadia", "Milan", "Asha",
    "Noah", "Elise", "Rafi", "Clara", "Jonah", "Mira", "Tomas", "Leah",
]
STORY_OBJECTS = [
    "a folded ticket", "a chipped cup", "a brass key", "a rain-dark notebook",
    "a jar of screws", "a paper bag of bread", "a greenhouse latch", "a library card",
    "an empty seed packet", "a dull flashlight", "a receipt tucked into a sleeve", "a wool scarf",
]
STORY_GESTURES = [
    "turned the object over twice before deciding what it meant",
    "paused long enough for the room to speak first",
    "looked toward the window as if the glass might answer back",
    "straightened a stack that did not really need straightening",
    "counted the familiar sounds until they rearranged themselves",
    "waited for the feeling to name itself instead of rushing it",
]
ESSAY_THEMES = [
    "why small routines feel larger in shared spaces",
    "how quiet public places teach attention without ever using the word",
    "why imperfect tools often invite better observation",
    "how waiting changes the shape of ordinary time",
    "why a repeated walk can become a private form of study",
    "how maintenance becomes a language of care",
]
PRODUCT_HINTS = [
    "desk setup", "greenhouse bench", "repair kit", "kitchen shelf",
    "balcony planter", "tool drawer", "weather station", "station counter",
]
CODE_ADJECTIVES = [
    "alder", "amber", "basalt", "brisk", "cedar", "cinder", "clear", "cobalt",
    "ember", "fern", "glossed", "granite", "harbor", "lattice", "maple", "meadow",
    "mellow", "north", "orchard", "quartz", "satin", "silver", "stone", "velvet",
]
CODE_OBJECTS = [
    "anchor", "beacon", "bracket", "catalog", "compass", "diary", "engine", "folio",
    "gauge", "handle", "hinge", "index", "journal", "ladder", "marker", "notebook",
    "parcel", "relay", "shelf", "signal", "spindle", "thread", "valve", "window",
]
CODE_SITES = [
    "annex", "arboretum", "balcony", "basement", "causeway", "courtyard", "dock", "gallery",
    "garden", "greenhouse", "harbor", "kitchen", "landing", "library", "market", "observatory",
    "pantry", "platform", "repair-bay", "seed-room", "stairwell", "station", "terrace", "workbench",
]
CODE_MOTIONS = [
    "bending", "circling", "drifting", "easing", "folding", "gliding", "holding", "leaning",
    "lifting", "lingering", "opening", "resting", "settling", "shifting", "sliding", "softening",
    "steadying", "threading", "tilting", "turning", "waiting", "warming", "weaving", "widening",
]
CODE_MATERIALS = [
    "brick", "bronze", "canvas", "copper", "cotton", "glass", "granite", "linen",
    "marble", "oak", "paper", "pine", "plaster", "rope", "salt", "sandstone",
    "silk", "slate", "steel", "stone", "timber", "tin", "water", "wool",
]

SOURCE_TARGETS = {
    "books_expository": 3_000_000,
    "technical_notes": 3_000_000,
    "reference_articles": 4_000_000,
    "grounded_forum_qa": 4_000_000,
    "support_knowledge_qa": 3_500_000,
    "synthetic_grounded_dialogue": 4_000_000,
    "explanatory_chat_sessions": 3_000_000,
    "long_form_stories": 3_000_000,
    "essays_and_reflections": 2_000_000,
}


def pick(items: list[str] | tuple[str, ...], index: int, *, step: int = 1, offset: int = 0) -> str:
    return items[(index * step + offset) % len(items)]


def make_signature(index: int) -> dict[str, str]:
    return {
        "marker": f"{pick(ADJECTIVES, index, step=3)} {pick(NOUNS, index, step=5)}",
        "place": pick(PLACES, index, step=7),
        "tool": pick(TOOLS, index, step=11),
        "rhythm": pick(RHYTHMS, index, step=13),
        "verb": pick(VERBS, index, step=17),
        "material": pick(MATERIALS, index, step=19),
        "observer": pick(OBSERVERS, index, step=23),
        "product_hint": pick(PRODUCT_HINTS, index, step=29),
    }


def make_code(index: int, salt: int) -> dict[str, str]:
    value = index + (salt * 10_000)
    banks = [CODE_ADJECTIVES, CODE_OBJECTS, CODE_SITES, CODE_MOTIONS, CODE_MATERIALS]
    values = []
    for bank in banks:
        value, remainder = divmod(value, len(bank))
        values.append(bank[remainder])
    return {
        "adjective": values[0],
        "object": values[1],
        "site": values[2],
        "motion": values[3],
        "material": values[4],
        "phrase": " ".join(values),
    }


def normalize_text(text: str, width: int = 88) -> str:
    return textwrap.fill(" ".join(text.split()), width=width)


def compose_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(normalize_text(paragraph) for paragraph in paragraphs if paragraph.strip())


def book_document(index: int) -> tuple[str, str]:
    topic = TOPICS[index % len(TOPICS)]
    sig = make_signature(index)
    code = make_code(index, 1)
    paragraphs = [
        (
            f"{pick(REFLECTIVE_OPENS, index)} {topic.subject.capitalize()} feels confusing whenever {topic.issue}, "
            f"yet the pattern sharpens as soon as we track what changes in {sig['place']} {sig['rhythm']}. "
            f"{topic.mechanism.capitalize()} {topic.analogy}"
        ),
        (
            f"{sig['observer']} keeps a {sig['tool']} near {sig['place']} because the first clue is often small. "
            f"{topic.cue.capitalize()} When people miss that cue, they usually drift toward {topic.mistake}. "
            f"The result is extra effort without better information."
        ),
        (
            f"A steadier approach is to begin with the simplest observable contrast. Ask what happens before the system {sig['verb']}, "
            f"what changes after it does, and whether the shift is gradual or abrupt. {topic.fix.capitalize()} "
            f"That sequence turns uncertainty into a manageable inspection rather than a guessing contest."
        ),
        (
            f"In practical work, the case that teaches the most is rarely the dramatic failure. It is the ordinary day when {topic.context}, "
            f"the air feels ordinary, and the object still reveals its habits. Those mild cases show where the mechanism is honest and where the explanation needs revision."
        ),
        (
            f"People often reach for a single villain because it is emotionally tidy. They blame heat, water, software, or luck as though one label can end the story. "
            f"{topic.subject.capitalize()} is usually less theatrical. The mechanism threads through material limits, timing, and the small conditions that pile up around them."
        ),
        (
            f"{pick(CHECKLIST_LEADS, index)} First notice the baseline. Then compare it with the moment that feels wrong. Finally ask whether the difference points to normal load, deferred maintenance, or a genuine fault. "
            f"That order protects attention from the loudest explanation in the room."
        ),
        (
            f"The {sig['marker']} notes from {sig['place']} all point in the same direction: good maintenance does not begin by winning an argument. "
            f"It begins by narrowing the field of causes until the object starts to explain itself. From there, a modest correction often matters more than a dramatic replacement."
        ),
        (
            f"One recurring case in the archive is tagged {code['phrase']}. In that scenario the observer studies a {code['adjective']} {code['object']} "
            f"near the {code['site']}, notices the system {code['motion']}, and compares the change against a {code['material']} reference surface. "
            f"The point of the tag is not mystery. It is simply a way to keep this example distinct from the other ordinary cases that resemble it at first glance."
        ),
        (
            f"{pick(REFLECTIVE_CLOSES, index)} In that sense, {topic.subject} becomes a lesson in pace. "
            f"It rewards anyone willing to watch one change at a time and distrust the temptation to solve the whole problem with a single loud story."
        ),
    ]
    title = f"{topic.title}: {sig['marker'].title()} notes from {sig['place']} [{code['adjective']} {code['object']}]"
    return title, compose_paragraphs(paragraphs)


def technical_note(index: int) -> tuple[str, str]:
    topic = TOPICS[index % len(TOPICS)]
    sig = make_signature(index + 97)
    code = make_code(index, 2)
    overview = compose_paragraphs(
        [
            (
                f"{pick(REFLECTIVE_OPENS, index + 1)} For {topic.subject}, the useful starting point is still the same: {topic.mechanism}. "
                f"In the {sig['marker']} case observed around {sig['place']}, the most reliable cue was that {topic.cue}."
            ),
            (
                f"The common mistake is {topic.mistake}. A calmer fix is {topic.fix}. "
                f"That recommendation sounds modest, but it works because it forces the observer to compare changes in order instead of reacting to the loudest symptom."
            ),
            (
                f"{topic.analogy} The note that matters is rarely the dramatic one. It is the repeatable pattern that survives different weather, different timing, and different moods."
            ),
            (
                f"This memo is filed under the {code['phrase']} marker so it can be compared with similar observations without collapsing into them. "
                f"The label points to a {code['adjective']} {code['object']} near the {code['site']}, a system that keeps {code['motion']}, and a {code['material']} surface used as the visual baseline."
            ),
        ]
    )
    checklist = "\n".join(
        [
            "- Confirm the baseline before changing anything.",
            f"- Recheck the system {sig['rhythm']} rather than at a random moment.",
            f"- Write down what the {sig['tool']} shows before you interpret it.",
            f"- If the first fix fails, return to {topic.mechanism} before adding more variables.",
        ]
    )
    closing_tail = normalize_text(
        f"The {sig['material']} {sig['tool']} kept in {sig['place']} matters less than the order of observation it encourages."
    )
    note = (
        f"# {topic.title}: field memo\n\n"
        f"## Working picture\n\n{overview}\n\n"
        f"## Checklist\n\n{checklist}\n\n"
        f"## Closing note\n\n"
        f"{normalize_text(pick(REFLECTIVE_CLOSES, index + 4))} "
        f"{closing_tail}"
    )
    title = f"{topic.title}: field memo [{code['adjective']} {code['object']}]"
    return title, note


def reference_article(index: int) -> dict[str, str]:
    topic = TOPICS[index % len(TOPICS)]
    sig = make_signature(index + 211)
    code = make_code(index, 3)
    title = f"{topic.title}: a practical explainer [{code['adjective']} {code['object']}]"
    text = compose_paragraphs(
        [
            (
                f"{topic.subject.capitalize()} becomes clearer when we replace slogans with sequence. {topic.mechanism.capitalize()} "
                f"The example from {sig['place']} is useful because {topic.context} and the first visible clue is that {topic.cue}."
            ),
            (
                f"Misreadings usually begin with {topic.mistake}. That is attractive because it feels decisive, but it often hides the order of events. "
                f"The steadier correction is {topic.fix}. Once that sequence is in place, later decisions stop feeling improvised."
            ),
            (
                f"{topic.analogy} In practice, the {sig['marker']} marker is simply a reminder to watch the transition instead of only the outcome. "
                f"When the transition is clear, the diagnosis gets smaller and more honest."
            ),
            (
                f"Editors keep a separate note for the {code['phrase']} case because a {code['adjective']} {code['object']} observed near the {code['site']} "
                f"while the system is {code['motion']} does not behave exactly like the neighboring examples. That extra wording helps the archive preserve useful differences instead of smoothing them away."
            ),
            (
                f"{pick(REFLECTIVE_CLOSES, index + 2)} The goal is not a perfect theory for every case. "
                f"It is a reusable frame that still holds when a new room, a new tool, or a new season changes the details."
            ),
        ]
    )
    return {
        "id": f"ref-{index:05d}",
        "title": title,
        "text": text,
        "topic": topic.slug,
    }


def forum_qa(index: int) -> dict[str, str]:
    topic = TOPICS[index % len(TOPICS)]
    sig = make_signature(index + 307)
    code = make_code(index, 4)
    question = (
        f"In {sig['place']}, {topic.context}. The notes for this thread use the tag {code['phrase']}. "
        f"What is the clearest way to think about why this happens before I start changing everything?"
    )
    answer = compose_paragraphs(
        [
            (
                f"The useful answer begins with mechanism, not panic. {topic.mechanism.capitalize()} "
                f"That is why {topic.cue}, and it is also why {topic.mistake} usually wastes time."
            ),
            (
                f"A better next step is {topic.fix}. Keep the first round of checks small enough that you can still tell which change actually mattered. "
                f"Once that order is visible, the situation normally feels much less mysterious."
            ),
            (
                f"In the tagged {code['adjective']} {code['object']} case, the comparison point is a {code['material']} surface near the {code['site']} while the system is {code['motion']}. "
                f"That detail matters because it keeps this thread grounded in a specific observation rather than in a generic fear."
            ),
        ]
    )
    return {
        "id": f"forum-{index:05d}",
        "thread_id": f"{topic.slug}-{index // 3:04d}",
        "topic": topic.slug,
        "context": normalize_text(
            f"A person is working around {sig['place']} {sig['rhythm']} and notices that {topic.context}. "
            f"They have a {sig['tool']} nearby but are unsure which signal deserves trust first. "
            f"Their notebook labels the case {code['phrase']} so it can be compared with similar reports."
        ),
        "question": normalize_text(question),
        "answer": answer,
    }


def support_qa(index: int) -> dict[str, str]:
    topic = TOPICS[index % len(TOPICS)]
    sig = make_signature(index + 401)
    code = make_code(index, 5)
    return {
        "id": f"support-{index:05d}",
        "article_id": f"{topic.slug}-{index // 2:04d}",
        "product": topic.product,
        "context": normalize_text(
            f"A support article is being drafted for a {sig['product_hint']} where {topic.context}. "
            f"The writer wants an explanation that is calm, specific, and easy to act on. "
            f"The working draft calls this the {code['phrase']} scenario."
        ),
        "question": normalize_text(
            f"What should the article say when users ask about {topic.issue} in the {code['adjective']} {code['object']} scenario?"
        ),
        "answer": compose_paragraphs(
            [
                (
                    f"Start by explaining that {topic.mechanism}. Readers should know that {topic.cue} "
                    f"and that the most common unhelpful response is {topic.mistake}."
                ),
                (
                    f"Then offer an action sequence: {topic.fix}. If those checks do not change the behavior, the reader has a clearer basis for escalation and less reason to guess."
                ),
                (
                    f"For the tagged case, note that the observation was anchored to a {code['material']} surface near the {code['site']} while the system kept {code['motion']}. "
                    f"Those details make the article specific enough to teach without pretending every room behaves the same way."
                ),
            ]
        ),
    }


def grounded_dialogue(index: int) -> dict[str, object]:
    topic = TOPICS[index % len(TOPICS)]
    sig = make_signature(index + 503)
    code = make_code(index, 6)
    return {
        "id": f"dialogue-{index:05d}",
        "conversation_id": f"{topic.slug}-{index // 2:04d}",
        "topic": topic.slug,
        "messages": [
            {
                "role": "user",
                "content": normalize_text(
                    f"I keep noticing that {topic.context} around {sig['place']} {sig['rhythm']}. "
                    f"My notes call it the {code['phrase']} case. Why does it seem to shift so quickly?"
                ),
            },
            {
                "role": "assistant",
                "content": normalize_text(
                    f"The short answer is mechanism. {topic.mechanism.capitalize()} "
                    f"That makes the change feel sudden even when the setup has been building for a while."
                ),
            },
            {
                "role": "user",
                "content": normalize_text(
                    f"So the first clue is not the whole story? I was mostly reacting to the moment when the {sig['marker']} signal became obvious near the {code['site']}."
                ),
            },
            {
                "role": "assistant",
                "content": normalize_text(
                    f"Right. {topic.cue.capitalize()} If you only react to the loudest symptom, it is easy to slide into {topic.mistake}."
                ),
            },
            {
                "role": "user",
                "content": normalize_text(
                    f"What would you check first if you wanted a calmer diagnosis and not just a quick guess?"
                ),
            },
            {
                "role": "assistant",
                "content": normalize_text(
                    f"I would start with {topic.fix}. In the {code['adjective']} {code['object']} case, I would also compare the behavior against a {code['material']} reference while the system is {code['motion']}. "
                    f"That keeps the variables small enough that the object can tell you what changed instead of making you invent a dramatic story."
                ),
            },
        ],
    }


def explanatory_chat(index: int) -> dict[str, object]:
    topic = TOPICS[index % len(TOPICS)]
    sig = make_signature(index + 601)
    code = make_code(index, 7)
    return {
        "id": f"session-{index:05d}",
        "session_id": f"{topic.slug}-{index // 2:04d}",
        "domain": topic.product,
        "turns": [
            {
                "speaker": "customer",
                "text": normalize_text(
                    f"I'm writing notes for a teammate because {topic.context}. The example came from {sig['place']}, the folder labels it {code['phrase']}, and I want to explain it without sounding dramatic."
                ),
            },
            {
                "speaker": "advisor",
                "text": normalize_text(
                    f"Lead with the mechanism: {topic.mechanism}. People trust the explanation more when they can trace what changes first."
                ),
            },
            {
                "speaker": "customer",
                "text": normalize_text(
                    f"Should I mention the warning signs separately? The part that stands out to me is that {topic.cue}."
                ),
            },
            {
                "speaker": "advisor",
                "text": normalize_text(
                    f"Yes. Separate normal signals from urgent ones, then name the common mistake as {topic.mistake}. That keeps the reader from overreacting or underreacting."
                ),
            },
            {
                "speaker": "customer",
                "text": normalize_text(
                    f"And the action section should probably stay short?"
                ),
            },
            {
                "speaker": "advisor",
                "text": normalize_text(
                    f"Short and ordered. Say {topic.fix}. In the tagged case, mention the {code['adjective']} {code['object']} near the {code['site']} and the fact that the system keeps {code['motion']} against a {code['material']} baseline. "
                    f"Once the steps are in sequence, the reader has something usable instead of a cloud of warnings."
                ),
            },
        ],
    }


def story_document(index: int) -> tuple[str, str]:
    scene = SCENES[index % len(SCENES)]
    sig = make_signature(index + 701)
    code = make_code(index, 8)
    name = pick(STORY_NAMES, index, step=3)
    companion = pick(STORY_NAMES, index + 5, step=5)
    obj = pick(STORY_OBJECTS, index, step=7)
    gesture = pick(STORY_GESTURES, index, step=11)
    paragraphs = [
        (
            f"{name} returned to {scene.place} because {scene.routine}. The habit had outlived any single reason for keeping it, "
            f"but that was part of its usefulness. The same route, the same small sounds, and the same worn surfaces kept offering a scale for the day."
        ),
        (
            f"That evening {name} carried {obj} with the careless concentration people reserve for objects they do not yet understand. "
            f"{scene.image} {name} {gesture}, as if the place might explain the object simply by being itself."
        ),
        (
            f"Then {scene.disruption}. It was not large enough to count as drama, yet it tugged at every earlier assumption. "
            f"{companion} would have called it a trivial interruption, but the room had already changed shape around it."
        ),
        (
            f"The first response was practical. Check the latch, move the cup, reread the note, start the route again. None of that settled the feeling. "
            f"What unsettled {name} was not danger. It was the sense that the ordinary script had become too small for the moment it was trying to hold."
        ),
        (
            f"So {name} stayed still long enough for the quieter details to return. A breeze moved through the space with the patience of a clerk sorting papers. "
            f"A distant sound repeated until it felt intentional. Even the {sig['marker']} memory tied to {sig['place']} surfaced with a steadier outline. "
            f"In that moment the place felt marked by the private code {code['phrase']}, as if the room had filed its own version of the evening."
        ),
        (
            f"{scene.shift} The change was modest, almost embarrassing in its simplicity, but it rearranged the whole evening. "
            f"The object in hand remained the same object. The place remained the same place. The meaning of standing there had shifted a few degrees and that was enough."
        ),
        (
            f"{scene.closing} {name} left by the side route, carrying {obj} more lightly than before, not because it had become easier to explain, "
            f"but because it no longer needed to be solved immediately."
        ),
    ]
    title = f"{scene.slug.replace('-', ' ').title()}: {sig['marker'].title()} evening [{code['adjective']} {code['object']}]"
    return title, compose_paragraphs(paragraphs)


def essay_document(index: int) -> tuple[str, str]:
    sig = make_signature(index + 809)
    code = make_code(index, 9)
    theme = pick(ESSAY_THEMES, index, step=5)
    topic = TOPICS[index % len(TOPICS)]
    title = f"On {theme} [{code['adjective']} {code['object']}]"
    intro = compose_paragraphs(
        [
            (
                f"People often talk about attention as if it belongs only to emergencies or masterpieces. Ordinary places disagree. "
                f"A {sig['marker']} moment in {sig['place']} can teach more about attention than a dramatic event because it arrives without demanding an immediate performance."
            ),
            (
                f"I keep returning to this while thinking about {topic.subject}. {topic.analogy} "
                f"The same habit appears in civic rooms, work benches, kitchens, and station platforms: the best understanding grows from repeated contact with an imperfect but legible system."
            ),
            (
                f"That is why {theme} matters. Repetition is often treated as the enemy of perception, yet repetition is what gives small differences somewhere to stand. "
                f"Without a known rhythm, every change feels equally loud and equally mysterious."
            ),
        ]
    )
    body = compose_paragraphs(
        [
            (
                f"When a place is visited often enough, it begins to produce its own grammar. The hinge complains at the same point in the swing. "
                f"The radiator clicks at the same minute after the heat starts. The librarian's cart reaches the landing just before the school crowd thins. "
                f"These are not grand revelations, but they teach scale."
            ),
            (
                f"Scale is what protects a person from melodrama. Without it, every warm battery, every wilted leaf, and every late train looks like a verdict. "
                f"With it, the same events become clues. The observer still cares, but care stops collapsing into panic."
            ),
            (
                f"I like places that leave room for this slower form of reading. They do not flatter the person who arrives. They simply continue being themselves until the visitor becomes accurate enough to notice. "
                f"In a culture that rewards instant explanation, that kind of accuracy can feel almost private. The private tag I keep returning to is {code['phrase']}, which is only another way of marking one case so it does not dissolve into every other one."
            ),
            (
                f"{pick(REFLECTIVE_CLOSES, index + 9)} The point is not nostalgia for worn tools or crooked clocks. "
                f"It is respect for environments that let meaning emerge through return, comparison, and the patience to see one more ordinary detail than yesterday. "
                f"Even a {code['adjective']} {code['object']} near the {code['site']} can become enough of a witness if the observer keeps returning while the room is {code['motion']}."
            ),
        ]
    )
    text = f"# {title}\n\n## Observation\n\n{intro}\n\n## Reflection\n\n{body}"
    return title, text


def document_chars(title: str, body: str) -> int:
    return len(title) + len(body)


def qa_chars(record: dict[str, str]) -> int:
    return len(record["context"]) + len(record["question"]) + len(record["answer"])


def dialogue_chars(record: dict[str, object], turns_key: str, text_key: str) -> int:
    turns = record[turns_key]
    assert isinstance(turns, list)
    return sum(len(str(turn[text_key])) for turn in turns if isinstance(turn, dict))


def has_data(path: Path) -> bool:
    if path.is_file():
        return path.exists() and path.stat().st_size > 0
    if path.is_dir():
        return path.exists() and any(child.is_file() for child in path.rglob("*"))
    return False


def reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def generate_text_dir(path: Path, target_chars: int, builder, extension: str) -> tuple[int, int]:
    path.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    created = 0
    while total_chars < target_chars:
        title, body = builder(created)
        file_path = path / f"{created:05d}{extension}"
        file_path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
        total_chars += document_chars(title, body)
        created += 1
    return created, total_chars


def generate_jsonl(path: Path, target_chars: int, builder, counter) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    created = 0
    with path.open("w", encoding="utf-8") as handle:
        while total_chars < target_chars:
            record = builder(created)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_chars += counter(record)
            created += 1
    return created, total_chars


def generate_csv(path: Path, target_chars: int, builder, fieldnames: list[str], counter) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    created = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        while total_chars < target_chars:
            record = builder(created)
            writer.writerow(record)
            total_chars += counter(record)
            created += 1
    return created, total_chars


def bootstrap(repo_root: Path, force: bool) -> None:
    data_root = repo_root / "data" / "raw"
    marker_path = data_root / ".mono_lm_bootstrap_large.json"
    actions = [
        ("books_expository", data_root / "expository" / "books"),
        ("technical_notes", data_root / "expository" / "technical_notes"),
        ("reference_articles", data_root / "expository" / "reference_articles.jsonl"),
        ("grounded_forum_qa", data_root / "qa" / "grounded_forum_qa.jsonl"),
        ("support_knowledge_qa", data_root / "qa" / "support_knowledge.csv"),
        ("synthetic_grounded_dialogue", data_root / "dialogue" / "synthetic_grounded_dialogue.jsonl"),
        ("explanatory_chat_sessions", data_root / "dialogue" / "explanatory_chat_sessions.jsonl"),
        ("long_form_stories", data_root / "prose" / "stories"),
        ("essays_and_reflections", data_root / "prose" / "essays"),
    ]
    created_labels: list[str] = []

    for _, path in actions:
        if force and path.exists():
            reset_path(path)

    for label, path in actions:
        if has_data(path):
            print(f"[skip] {label}: {path} already exists")
            continue
        target = SOURCE_TARGETS[label]
        if label == "books_expository":
            created, chars = generate_text_dir(path, target, book_document, ".txt")
        elif label == "technical_notes":
            created, chars = generate_text_dir(path, target, technical_note, ".md")
        elif label == "reference_articles":
            created, chars = generate_jsonl(path, target, reference_article, lambda row: len(row["title"]) + len(row["text"]))
        elif label == "grounded_forum_qa":
            created, chars = generate_jsonl(path, target, forum_qa, qa_chars)
        elif label == "support_knowledge_qa":
            created, chars = generate_csv(
                path,
                target,
                support_qa,
                ["id", "article_id", "product", "context", "question", "answer"],
                qa_chars,
            )
        elif label == "synthetic_grounded_dialogue":
            created, chars = generate_jsonl(path, target, grounded_dialogue, lambda row: dialogue_chars(row, "messages", "content"))
        elif label == "explanatory_chat_sessions":
            created, chars = generate_jsonl(path, target, explanatory_chat, lambda row: dialogue_chars(row, "turns", "text"))
        elif label == "long_form_stories":
            created, chars = generate_text_dir(path, target, story_document, ".txt")
        elif label == "essays_and_reflections":
            created, chars = generate_text_dir(path, target, essay_document, ".md")
        else:
            raise ValueError(f"Unhandled bootstrap source: {label}")
        created_labels.append(label)
        print(f"[created] {label}: {created} items / {chars} chars")

    if created_labels:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps(
                {
                    "mode": "synthetic_bootstrap",
                    "generated_sources": created_labels,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the raw large corpus for mono-lm.")
    parser.add_argument("--force", action="store_true", help="Regenerate the bootstrap corpus even if source paths already exist")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    bootstrap(repo_root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
