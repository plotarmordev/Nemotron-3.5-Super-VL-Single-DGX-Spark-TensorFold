"""The prompts tools/bench.py sends: those the README's Measured table ran, byte for byte.

The field log and the picture are generated from fixed seeds and literals; the PNG uses zlib's stored mode, so no
zlib version changes its bytes.
"""
import base64
import random
import struct
import zlib

MAX_TOKENS = 256  # a reply's length unless its prompt sets one

STATIONS = ["Alder Ridge", "Brennan Weir", "Cold Spring", "Dunmore Lift", "East Fen", "Fallow Creek",
            "Garrow Basin", "Hollin Cut", "Ivel Reach", "Juniper Hill"]
CREW = ["Okafor", "Lindqvist", "Marchetti", "Tanaka", "Dubois", "Haddad", "Novak", "Reyes", "Sørensen", "Chen"]
EVENTS = [
    "the intake screen was found partly blocked by leaf litter and cleared by hand",
    "pump {p} tripped on overcurrent and was restarted after the bearing temperature fell below {t} degrees",
    "turbidity rose to {n} NTU after overnight rain and the coagulant dose was raised by {d} percent",
    "a contractor reported a weeping joint on the {m} mm main, which was clamped the same afternoon",
    "the chlorine residual at the outlet held at {c} mg/L throughout the shift",
    "telemetry from the reservoir float dropped out for {k} minutes; the radio link was reseated",
    "a resident called about low pressure on the upper streets; the valve at the hill junction was found {o} percent open",
    "the standby generator ran its weekly test for {k} minutes with no faults",
    "sludge was drawn from settling tank {p} and the blanket depth measured {r} metres afterwards",
    "the flow meter on the transfer main was compared against the portable unit and read {e} percent high",
]
NOTES = [
    "Readings were logged at the usual intervals.", "No complaints were received.",
    "The crew noted that the access track is rutted and will need grading before winter.",
    "Spare filters were counted and two boxes were reordered.",
    "The night operator asked that the alarm thresholds be reviewed at the next meeting.",
    "Algae were visible along the north margin of the reservoir but samples were clear.",
    "The site gate lock was replaced after it seized in the cold.",
]
LANGUAGES = {
    "en": "Write in English a practical explanation of how a neighborhood library can stay useful during a week-long "
          "power outage. Discuss communication, lending books and helping older residents. Use connected paragraphs, "
          "not a numbered list.",
    "es": "Escribe en español una explicación práctica de cómo una biblioteca de barrio puede seguir siendo útil "
          "durante un apagón de una semana. Habla de la comunicación, el préstamo de libros y la ayuda a los vecinos "
          "mayores. Usa párrafos completos, no una lista numerada.",
    "fr": "Rédige en français une explication pratique sur la manière dont une bibliothèque de quartier peut rester "
          "utile pendant une panne électrique qui dure une semaine. Parle de la communication, du prêt de livres et "
          "de l’aide aux personnes âgées. Utilise des paragraphes, pas une liste numérotée.",
    "de": "Erkläre auf Deutsch, wie eine Stadtteilbibliothek während eines einwöchigen Stromausfalls weiterhin "
          "nützlich sein kann. Beschreibe die Kommunikation, die Ausleihe von Büchern und die Unterstützung älterer "
          "Menschen. Schreibe zusammenhängende Absätze und keine nummerierte Liste.",
    "it": "Scrivi in italiano una spiegazione pratica di come una biblioteca di quartiere possa rimanere utile "
          "durante un’interruzione di corrente di una settimana. Parla della comunicazione, del prestito dei libri e "
          "dell’aiuto agli anziani. Usa paragrafi completi, non un elenco numerato.",
    "ja": "一週間の停電が続く間、地域の図書館が住民の役に立ち続けるための具体的な方法を日本語で説明してください。"
          "情報の伝達、本の貸し出し、高齢者への支援について述べてください。"
          "番号付きの箇条書きではなく、つながりのある段落で書いてください。",
}


def corpus() -> str:
    """A water authority's field log, about 19k tokens: the cold prompts are prefixes of it."""
    rng = random.Random(20261016)
    out = ["HARROW VALLEY WATER AUTHORITY — CONSOLIDATED FIELD LOG\n"]
    for i in range(140):
        day, month = 1 + i % 28, 1 + (i // 28) % 12
        station = rng.choice(STATIONS)
        lead, second = rng.sample(CREW, 2)
        out.append(f"\nEntry {i + 1}: {station}, 2025-{month:02d}-{day:02d}, shift lead {lead}, with {second}.\n")
        flow = rng.randint(180, 950)
        level = rng.uniform(2.1, 9.8)
        out.append(f"Flow averaged {flow} m3/h and the wet well level was {level:.2f} m at handover. ")
        for _ in range(rng.randint(2, 4)):
            event = rng.choice(EVENTS).format(
                p=rng.randint(1, 4), t=rng.randint(55, 80), n=rng.randint(3, 40), d=rng.randint(5, 30),
                m=rng.choice([150, 225, 300, 450]), c=f"{rng.uniform(0.4, 1.6):.2f}", k=rng.randint(3, 45),
                o=rng.randint(10, 90), r=f"{rng.uniform(0.3, 1.9):.1f}", e=f"{rng.uniform(0.5, 6.0):.1f}")
            out.append(f"During the shift {event}. ")
        out.append(rng.choice(NOTES) + "\n")
    return "".join(out)


def png(width: int, height: int) -> bytes:
    """A red square, a blue disc and a green triangle on a pale grey ground."""
    rows = []
    for y in range(height):
        row = bytearray([0])  # filter type None
        for x in range(width):
            pixel = (235, 235, 235)
            if 24 <= x < 104 and 40 <= y < 120:
                pixel = (210, 30, 30)
            elif (x - 160) ** 2 + (y - 80) ** 2 <= 40 ** 2:
                pixel = (30, 60, 200)
            elif 120 <= y < 200 and abs(x - 256) <= (y - 120) // 2:
                pixel = (30, 160, 60)
            row += bytes(pixel)
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 0)) \
        + chunk(b"IEND", b"")


def user(content) -> list:
    return [{"role": "user", "content": content}]


def prompts() -> dict:
    image = base64.b64encode(png(320, 208)).decode()
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}}]
    chosen = [
        {"name": "code", "messages": user(
            "Write a Python function that parses an ISO 8601 duration such as P3DT4H5M6S into total seconds, "
            "with type hints, a docstring and three unit tests.")},
        {"name": "prose", "messages": user(
            "Write a quiet short story about a lighthouse keeper who receives a letter addressed to someone "
            "who left the island forty years ago.")},
        {"name": "reasoning", "messages": user(
            "A train leaves at 09:40 and travels 214 km at 82 km/h, then waits 17 minutes, then travels 96 km "
            "at 64 km/h. At what time does it arrive? Work through it step by step and check the answer.")},
        {"name": "repetition", "messages": user(
            "Repeat the following line exactly thirty times, numbering each line: "
            "The quick brown fox jumps over the lazy dog near the riverbank.")},
        {"name": "count", "max_tokens": 400, "messages": user(
            "Count from 1 to 1000. Output every number, separated by single spaces. Do not abbreviate or use "
            "ellipses. No other text.")},
        {"name": "multilingual", "messages": user(
            "Translate this sentence into French, German, Spanish, Japanese and Hindi, then explain one "
            "grammatical difference between the German and Japanese versions: \"The library closes early on "
            "Fridays, so bring the books back by Thursday evening.\"")},
        {"name": "tool_json", "tools": tools, "messages": user(
            "What is the weather in Oslo and in Lima right now? Use the tool for each city, in celsius.")},
        {"name": "gpu_explain", "messages": user(
            "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.")},
        {"name": "image", "messages": user([
            {"type": "text", "text": "Describe every shape in this image: its colour, its position and its "
                                     "approximate size relative to the image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}])},
        *({"name": f"lang_{code}", "messages": user(text)} for code, text in LANGUAGES.items()),
    ]
    return {p["name"]: p for p in chosen}
