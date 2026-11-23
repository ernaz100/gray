#!/usr/bin/env python3
"""How much walking is in the cached windows, and does the text say so?

    python -m gray.census --out <folder>

For every 8 s window of the cache: how far the head gets from where it started (the largest
horizontal distance, "reach"), where it ends, how far it travels (summed over 1 s steps, so
swaying does not count), how much it turns and how much its height changes. For the text of
the window: whether it mentions walking, whether it names a place to go, and whether it says
the person stays put. Also writes navigation.npz, the windows to train a navigation model on. A destination is counted only when a verb of going is followed by
a direction word and a thing ("walks towards the trash can", "enters the boiler room"); "standing
in the kitchen" does not count. Writes census.json and prints the tables.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from gray import frames
from gray.features import CACHE, split
from gray.rectify import SEQUENCES

REACH = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, math.inf)  # metres
WALK = re.compile(r"\b(walk\w*|steps? (forward|backward|back|to|toward|towards|into|over|aside|sideways)|stepping|moves? (forward|backward|back|to|toward|towards|into|across|around|over|away)|"
                  r"moving (forward|backward|to|toward|towards|into|across|around)|goes|going|jog\w*|runs?|running|approach\w*|enters?|entering|leaves?|leaving|exits?|"
                  r"cross(es|ing)?|returns?|returning|heads? (to|toward|towards|over|into|back)|climbs?|climbing|descends?|upstairs|downstairs)\b", re.I)
# A destination: a verb of going, at most three words, a preposition of direction, a determiner and a noun phrase
# ("walks towards the trash can"), or a verb that takes the place directly ("enters the boiler room").
GOING = r"(?:walk\w*|goes|going|go|heads?|heading|moves?|moving|steps?|stepping|proceeds?|proceeding|runs?|running|jogs?|jogging|returns?|returning|comes?|coming|climbs?|climbing|crawls?|strides?)"
TOWARD = r"(?:to|toward|towards|into|onto|over to|up to|back to|out of|through|across|down|up|along|past|around|inside|outside|out to)"
THE = r"(?:the|a|an|his|her|their|another)"
PHRASE = r"((?:[a-z'-]+ ){0,3}?[a-z'-]+?)(?= (?:and|while|with|as|then|in|on|at|to|using|before|after|where|which|by|from|near|but|holding|carrying|until|of|for|she|he|c)\b|[,.;:]|$)"
DESTINATION = re.compile(rf"\b{GOING}\b(?: [a-z'-]+){{0,3}}? {TOWARD} {THE} {PHRASE}|\b(?:enters?|entering|approach\w*|leaves?|leaving|exits?|exiting|reach(?:es)? (?!for|out|up|down|into|his|her))\b(?: {THE})? ?{PHRASE}", re.I)
NOT_A_PLACE = {"left", "right", "side", "front", "back", "air", "direction", "other", "same", "opposite", "floor", "ground", "peer", "camera", "way", "hand", "hands", "foot", "feet",
               "leg", "legs", "body", "position", "step", "steps", "bit", "little", "few", "couple", "moment", "while", "walking", "standing", "it", "them", "her", "him", "first", "second",
               "halt", "stop", "peers", "colleague", "colleagues", "friend", "friends", "face", "person", "people", "man", "woman", "instructions", "instruction"}
ROOMS = {"kitchen", "room", "area", "hallway", "hall", "bedroom", "bathroom", "office", "backyard", "garage", "corridor", "balcony", "patio", "porch", "lounge", "garden", "yard", "lobby",
         "basement", "outside", "inside", "house", "deck", "pantry", "closet", "foyer", "entrance", "driveway", "lawn", "trail", "path", "street", "road", "park", "sidewalk", "pathway", "pavement"}


def destinations(text: str) -> list[tuple[str, bool]]:
    """Noun phrases the text says the person goes to, and whether each is a room or area (otherwise an object)."""
    found = []
    for match in DESTINATION.finditer(text.lower()):
        phrase = (match.group(1) or match.group(2) or "").strip()
        words = phrase.split()
        if not words or words[-1] in NOT_A_PLACE or words[0] in NOT_A_PLACE:
            continue
        found.append((phrase, words[-1] in ROOMS))  # the last word decides: "kitchen counter" is an object
    return found


STAY = re.compile(r"\b(in place|stands? still|standing still|while standing|is standing|stands (in|at|near|by|next)|sits?|sitting|seated|is lying|lies)\b", re.I)


def one(job: tuple[str, str, str]) -> dict | None:
    name, cache, videos = job
    folder = Path(cache) / name
    head = torch.from_numpy(np.load(folder / "head.npy").astype(np.float32))  # (W, 120, 9), anchored at the first token
    index = np.load(folder / "text_index.npy")
    sentences = [s["text"] for s in json.loads((folder / "sentences.json").read_text())]
    meta = json.loads((folder / "meta.json").read_text())
    flat = head[..., [0, 2]]
    distance = flat.norm(dim=-1)
    seconds = flat[:, list(range(0, 120, 15)) + [119]]
    text = [" ".join(sentences[i] for i in row if i >= 0) for row in index]
    found = [destinations(t) for t in text]
    return {
        "name": name, "location": meta["location"], "script": meta["script"],
        "layout": (Path(videos) / name / "objects" / "boxy" / "scene_objects.csv").exists(),
        "reach": distance.max(dim=1).values.numpy(), "end": distance[:, -1].numpy(),
        "travel": (seconds[:, 1:] - seconds[:, :-1]).norm(dim=-1).sum(dim=1).numpy(),
        "turn": (frames.heading(head[:, -1, 3:]).abs() * 180 / math.pi).numpy(),
        "drop": (head[:, 0, 1] - head[..., 1].min(dim=1).values).numpy(),
        "has_text": np.array([bool(t) for t in text]), "walk": np.array([bool(WALK.search(t)) for t in text]),
        "stay": np.array([bool(STAY.search(t)) for t in text]),
        "place": np.array([bool(d) for d in found]), "room": np.array([any(r for _p, r in d) for d in found]),
        "object": np.array([any(not r for _p, r in d) for d in found]), "text": text, "found": found,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--videos", type=Path, default=SEQUENCES)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--min-reach", type=float, default=1.0, help="navigation subset: windows that get this far (m) are all kept")
    parser.add_argument("--keep-staying", type=float, default=0.12, help="navigation subset: share of the other windows that is kept")
    parser.add_argument("--layout", type=Path, default=None, help="folder of gray/layout.py; with it the navigation subset keeps indoor windows only")
    parser.add_argument("--catalog", type=Path, default=Path("/u/enazaren/projects/STEER/data/nymeria_catalog.csv"), help="says per activity whether it is indoor, outdoor or mixed")
    parser.add_argument("--min-floor", type=float, default=0.8, help="indoor: at least this share of the recorded head path is over annotated floor")
    parser.add_argument("--limit", type=int, default=None, help="only the first recordings (a test)")
    args = parser.parse_args()

    names = sorted(p.parent.name for p in args.cache.glob("*/head.npy"))[: args.limit]
    with Pool(args.workers) as pool:
        rows = [r for r in pool.imap_unordered(one, [(n, str(args.cache), str(args.videos)) for n in names], chunksize=4) if r]
    rows.sort(key=lambda r: r["name"])
    keys = ("reach", "end", "travel", "turn", "drop", "has_text", "walk", "place", "room", "object", "stay")
    a = {k: np.concatenate([r[k] for r in rows]) for k in keys}
    text = [t for r in rows for t in r["text"]]
    found = [d for r in rows for d in r["found"]]
    script = np.array([r["script"] for r in rows for _ in r["text"]])
    layout = np.concatenate([np.full(len(r["text"]), r["layout"]) for r in rows])
    total = len(text)
    print(f"{total:,d} windows of 8 s from {len(rows)} recordings; {a['has_text'].mean():.0%} have text; "
          f"{layout.mean():.0%} are in recordings with annotated walls, floors and objects ({sum(r['layout'] for r in rows)} recordings)\n")

    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    print("                         " + "".join(f"{q:>8.0%}" for q in quantiles))
    for key, label in (("reach", "reach, m"), ("end", "end distance, m"), ("travel", "travel, m"), ("turn", "turn at the end, deg"), ("drop", "head drop, m")):
        print(f"  {label:23s}" + "".join(f"{np.quantile(a[key], q):8.2f}" for q in quantiles))

    rng = np.random.default_rng(0)
    table, examples = [], {}
    print("\n  share of the windows with text         names a       of which    of which  walking but")
    print("  reach             windows   share  destination     room or area   an object  no destination  says staying")
    for lo, hi in zip(REACH[:-1], REACH[1:]):
        mask = (a["reach"] >= lo) & (a["reach"] < hi)
        with_text = mask & a["has_text"]
        share = lambda key: float(a[key][with_text].mean()) if with_text.any() else float("nan")
        label = f"{lo:g} to {hi:g} m" if math.isfinite(hi) else f"over {lo:g} m"
        row = {"reach": label, "windows": int(mask.sum()), "share": float(mask.mean()), "walk": share("walk"), "place": share("place"), "room": share("room"),
               "object": share("object"), "walk_only": float((a["walk"] & ~a["place"])[with_text].mean()) if with_text.any() else float("nan"),
               "stay": share("stay"), "median_travel_m": float(np.median(a["travel"][mask])) if mask.any() else float("nan")}
        table.append(row)
        picked = rng.choice(np.flatnonzero(with_text), size=min(8, int(with_text.sum())), replace=False)
        examples[label] = [{"reach_m": float(a["reach"][i]), "travel_m": float(a["travel"][i]), "text": text[i]} for i in picked]
        print(f"  {label:15s} {row['windows']:9,d} {row['share']:7.1%} {row['place']:12.0%} {row['room']:16.0%} {row['object']:11.0%} {row['walk_only']:15.0%} {row['stay']:13.0%}")

    moving = a["reach"] >= 1.0
    print(f"\ntext against movement (windows with text): of those that mention walking, {moving[a['walk']].mean():.0%} reach 1 m or more; "
          f"of those that do not, {moving[a['has_text'] & ~a['walk']].mean():.0%} do")
    print(f"moving on the spot: {((a['reach'] < 1.0) & (a['travel'] >= 1.5)).mean():.1%} of windows travel 1.5 m or more but stay within 1 m")

    named = a["place"]
    print(f"destinations: {named.sum():,d} windows name one ({named[a['has_text']].mean():.0%} of those with text); {moving[named].mean():.0%} of them reach 1 m or more, "
          f"{(a['reach'] >= 2.0)[named].mean():.0%} reach 2 m or more; of the windows that reach 2 m or more and have text, {named[(a['reach'] >= 2.0) & a['has_text']].mean():.0%} name one")
    counts = {True: defaultdict(int), False: defaultdict(int)}
    for d in found:
        for phrase, room in set(d):
            counts[room][phrase] += 1
    top = {kind: sorted(counts[room].items(), key=lambda kv: -kv[1])[:40] for kind, room in (("rooms and areas", True), ("objects", False))}
    for kind, items in top.items():
        print(f"\nmost named {kind} (windows): " + ", ".join(f"{p} {n:,d}" for p, n in items))
    spans = [m.group(0) for i in rng.choice(np.flatnonzero(named), size=min(30, int(named.sum())), replace=False) for m in [DESTINATION.search(text[i].lower())] if m]
    print("\nmatched wording, 30 random windows: " + " | ".join(spans))

    by_script = defaultdict(list)
    for s, r in zip(script, a["reach"]):
        by_script[s].append(r)
    scripts = sorted(({"script": s, "windows": len(v), "median_reach_m": float(np.median(v)), "share_1m": float(np.mean(np.array(v) >= 1.0)),
                       "share_2m": float(np.mean(np.array(v) >= 2.0))} for s, v in by_script.items()), key=lambda r: -r["share_1m"])
    print("\n  activity script                       windows  median reach  reach 1 m+  reach 2 m+")
    for r in scripts:
        print(f"  {r['script'][:36]:36s} {r['windows']:8,d} {r['median_reach_m']:13.2f} {r['share_1m']:11.0%} {r['share_2m']:11.0%}")

    print("\nexamples (reach, travel, text):")
    for label, items in examples.items():
        print(f"  {label}")
        for item in items[:4]:
            print(f"    {item['reach_m']:.1f} m, {item['travel_m']:.1f} m: {item['text'][:230]}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "census.json").write_text(json.dumps({"windows": total, "recordings": len(rows), "by_reach": table, "by_script": scripts, "examples": examples, "destinations": {k: v for k, v in top.items()}, "matched_wording": spans,
                                                      "quantiles": {k: {str(q): float(np.quantile(a[k], q)) for q in quantiles} for k in ("reach", "end", "travel", "turn", "drop")}}, indent=1))
    np.savez_compressed(args.out / "windows.npz", **{k: a[k] for k in keys}, layout=layout,
                        sequence=np.array([r["name"] for r in rows for _ in r["text"]]), window=np.concatenate([np.arange(len(r["text"])) for r in rows]))

    # The navigation subset: every window that goes somewhere, and a random part of those that do not.
    going = a["reach"] >= args.min_reach
    kept = going | (np.random.default_rng(0).random(total) < args.keep_staying)
    sequence = np.array([r["name"] for r in rows for _ in r["text"]])
    window = np.concatenate([np.arange(len(r["text"])) for r in rows])
    floor = np.full(total, np.nan, dtype=np.float32)  # share of the head path over annotated floor; nan without a map
    if args.layout is not None:
        import csv

        with args.catalog.open() as handle:
            setting = {row["script"]: row["setting"] for row in csv.DictReader(handle)}
        where = np.array([setting.get(s, "unknown") for s in script])
        mapped = np.load(args.layout / "windows.npz")
        position = {pair: i for i, pair in enumerate(zip(sequence.tolist(), window.tolist()))}
        rows_of = np.array([position.get(pair, -1) for pair in zip(mapped["sequence"].tolist(), mapped["window"].tolist())])
        floor[rows_of[rows_of >= 0]] = mapped["floor"][rows_of >= 0]
        has_map = np.isfinite(floor)
        indoor = (where != "outdoor") & np.where(has_map, floor >= args.min_floor, where == "indoor")
        print(f"\nindoor windows only: of the {kept.sum():,d} windows picked, {(kept & (where == 'outdoor')).sum():,d} are from outdoor activities, "
              f"{(kept & (where != 'outdoor') & has_map & (floor < args.min_floor)).sum():,d} have a map but less than {args.min_floor:.0%} of the path over annotated floor, "
              f"{(kept & (where == 'mixed') & ~has_map).sum():,d} are from mixed activities without a map; {(kept & indoor).sum():,d} remain "
              f"({(kept & indoor & has_map).sum():,d} with a map, {(kept & indoor & ~has_map).sum():,d} from indoor activities without one)")
        kept &= indoor
    np.savez_compressed(args.out / "navigation.npz", sequence=sequence[kept], window=window[kept], going=going[kept], reach=a["reach"][kept], floor=floor[kept],
                        has_text=a["has_text"][kept], walk=a["walk"][kept], room=a["room"][kept], object=a["object"][kept],
                        destination=np.array(["; ".join(sorted({p for p, _r in found[i]})) for i in np.flatnonzero(kept)]))
    held = {item for item in split(args.cache, 0.15, 0.5)[1]} if (args.cache / "index.json").exists() else set()
    usable = {item for part in split(args.cache, 0.15, 0.5) for item in part} if held else set()
    print(f"\nnavigation subset: {kept.sum():,d} of {total:,d} windows; {(kept & going).sum():,d} get {args.min_reach:g} m or further, {(kept & ~going).sum():,d} "
          f"(from {args.keep_staying:.0%} of the rest) do not; {a['has_text'][kept].mean():.0%} have text, {a['place'][kept].mean():.0%} name a destination "
          f"({a['room'][kept].mean():.0%} a room or area, {a['object'][kept].mean():.0%} an object)")
    if held:
        pairs = list(zip(sequence[kept].tolist(), window[kept].tolist()))
        in_held = np.array([pair in held for pair in pairs])
        in_use = np.array([pair in usable for pair in pairs])
        print(f"  with the split used for training (every sixth location held out, recordings with a bad clock dropped): {(in_use & ~in_held).sum():,d} training windows, "
              f"{in_held.sum():,d} held-out windows, of which {(in_held & going[kept]).sum():,d} go somewhere and {(in_held & a['place'][kept]).sum():,d} name a destination")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
