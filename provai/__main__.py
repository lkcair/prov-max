import argparse
import importlib
from pathlib import Path

from .graph import check


def main():
    ap = argparse.ArgumentParser(prog="python -m provai")
    sub = ap.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert", help="convert captured runs into PROV-AI records")
    conv.add_argument("framework")
    conv.add_argument("runs", nargs="+")
    ev = sub.add_parser("evaluate", help="ask every competency question of every run, spans only and fully integrated")
    ev.add_argument("frameworks", nargs="*")
    hm = sub.add_parser("hermit", help="export the OWL file and compare HermiT with the OWL 2 RL closure on runs")
    hm.add_argument("runs", nargs="+")
    args = ap.parse_args()
    if args.command == "evaluate":
        from .evaluate import evaluate, show
        show(evaluate(args.frameworks))
        return 0
    if args.command == "hermit":
        from .hermit import compare, export
        print(f"prov-ai.owl written, isomorphic to the Turtle: {export()}")
        for run in args.runs:
            report, n_hermit, n_rl, only_hermit, only_rl = compare(Path(run))
            print(f"{run}: {'; '.join(report)}; inferred facts HermiT {n_hermit}, OWL 2 RL {n_rl}, "
                  f"only HermiT {len(only_hermit)}, only OWL 2 RL {len(only_rl)}")
            for d in only_hermit[:5] + only_rl[:5]:
                print("  ", *d)
        return 0
    converter = importlib.import_module(f"provai.converters.{args.framework}")
    for run in args.runs:
        rec = converter.convert(Path(run))
        (Path(run) / "record.ttl").write_text(rec.turtle())
        ok, messages = check(rec.g)
        print(f"{run}: {len(rec.g)} triples, {'conforms' if ok else 'does not conform'}")
        for m in sorted(set(messages)):
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
