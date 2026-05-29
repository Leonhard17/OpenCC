"""Command-line entry point: ``constitutional-classifier``.

    constitutional-classifier serve [--config config.yaml] [--host H] [--port P]
    constitutional-classifier check "<text>" [--config config.yaml] [--mode block|annotate]
"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import Decision, Pipeline, PipelineConfig


def _load_config(path: str | None) -> PipelineConfig:
    return PipelineConfig.from_yaml(path) if path else PipelineConfig()


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    config = _load_config(args.config)
    host = args.host or config.host
    port = args.port or config.port
    print(f"Starting OpenCC pipeline server on http://{host}:{port} ...", file=sys.stderr)
    uvicorn.run(create_app(config), host=host, port=port)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    mode = Decision(args.mode) if args.mode else None
    result = Pipeline(config).check(args.text, mode=mode)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(prog="constitutional-classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the local HTTP service.")
    p_serve.add_argument("--config", default=None, help="Path to a pipeline YAML config.")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=_cmd_serve)

    p_check = sub.add_parser("check", help="Moderate one input and print the JSON result.")
    p_check.add_argument("text", help="The input text to moderate.")
    p_check.add_argument("--config", default=None, help="Path to a pipeline YAML config.")
    p_check.add_argument("--mode", choices=["block", "annotate"], default=None)
    p_check.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
