from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence


COMMANDS = {
    "client": "fedavg.client",
    "matrix": "fedavg.experiment_matrix",
    "server": "fedavg.server",
    "simulate": "fedavg.simulate",
}


def _usage() -> str:
    commands = ", ".join(sorted(COMMANDS))
    return (
        "usage: python -m fedavg <command> [args]\n\n"
        f"commands: {commands}\n\n"
        "examples:\n"
        "  python -m fedavg simulate --config configs/mnist_iid_b16_e1.yaml --clients 2\n"
        "  python -m fedavg server --config configs/pi_server.yaml\n"
        "  python -m fedavg client --config configs/pi_client.yaml --client-id pi0 --client-index 0"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return

    command = args.pop(0)
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"unknown command: {command}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)

    module = importlib.import_module(module_name)
    entrypoint = getattr(module, "main")
    main_func: Callable[[list[str]], None] = entrypoint
    main_func(args)


if __name__ == "__main__":
    main()