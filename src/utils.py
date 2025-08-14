import argparse
import logging

from settings import (
    CMD_CHAT,
    CMD_EXPLORE,
    CMD_LOAD,
    RuntimeArgs,
)


def init_logger(level: int) -> None:
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=level,
    )


def get_runtime_args() -> RuntimeArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--log-level", type=int, default=20, help="The logging level to use."
    )
    parser.add_argument(
        "-t",
        "--template-path",
        type=str,
        default="templates",
        help="Path to the templates.",
    )
    parser.add_argument(
        "--ignore-graph", action="store_false", help="Do not use the knowledge graph."
    )

    action_subparser = parser.add_subparsers(dest="action")

    load_parser = action_subparser.add_parser(
        CMD_LOAD, help="Load PDF document in graph database."
    )
    load_parser.add_argument(
        "-s",
        "--slice",
        default=None,
        help="Which chunk slice to include (format: start:end).",
    )
    load_parser.add_argument("-p", "--path", default=None, help="Path to PDF file.")

    explore_parser = action_subparser.add_parser(
        CMD_EXPLORE, help="Explore PDF document."
    )
    explore_parser.add_argument(
        "-s",
        "--slice",
        default=None,
        help="Which chunk slice to include (format: start:end).",
    )
    explore_parser.add_argument("-p", "--path", default=None, help="Path to PDF file.")

    action_subparser.add_parser(CMD_CHAT, help="Chat with the LLM.")

    return parser.parse_args()


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()
