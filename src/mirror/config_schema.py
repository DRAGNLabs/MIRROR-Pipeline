"""Generate JSON Schemas for config files from the jsonargparse parsers.

The schemas let the VS Code Red Hat YAML extension validate config files as you
edit them. Validation is intentionally top-level only: it catches unknown keys
(typos), wrong scalar types, and bad enum values. It does not validate the
`init_args` of polymorphic `class_path` subclasses, which are open-ended.

Regenerate with: python -m mirror.config_schema
"""
import json
import types
import typing
from pathlib import Path

from jsonargparse import ArgumentParser

from mirror.cli_parsers import build_parser

SUBCOMMANDS = ("fit", "format", "eval", "infer")
_SCHEMAS_DIR = Path(__file__).parent.parent.parent / "schemas"
_SIMPLE = {int: "integer", float: "number", str: "string", bool: "boolean"}
_LEAF = "__action__"


def _nullable(schema: dict) -> dict:
    if isinstance(schema.get("type"), str):
        schema["type"] = [schema["type"], "null"]
    elif "enum" in schema and None not in schema["enum"]:
        schema["enum"].append(None)
    return schema


def _type_schema(hint) -> dict:
    """Map a type hint to a JSON Schema fragment; unknown/complex types allow anything."""
    if typing.get_origin(hint) is typing.Literal:
        return {"enum": list(typing.get_args(hint))}
    if typing.get_origin(hint) is typing.Union or isinstance(hint, types.UnionType):
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        nullable = len(non_none) != len(typing.get_args(hint))
        if len(non_none) == 1:
            sub = _type_schema(non_none[0])
            return _nullable(sub) if nullable else sub
        return {}
    return {"type": _SIMPLE[hint]} if hint in _SIMPLE else {}


def _action_schema(action) -> dict:
    if getattr(action, "choices", None):
        return {"enum": list(action.choices) + [None]}
    return _type_schema(getattr(action, "_typehint", None))


def _insert(tree: dict, parts: list[str], action) -> None:
    node = tree
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict) or _LEAF in node[part]:
            node[part] = {}
        node = node[part]
    node.setdefault(parts[-1], {_LEAF: action})


def _build_props(node: dict) -> dict:
    props = {}
    for name, child in node.items():
        if _LEAF in child:
            props[name] = _action_schema(child[_LEAF])
        else:
            props[name] = {"type": "object", "properties": _build_props(child), "additionalProperties": False}
    return props


def generate_schema(parser: ArgumentParser) -> dict:
    tree: dict = {}
    for action in parser._actions:
        dest = action.dest
        if dest in ("help", "print_config") or dest.endswith(".help"):
            continue
        _insert(tree, dest.split("."), action)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": _build_props(tree),
        "additionalProperties": False,
    }


def write_schemas(out_dir: Path = _SCHEMAS_DIR) -> None:
    out_dir.mkdir(exist_ok=True)
    for subcommand in SUBCOMMANDS:
        schema = generate_schema(build_parser(subcommand))
        (out_dir / f"{subcommand}.schema.json").write_text(json.dumps(schema, indent=2) + "\n")


if __name__ == "__main__":
    write_schemas()
