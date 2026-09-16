"""Physical input dimensions and diagnostic-only fixes for verified upstream launchers."""

import ast
from pathlib import Path
import re

import yaml

from .data import input_features


def background_diagnostics(tree, features):
    # Upstream samples omit mass; its diagnostic loop assumes an extra column.
    loops = [node for node in tree.body if isinstance(node, ast.For)
             and ast.dump(node.target) == ast.dump(ast.Name(id="i", ctx=ast.Store()))
             and ast.dump(node.iter) == ast.dump(ast.parse("range(5)", mode="eval").body)]
    if len(loops) != 1 or features not in (4, 5):
        raise ValueError("Unrecognized upstream sideband diagnostic loop")
    loop = loops[0]
    expected = ast.dump(ast.parse("x_samples[:, i]", mode="eval").body)
    samples = [node for node in ast.walk(loop)
               if isinstance(node, ast.Subscript) and ast.dump(node) == expected]
    if len(samples) != 1:
        raise ValueError("Unrecognized upstream sideband diagnostic sample column")
    loop.iter = ast.parse(f"range(1, {features + 1})", mode="eval").body
    samples[0].slice = ast.parse("x_samples[:, i - 1]", mode="eval").body.slice
    return ast.fix_missing_locations(tree)


def model_config(sources, destination, variant):
    input_features(variant)
    original = Path(sources) / "scripts/DE_MAF_model.yml"
    if variant != "deltaR":
        return original
    text = original.read_text()
    config = yaml.safe_load(text)
    if config.get("num_inputs") != 4 or config.get("num_cond_inputs") != 1:
        raise ValueError("Unexpected upstream R-ANODE background dimensions")
    updated, count = re.subn(r"(?m)^num_inputs: 4[ \t]*$", "num_inputs: 5", text)
    if count != 1 or yaml.safe_load(updated) != {**config, "num_inputs": 5}:
        raise ValueError("R-ANODE DeltaR may change only background num_inputs")
    destination = Path(destination)
    destination.write_text(updated)
    return destination


def extend_delta_r(tree, script_name):
    if script_name == "nflows_CR_data.py":
        replacements = [("range(1, 5, 1)", "range(1, 6, 1)", 3)]
    elif script_name == "r_anode.py":
        replacements = [
            ("flows_model_RQS(device=device, num_features=5, context_features=None)",
             "flows_model_RQS(device=device, num_features=6, context_features=None)", 1),
            ("x_samples.reshape(-1, 5)", "x_samples.reshape(-1, 6)", 1),
            ("samples.reshape(-1, 5)", "samples.reshape(-1, 6)", 1),
            ("range(0, 5)", "range(0, 6)", 1),
        ]
    else:
        raise ValueError("Unrecognized upstream R-ANODE launcher for DeltaR")
    for original, replacement, count in replacements:
        expected = ast.dump(ast.parse(original, mode="eval").body)
        matches = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and ast.dump(node) == expected]
        if len(matches) != count:
            raise ValueError("Unrecognized upstream R-ANODE DeltaR expression: " + original)
        for node in matches:
            updated = ast.parse(replacement, mode="eval").body
            node.args, node.keywords = updated.args, updated.keywords
    return ast.fix_missing_locations(tree)
