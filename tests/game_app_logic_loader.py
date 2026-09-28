import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAME_APP = ROOT / "game_app.py"

CONSTANT_NAMES = {
    "MIN_PLAYER_ANSWER_LENGTH",
    "GAME_ID_RE",
}

SYMBOL_NAMES = {
    "normalize_answer_strict",
    "validate_player_answer_text",
    "is_valid_game_id",
    "online_players",
    "pick_vip_sid",
}


def _is_supported_import(node):
    if isinstance(node, ast.Import):
        return any(alias.name == "re" for alias in node.names)
    return False


def _is_supported_assignment(node):
    if not isinstance(node, ast.Assign):
        return False
    return any(isinstance(target, ast.Name) and target.id in CONSTANT_NAMES for target in node.targets)


def _is_supported_symbol(node):
    return isinstance(node, ast.FunctionDef) and node.name in SYMBOL_NAMES


def load_game_app_logic():
    tree = ast.parse(GAME_APP.read_text(encoding="utf-8"))
    selected_nodes = []

    for node in tree.body:
        if _is_supported_import(node) or _is_supported_assignment(node) or _is_supported_symbol(node):
            selected_nodes.append(node)

    namespace = {}
    module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(module, str(GAME_APP), "exec"), namespace)
    return namespace
