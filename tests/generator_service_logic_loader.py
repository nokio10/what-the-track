import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SERVICE = ROOT / "src" / "game" / "generator_service.py"

CONSTANT_NAMES = {
    "ALBUM_TRACK_RE",
    "TRACK_ONLY_RE",
    "DEFAULT_CPU_MDX_BATCH_SIZE",
    "DEFAULT_CPU_WHISPER_THREADS",
    "DEFAULT_CPU_WHISPER_BATCH_SIZE",
    "DEFAULT_GPU_MDX_BATCH_SIZE",
    "DEFAULT_GPU_WHISPER_THREADS",
    "DEFAULT_GPU_WHISPER_BATCH_SIZE",
    "MIN_AUDIO_POSITION",
    "MIN_QUESTION_DURATION_MS",
    "TARGET_QUESTION_DURATION_MS",
    "QUESTION_WORD_GUARD_MS",
    "CONTEXT_LOOKBACK_WORDS",
    "CONSENSUS_TIME_TOLERANCE_SEC",
    "CONSENSUS_CONTEXT_WORDS",
    "CONSENSUS_MIN_RUN",
    "CONSENSUS_SILENCE_SEC",
    "CONSENSUS_MIX_SUPPORT_SEC",
    "CONSENSUS_LINE_END_PAUSE_SEC",
    "ANSWER_FUNCTION_POS",
    "CONSENSUS_STRONG_END",
    "CONSENSUS_ANY_END",
    "GIGAAM_URL",
    "GAME_ID_RE",
    "ALBUM_ONLY_RE",
    "USER_PLAYLIST_RE",
    "GIGAAM_TIMEOUT_SEC",
    "job_status_lock",
    "job_status",
    "_GIGAAM_STATE",
    "COMPUTE_TYPE",
    "SCORE_EOL_BONUS",
    "SCORE_PUNCTUATION_BONUS",
    "SCORE_LONG_WORD_BONUS",
    "SCORE_MEDIUM_WORD_BONUS",
    "SCORE_UNIQUE_WORD_BONUS",
    "SCORE_RARE_WORD_BONUS",
    "SCORE_FREQUENT_PENALTY",
    "SCORE_VERB_ENDING_PENALTY",
    "STOP_WORDS",
    "ALLOWED_SINGLE_CHAR_WORDS",
    "RUSSIAN_VOWELS",
    "QUESTION_WORD_BLOCKING_SEPARATORS",
}

SYMBOL_NAMES = {
    "extract_yandex_track_identity",
    "_identity_label",
    "dedupe_yandex_track_urls",
    "YandexRequestThrottle",
    "_env_int",
    "get_runtime_tuning",
    "_log",
    "clean_word",
    "_word_has_vowel",
    "_should_preserve_skipped_token",
    "_clean_lyrics_line_text",
    "_line_words",
    "_visible_text_tokens",
    "_levenshtein_ratio",
    "_token_overlap",
    "_line_similarity",
    "_collect_line_records",
    "_global_align_lines",
    "_align_words_in_line",
    "should_keep_word_token",
    "extract_rhyme_words_from_lyrics",
    "recover_segment_skipped_tokens",
    "get_russian_syllable_tail",
    "extract_separator_blocked_fragments",
    "is_separator_fragment_candidate",
    "score_lyrics_rhyme_candidates",
    "_occurrence_repeat_penalty",
    "_line_repeat_penalty",
    "_score_occurrence",
    "rank_lyrics_candidates",
    "select_word_from_lyrics_algorithmically",
    "score_candidates",
    "get_algorithmic_choice",
    "calculate_timings",
    "question_window_has_enough_context",
    "_build_context_from_answer_line",
    "_find_previous_lyrics_line",
    "_get_previous_official_lyrics_line",
    "_should_break_general_context_line",
    "build_context_string",
    "remap_words_to_official_by_line",
    "build_question_payload",
    "check_spoiler_in_question",
    "consensus_tokens",
    "_consensus_pair",
    "_median",
    "consensus_clusters",
    "consensus_tier",
    "consensus_word_tiers",
    "consensus_word_info",
    "consensus_line_end",
    "consensus_phrase_signals",
    "GigaamUnavailable",
    "_ForwardedLogs",
    "_ForwardedStatus",
    "_apply_generation_event",
    "GigaamTrackError",
    "whisper_compute_type",
    "_env_float",
    "is_valid_game_id",
    "raw_word_index",
    "raw_spoiler",
    "expand_yandex_urls",
    "transcribe_gigaam",
    "consensus_priority",
    "_morph_analyzer",
    "answer_pos_class",
    "consensus_candidates",
    "mix_support",
    "map_tiers_to_words",
    "select_general_question",
}


def _is_supported_import(node):
    if isinstance(node, ast.Import):
        return any(alias.name in {"os", "re", "time", "threading"} for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module == "collections" and any(alias.name == "Counter" for alias in node.names)
    return False


def _is_supported_assignment(node):
    if not isinstance(node, ast.Assign):
        return False
    return any(isinstance(target, ast.Name) and target.id in CONSTANT_NAMES for target in node.targets)


def _is_supported_symbol(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name in SYMBOL_NAMES
    return False


def load_generator_service_logic():
    tree = ast.parse(GENERATOR_SERVICE.read_text(encoding="utf-8"))
    selected_nodes = []

    for node in tree.body:
        if _is_supported_import(node) or _is_supported_assignment(node) or _is_supported_symbol(node):
            selected_nodes.append(node)

    namespace = {}
    module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(module, str(GENERATOR_SERVICE), "exec"), namespace)
    return namespace
