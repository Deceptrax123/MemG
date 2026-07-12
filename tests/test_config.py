from mcp.config import build_runtime_config, parse_args


def test_cli_defaults_build_runtime_config(tmp_path):
    args = parse_args(["--device", "cpu", "--cache-dir", str(tmp_path), "--top-k", "7"])
    config = build_runtime_config(args)
    assert config.device == "cpu"
    assert config.cuda is False
    assert config.top_k == 7
    assert config.cache_dir == str(tmp_path)


def test_underscore_cli_alias_is_backwards_compatible():
    assert parse_args(["--top_k", "9"]).top_k == 9
