from agent_smithers.cli import build_parser


def test_cli_overrides():
    p = build_parser()
    args = p.parse_args([
        "--env-file", ".env.test",
        "--log-level", "DEBUG",
        "--model", "grok-4",
        "--store-path", "st",
        "--e2e",
    ])
    assert args.env_file == ".env.test"
    assert args.log_level == "DEBUG"
    assert args.model == "grok-4"
    assert args.store_path == "st"
    assert args.e2e is True


def test_cli_no_e2e_flag():
    p = build_parser()
    args = p.parse_args(["--no-e2e"])
    assert args.no_e2e is True
    assert args.e2e is False


def test_cli_server_models_flag():
    p = build_parser()
    args = p.parse_args(["--server-models"])
    assert args.server_models is True


def test_cli_verbose_flag():
    p = build_parser()
    args = p.parse_args(["--verbose"])
    assert args.verbose_mode is True


def test_cli_defaults():
    p = build_parser()
    args = p.parse_args([])
    assert args.model is None
    assert args.store_path is None
    assert args.e2e is False
    assert args.no_e2e is False
    assert args.server_models is False
    assert args.verbose_mode is False


def test_cli_short_flags():
    p = build_parser()
    args = p.parse_args(["-L", "WARNING", "-e", "custom.env", "-m", "grok-4", "-s", "mystore", "-E", "-S", "-v"])
    assert args.log_level == "WARNING"
    assert args.env_file == "custom.env"
    assert args.model == "grok-4"
    assert args.store_path == "mystore"
    assert args.e2e is True
    assert args.server_models is True
    assert args.verbose_mode is True
