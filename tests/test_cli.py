from agent_smithers.cli import build_parser


def test_cli_overrides():
    p = build_parser()
    args = p.parse_args([
        "--env-file", ".env.test",
        "--log-level", "DEBUG",
        "--model", "gpt-4o",
        "--store-path", "st",
        "--e2e",
    ])
    assert args.env_file == ".env.test"
    assert args.log_level == "DEBUG"
    assert args.model == "gpt-4o"
    assert args.store_path == "st"
    assert args.e2e is True
