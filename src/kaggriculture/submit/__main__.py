import argparse
from pathlib import Path

from kaggriculture.submit.build import DEFAULT_AGENT, build
from kaggriculture.submit.monitor import download_logs, download_replay, episodes, submissions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m kaggriculture.submit")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="package an agent into submissions/<tag>/")
    build_p.add_argument("--tag", default=None, help="submissions/<tag>/ directory name (default: UTC timestamp)")
    build_p.add_argument("--agent", type=Path, default=DEFAULT_AGENT, help="entry-point agent file")
    build_p.add_argument("--extra", type=Path, nargs="*", default=[], help="extra files for a multi-file bundle")

    subs_p = sub.add_parser("submissions", help="list submissions and their status")

    eps_p = sub.add_parser("episodes", help="list episodes for a submission")
    eps_p.add_argument("submission_id")

    replay_p = sub.add_parser("replay", help="download a replay JSON")
    replay_p.add_argument("episode_id")
    replay_p.add_argument("--tag", required=True, help="downloads into submissions/<tag>/")

    logs_p = sub.add_parser("logs", help="download an agent's logs for an episode")
    logs_p.add_argument("episode_id")
    logs_p.add_argument("agent_index", type=int)
    logs_p.add_argument("--tag", required=True, help="downloads into submissions/<tag>/")

    args = parser.parse_args(argv)

    if args.command == "build":
        artifact = build(args.tag, args.agent, tuple(args.extra))
        print(artifact)
    elif args.command == "submissions":
        print(submissions())
    elif args.command == "episodes":
        print(episodes(args.submission_id))
    elif args.command == "replay":
        print(download_replay(args.episode_id, Path("submissions") / args.tag))
    elif args.command == "logs":
        print(download_logs(args.episode_id, args.agent_index, Path("submissions") / args.tag))


if __name__ == "__main__":
    main()
