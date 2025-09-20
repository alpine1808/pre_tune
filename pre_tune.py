import os, json, sys
from pre_tune_app.cli.args import parse_args
from pre_tune_app.config.settings import AppConfig, setup_logging
from pre_tune_app.infra.db import build_engine
from pre_tune_app.infra.repositories.documents_repo import SqlDocumentsRepository
from pre_tune_app.infra.repositories.chunks_repo import SqlChunksRepository
from pre_tune_app.pipeline.factory import build_pipeline
from pre_tune_app.main_service import PreTuneService

def _load_dotenv_if_present(filename: str = ".env") -> None:
    # Simple .env loader (no external dependency). Only sets variables not already present.
    if not os.path.exists(filename):
        return
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

def main() -> int:
    _load_dotenv_if_present()  # optional .env
    setup_logging()
    args = parse_args()
    cfg = AppConfig.from_env()  # <-- convenience: read flags from ENV

    engine = build_engine(cfg)
    docs_repo = SqlDocumentsRepository(engine)
    chunks_repo = SqlChunksRepository(engine)
    pipeline = build_pipeline(cfg)
    service = PreTuneService(docs_repo, chunks_repo, pipeline)

    cleaned = service.run_for_filename(args.filename)
    payload = [c.to_dict() for c in cleaned]
    if args.json_out == "-" or not args.json_out.strip():
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return 0

if __name__ == "__main__":
    sys.exit(main())
