from src.pipeline import RefineryPipeline


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the document refinery pipeline")
    parser.add_argument("document", help="Path to input document (PDF or text)")
    parser.add_argument("--config", default="rubric/extraction_rules.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default=".refinery", help="Output artifact directory")
    args = parser.parse_args()

    pipeline = RefineryPipeline(config_path=args.config, output_dir=args.output_dir)
    result = pipeline.run(args.document)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
