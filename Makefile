scrap:
	cd services/scrapper && uv run scrapper.py

hpo:
	uv run python -m src.sft_process.hpo

onnx:
	uv run python -m src.sft_process.export_to_onnx