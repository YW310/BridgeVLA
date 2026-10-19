"""FastAPI inference service for checkpoint-bound BridgeVLA OHT policies."""
import argparse
from contextlib import asynccontextmanager
import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .runtime.inference_api import API_VERSION, APIError
from .runtime.inference_service import InferenceService
from .runtime.transport import MAX_BYTES


def create_app(model_path, *, max_bytes=MAX_BYTES, **service_options):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    @asynccontextmanager
    async def lifespan(app):
        service = InferenceService(model_path, **service_options)
        app.state.service = service
        try:
            await service.start()
            yield
        finally:
            await service.close()

    app = FastAPI(title="BridgeVLA OHT inference", version=API_VERSION, lifespan=lifespan)
    app.state.service = None

    def runtime():
        if app.state.service is None:
            raise APIError(503, "not_ready", "Model has not been loaded")
        return app.state.service

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        if not exc.logged and app.state.service is not None:
            app.state.service.log_rejection(request.url.path, exc)
        return JSONResponse(status_code=exc.status, content=exc.payload())

    @app.get("/health")
    async def health():
        if app.state.service is None:
            return JSONResponse(status_code=503, content=dict(ready=False, busy=False))
        status = runtime().health()
        return JSONResponse(status_code=200 if status["ready"] else 503, content=status)

    @app.get("/model_info")
    async def model_info():
        result = runtime().model_info()
        result["max_request_bytes"] = max_bytes
        return result

    async def dispatch(request, operation):
        started = perf_counter()
        service = runtime()
        service.check_available()
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise APIError(415, "unsupported_media_type", "Use Content-Type: application/json")
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError as exc:
                raise APIError(400, "invalid_length", "Invalid Content-Length") from exc
            if length < 0:
                raise APIError(400, "invalid_length", "Invalid Content-Length")
            if length > max_bytes:
                raise APIError(413, "request_too_large", "HTTP body exceeds the configured byte limit")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_bytes:
                raise APIError(413, "request_too_large", "HTTP body exceeds the configured byte limit")
            body.extend(chunk)
        return await service.execute(operation, bytes(body), started)

    @app.post("/predict")
    async def predict(request: Request):
        return await dispatch(request, "predict")

    @app.post("/reset")
    async def reset(request: Request):
        return await dispatch(request, "reset")

    return app


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-path", "--model_path", dest="model_path",
                       help="Checkpoint file or run directory containing model_last.pth")
    model.add_argument("--checkpoint", dest="model_path", help="Alias for a checkpoint path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrain-path")
    parser.add_argument("--predictor")
    parser.add_argument("--predictor-provenance")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--output-dir", "--output_dir", default="logs/oht_vla_server")
    parser.add_argument("--model-id", help="Optional deployment identifier; default derives from checkpoint path/stat")
    parser.add_argument("--max-request-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--warmup-replay")
    parser.add_argument("--warmup-split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--warmup-index", type=int, default=0)
    return parser


def main(argv=None):
    import uvicorn
    args = vars(build_arg_parser().parse_args(argv))
    host, port = args.pop("host"), args.pop("port")
    max_bytes = args.pop("max_request_bytes")
    app = create_app(max_bytes=max_bytes, **args)
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
