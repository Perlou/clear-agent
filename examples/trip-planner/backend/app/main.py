"""FastAPI 入口"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import get_service
from .config import get_settings
from .schemas import TripPlanResponse, TripRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("trip-planner")

settings = get_settings()

app = FastAPI(
    title="ClearAgent 旅行规划助手",
    version="0.1.0",
    description="基于 ClearAgent 框架（MCP + ReActAgent + 结构化输出）的旅行规划 demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    logger.info("启动中：校验配置并预热 Agent ...")
    try:
        settings.assert_ready()
        # 预热（首请求时间从 30s 降到秒级）
        get_service()
        logger.info("✅ Agent 已就绪，监听 %s:%d", settings.app_host, settings.app_port)
    except Exception as exc:  # noqa: BLE001
        # 不抛 —— 让 / health 仍可访问，便于排查
        logger.error("❌ 启动初始化失败：%s", exc)


@app.get("/")
def root() -> dict:
    return {
        "name": "clear-agent-trip-planner",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/trip/plan", response_model=TripPlanResponse)
def plan(request: TripRequest) -> TripPlanResponse:
    """生成旅行计划"""
    if request.travel_days <= 0:
        raise HTTPException(status_code=400, detail="travel_days 必须为正整数")
    try:
        service = get_service()
        trip = service.plan(request)
        return TripPlanResponse(success=True, message="ok", data=trip)
    except Exception as exc:  # noqa: BLE001
        logger.exception("规划失败：%s", exc)
        return TripPlanResponse(success=False, message=str(exc), data=None)
