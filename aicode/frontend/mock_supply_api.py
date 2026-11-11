"""
Mock 第三方办公用品管理系统 - Supply Management System API

模拟一个第三方系统的 REST API，接受领用申请并返回处理结果。

启动方式:
    python -m uvicorn aicode.frontend.mock_supply_api:app --port 8001

或者作为 FastAPI 子应用挂载到主服务。
"""

import json
import uuid
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="办公用品管理系统 Mock API", version="1.0.0")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================

class ApplicationItem(BaseModel):
    name: str = Field(..., description="物品名称")
    quantity: int = Field(..., ge=1, description="申请数量")
    unit: str = Field(default="个", description="单位")
    category: str = Field(default="其他", description="分类")
    estimated_price: float = Field(default=0, description="预估单价")


class SupplyApplication(BaseModel):
    applicant: str = Field(..., description="申请人")
    department: str = Field(..., description="部门")
    application_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    items: List[ApplicationItem] = Field(..., min_length=1)
    total_items: int = Field(default=0, description="物品种类数")
    total_quantity: int = Field(default=0, description="物品总数量")
    reason: str = Field(default="常规领用", description="申请原因")
    predicted_by: str = Field(default="手动", description="预测来源")
    urgency: str = Field(default="normal", description="紧急程度 normal/urgent")


class ApplicationResponse(BaseModel):
    application_id: str
    status: str
    message: str
    submitted_at: str
    estimated_delivery: Optional[str] = None
    approver: Optional[str] = None
    items_approved: List[dict] = []


# ==================== 申请存储（内存） ====================

applications_db: dict = {}


# ==================== API 端点 ====================

@app.post("/api/v1/supply/application", response_model=ApplicationResponse)
async def submit_application(application: SupplyApplication):
    """
    提交领用申请

    模拟第三方系统的处理流程：
    1. 校验申请数据
    2. 生成申请单号
    3. 返回受理结果
    """
    # 生成申请单号
    now = datetime.now()
    app_id = f"SUP-{now.strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"

    # 模拟物品库存检查
    approved_items = []
    for item in application.items:
        quantity = item.quantity
        if application.urgency == "urgent":
            quantity = item.quantity  # 紧急申请全部通过
        elif item.quantity > 100:
            quantity = item.quantity // 2  # 超量申请减半

        approved_items.append({
            "name": item.name,
            "requested": item.quantity,
            "approved": quantity,
            "unit": item.unit,
            "status": "已批准" if quantity == item.quantity else "部分批准"
        })

    # 预估送达时间
    delivery_date = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    if application.urgency == "urgent":
        delivery_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    # 模拟审批人
    approvers = {
        "技术部": "李总监",
        "产品部": "王经理",
        "设计部": "赵主管",
        "运营部": "钱经理",
    }

    response_data = {
        "application_id": app_id,
        "status": "已受理",
        "message": f"领用申请已提交成功，{len(approved_items)}类物品进入审批流程",
        "submitted_at": now.isoformat(),
        "estimated_delivery": delivery_date,
        "approver": approvers.get(application.department, "行政部"),
        "items_approved": approved_items,
    }

    # 保存记录
    applications_db[app_id] = {
        **application.dict(),
        "response": response_data,
        "created_at": now.isoformat(),
    }

    # 提交成功后在服务端控制台打印提示信息
    print(
        f"\n[申请提交成功] 单号: {app_id} | 申请人: {application.applicant} | "
        f"部门: {application.department} | 物品种类: {len(approved_items)} | "
        f"紧急程度: {application.urgency} | 预计送达: {delivery_date}"
    )

    return response_data


@app.get("/api/v1/supply/application/{application_id}")
async def get_application_status(application_id: str):
    """
    查询申请状态
    """
    app_record = applications_db.get(application_id)
    if not app_record:
        raise HTTPException(status_code=404, detail="申请单号不存在")

    return app_record.get("response", {})


@app.get("/api/v1/supply/applications")
async def list_applications(limit: int = 20):
    """
    列出所有申请
    """
    records = list(applications_db.values())[:limit]
    return {
        "total": len(applications_db),
        "records": records,
    }


@app.get("/api/v1/supply/health")
async def health_check():
    """
    健康检查
    """
    return {
        "status": "healthy",
        "service": "办公用品管理系统",
        "version": "1.0.0",
        "uptime": str(datetime.now()),
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
