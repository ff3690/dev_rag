from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from datetime import datetime
from backend.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    # 姓名：用于邮件发送人落款
    name = Column(String(50), default="", nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    department = Column(String(50), nullable=False)
    role = Column(String(20), default="developer", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Dept(Base):
    """部门表：记录部门名称、负责人及上级部门，用于邮件抄送递归查询"""
    __tablename__ = "dept"

    id = Column(Integer, primary_key=True, index=True)
    deptname = Column(String(50), unique=True, index=True, nullable=False)
    manager_name = Column(String(50), default="", nullable=False)
    manager_mail = Column(String(100), default="", nullable=False)
    superior = Column(String(50), default="", nullable=False)

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(20), nullable=False)
    category = Column(String(50), nullable=False)
    department = Column(String(50), nullable=False)
    vector_index = Column(String(100))
    status = Column(String(20), default="pending", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    """问答历史记录表（每个问答对为一条记录）"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    # 所属用户（仅记录并展示当前登录用户自己的历史，实现用户隔离）
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # 用户提出的问题
    question = Column(String, nullable=False)
    # AI 生成的完整回答
    answer = Column(String)
    # 引用来源（JSON 字符串，命中文档列表）
    sources = Column(String)
    # Token 用量（JSON 字符串，prompt/completion/total）
    token_usage = Column(String)
    # 使用的模型名称（新增，用于历史记录展示）
    model = Column(String, default="")
    conversation_type = Column(String(20), default="rag", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Permission(Base):
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(20), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(Integer)
    access_level = Column(String(20), nullable=False)


class LeaveBalance(Base):
    """
    请假余额表：按用户+假别+年度维度记录假期额度

    典型用法：
      - 用户发起请假申请前，通过 query_leave_balance 工具读取 remaining_days
      - 假别枚举：annual 年假 / sick 病假 / personal 事假 / maternity 产假 / marriage 婚假
      - 每年初由管理员批量初始化（seed 或脚本），请假获批后 used_days += 天数
    """
    __tablename__ = "leave_balances"

    id = Column(Integer, primary_key=True, index=True)
    # 所属用户：关联 users.id（余额是"当前登录用户"自己的，实现用户隔离）
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    # 假别：annual / sick / personal / maternity / marriage
    leave_type = Column(String(20), nullable=False)
    # 年度总额度（天）
    total_days = Column(Integer, nullable=False, default=0)
    # 已使用天数（天）
    used_days = Column(Integer, nullable=False, default=0)
    # 年度（如 2026），用于跨年重置
    year = Column(Integer, nullable=False)
    # 同用户同假别同年唯一，防止重复建行
    __table_args__ = (
        UniqueConstraint("user_id", "leave_type", "year", name="uq_leave_balance_user_type_year"),
    )