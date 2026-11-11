from backend.db.database import engine, Base, SessionLocal
from backend.auth import models
from backend.graph import models as graph_models
from passlib.context import CryptContext  # 用于用户样例密码哈希

DEPT_SAMPLE_DATA = [
    {"deptname": "HIS研发1组", "manager_name": "张三", "manager_mail": "123349332@qq.com", "superior": "产品开发部"},
    {"deptname": "产品开发部", "manager_name": "李四", "manager_mail": "1212@qq.com", "superior": "HIT事业部"},
    {"deptname": "HIT事业部", "manager_name": "王五", "manager_mail": "321@qq.com", "superior": "集团"},
    {"deptname": "集团", "manager_name": "赵六", "manager_mail": "888@qq.com", "superior": ""},
]

# 用户样例数据：3 个测试账户（password 与 username 相同，便于开发/测试登录）
# 明文密码说明：仅用于本地开发环境 seed，真实生产需手动创建用户
USER_SAMPLE_DATA = [
    {
        "username": "zhangsan",
        "password_plain": "zhangsan",
        "name": "张三",
        "email": "zhangsan@dev.com",
        "department": "HIS研发1组",
        "role": "developer",
    },
    {
        "username": "lisi",
        "password_plain": "lisi",
        "name": "李四",
        "email": "lisi@dev.com",
        "department": "产品开发部",
        "role": "manager",
    },
    {
        "username": "wangwu",
        "password_plain": "wangwu",
        "name": "王五",
        "email": "wangwu@dev.com",
        "department": "HIT事业部",
        "role": "director",
    },
]

# 余额样例数据：2026 年度，按 user_id 对齐上面 USER_SAMPLE_DATA 的插入顺序
# 注：users 表按 USER_SAMPLE_DATA 顺序 add+flush 后，分配的 id 即为 1/2/3
LEAVE_BALANCE_SAMPLE_DATA = [
    # user_id=1（张三）：年假10天已用3, 事假5天已用1, 病假15天已用0, 婚假10天已用0
    {"user_id": 1, "leave_type": "annual",   "total_days": 10, "used_days": 3, "year": 2026},
    {"user_id": 1, "leave_type": "personal", "total_days": 5,  "used_days": 1, "year": 2026},
    {"user_id": 1, "leave_type": "sick",     "total_days": 15, "used_days": 0, "year": 2026},
    {"user_id": 1, "leave_type": "marriage", "total_days": 10, "used_days": 0, "year": 2026},
    # user_id=2（李四）：年假15天已用2, 事假5天已用0, 病假15天已用0
    {"user_id": 2, "leave_type": "annual",   "total_days": 15, "used_days": 2, "year": 2026},
    {"user_id": 2, "leave_type": "personal", "total_days": 5,  "used_days": 0, "year": 2026},
    {"user_id": 2, "leave_type": "sick",     "total_days": 15, "used_days": 0, "year": 2026},
    # user_id=3（王五）：年假20天已用10, 事假5天已用2, 病假15天已用5（余额少,便于测"不通过"场景）
    {"user_id": 3, "leave_type": "annual",   "total_days": 20, "used_days": 10, "year": 2026},
    {"user_id": 3, "leave_type": "personal", "total_days": 5,  "used_days": 2,  "year": 2026},
    {"user_id": 3, "leave_type": "sick",     "total_days": 15, "used_days": 5,  "year": 2026},
]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash_password(plain: str) -> str:
    """对明文密码做 bcrypt 哈希（与 auth.utils 保持一致）"""
    return pwd_context.hash(plain)


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("All tables created successfully")


def seed_dept_data():
    """向 dept 表插入部门样例数据（幂等，已存在则跳过）"""
    session = SessionLocal()
    try:
        count = session.query(models.Dept).count()
        if count > 0:
            print("dept 表已有数据，跳过初始化")
            return
        for item in DEPT_SAMPLE_DATA:
            session.add(models.Dept(**item))
        session.commit()
        print(f"dept 表已插入 {len(DEPT_SAMPLE_DATA)} 条部门样例数据")
    finally:
        session.close()


def seed_user_data():
    """向 users 表插入用户样例数据（幂等，已存在则跳过）"""
    session = SessionLocal()
    try:
        count = session.query(models.User).count()
        if count > 0:
            print("users 表已有数据，跳过初始化")
            return
        for item in USER_SAMPLE_DATA:
            user = models.User(
                username=item["username"],
                password_hash=_hash_password(item["password_plain"]),
                name=item["name"],
                email=item["email"],
                department=item["department"],
                role=item["role"],
            )
            session.add(user)
        session.commit()
        print(f"users 表已插入 {len(USER_SAMPLE_DATA)} 条用户样例数据")
    finally:
        session.close()


def seed_leave_balance_data():
    """向 leave_balances 表插入余额样例数据（幂等，已存在则跳过）"""
    session = SessionLocal()
    try:
        count = session.query(models.LeaveBalance).count()
        if count > 0:
            print("leave_balances 表已有数据，跳过初始化")
            return
        for item in LEAVE_BALANCE_SAMPLE_DATA:
            session.add(models.LeaveBalance(**item))
        session.commit()
        print(f"leave_balances 表已插入 {len(LEAVE_BALANCE_SAMPLE_DATA)} 条余额样例数据")
    finally:
        session.close()


if __name__ == "__main__":
    create_tables()
    seed_dept_data()
    seed_user_data()
    seed_leave_balance_data()