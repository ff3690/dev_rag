from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    password: str
    email: EmailStr
    name: str = ""
    department: str
    role: str = "developer"

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    name: str
    department: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None