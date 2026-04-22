from pydantic import BaseModel

class User(BaseModel):
    id: str
    username: str
    email: str | None = None
    disabled: bool | None = None
    email_verified: bool | None = None

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

