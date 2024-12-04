from typing import Union, List
from pydantic import BaseModel
from fastapi import Query
from typing import Optional
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str


class UserInDB(User):
    hashed_password: str


class cv_json_1(BaseModel):
    UserId: str

