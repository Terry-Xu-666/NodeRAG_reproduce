import uuid
from hashlib import md5, sha256
from typing import List
from random import getrandbits

def genid(input: List[str],type:str) -> str:
    match type:
        case "md5":
            return md5_hash(input)
        case "sha256":
            return sha256_hash(input)
        case "uuid":
            return uuid_hash(input)
        case _:
            raise ValueError("Type not supported")

def md5_hash(input: List[str]) -> str:
    hashed = md5("".join(input).encode('utf-8')).hexdigest()
    return f'{hashed}'

def sha256_hash(input: List[str]) -> str:
    hashed = sha256("".join(input).encode('utf-8')).hexdigest()
    return f'{hashed}'

def uuid_hash() -> str:
    return str(uuid.uuid4(getrandbits(128), version=4).hex())