from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# --- Secret key (in real apps, store this in .env file) ---
SECRET_KEY = "your-secret-key-keep-it-safe"
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30  # token expires after 30 minutes

# --- Password hashing setup ---
pwd_context = CryptContext(schemes=["bcrypt"])

# --- Token scheme (reads token from request header) ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# --- Hash a plain password ---
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# --- Check if plain password matches hashed password ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# --- Create a JWT token ---
def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})  # add expiry time
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# --- Decode and verify a JWT token ---
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
