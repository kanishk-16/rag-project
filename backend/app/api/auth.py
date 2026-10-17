from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.core.security import hash_password, verify_password, create_access_token, decode_access_token

router = APIRouter()
security_scheme = HTTPBearer(auto_error=False)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    email: str
    password: str


class UserLoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str


# ── Dependency: Current Authenticated User ───────────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Validate the Bearer token and return the authenticated User model.
    Raises 401 if token is missing, invalid, or expired.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required. Please login first.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(request: UserRegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    email = request.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    user = User(
        email=email,
        hashed_password=hash_password(request.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse(id=user.id, email=user.email)


@router.post("/login", response_model=TokenResponse)
def login_user(request: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticate with email and password to receive a JWT access token."""
    email = request.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    return TokenResponse(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
def get_my_profile(current_user: User = Depends(get_current_user)):
    """Get the currently logged-in user profile."""
    return UserResponse(id=current_user.id, email=current_user.email)
