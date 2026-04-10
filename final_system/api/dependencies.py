# api/dependencies.py
#
# Shared FastAPI-зависимости: авторизация, настройки.

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.settings import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    Проверяет Bearer-токен.
    Если API_KEY не задан в настройках — авторизация отключена (режим разработки).
    """
    if settings.api_key is None:
        return  # auth disabled

    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Невалидный или отсутствующий Bearer-токен"},
        )
