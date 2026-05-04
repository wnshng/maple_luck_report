from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NexonLoginGuide:
    """Placeholder for future Nexon Login / OpenID data consent integration.

    Nexon documents "Login for Game Data Usage" as a Nexon Login based consent flow.
    A production implementation needs OAuth/OpenID application settings, redirect URI
    handling, token storage, refresh logic, and data scopes. The Streamlit MVP keeps
    API Key direct input as the active authentication method.
    """

    title: str = "넥슨 게임 데이터 활용 로그인"
    status: str = "TODO"
    message: str = (
        "넥슨 게임 데이터 활용 로그인 방식은 추후 OAuth/동의 기반 연동으로 확장 예정입니다. "
        "현재는 Open API Key 입력 방식으로 이용해주세요."
    )


def get_nexon_login_guide() -> NexonLoginGuide:
    return NexonLoginGuide()

