"""Комментарий с ревью в PR: один на PR, обновляется при каждом новом push.

Кто и куда — из окружения GitHub Actions: `GITHUB_TOKEN`, `GITHUB_REPOSITORY`
и номер PR из события в `GITHUB_EVENT_PATH` (или из `PR_NUMBER`, если ревью
зовут не из workflow). Комментарий помечен HTML-маркером; перед публикацией
ищется свой прежний комментарий с этим маркером и перезаписывается — иначе
каждый push плодил бы по комментарию, и PR превращался бы в ленту.

Без токена или вне события PR публикация — отказ с причиной, а не падение:
ревью к тому моменту уже напечатано и записано в файл.
"""

import json
import os
from pathlib import Path

import httpx

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
TIMEOUT = 30
PER_PAGE = 100


class PublishError(RuntimeError):
    """Нечем или некуда публиковать."""


def _pr_number() -> int:
    explicit = os.environ.get("PR_NUMBER")
    if explicit and explicit.isdigit():
        return int(explicit)
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not Path(event_path).is_file():
        raise PublishError("нет PR_NUMBER и события GITHUB_EVENT_PATH — не из workflow pull_request")
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    number = (event.get("pull_request") or {}).get("number") or event.get("number")
    if not number:
        raise PublishError("в событии нет pull_request.number — workflow запущен не на PR")
    return int(number)


def comment(body: str, *, marker: str) -> str:
    """Публикует или обновляет комментарий с маркером. Возвращает его URL."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token:
        raise PublishError("нет GITHUB_TOKEN")
    if not repo:
        raise PublishError("нет GITHUB_REPOSITORY")
    number = _pr_number()

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(headers=headers, timeout=TIMEOUT) as client:
        existing = None
        page = 1
        while existing is None:
            response = client.get(f"{API}/repos/{repo}/issues/{number}/comments", params={"per_page": PER_PAGE, "page": page})
            _check(response)
            items = response.json()
            existing = next((item for item in items if marker in (item.get("body") or "")), None)
            if not items or len(items) < PER_PAGE:
                break
            page += 1

        if existing:
            response = client.patch(f"{API}/repos/{repo}/issues/comments/{existing['id']}", json={"body": body})
        else:
            response = client.post(f"{API}/repos/{repo}/issues/{number}/comments", json={"body": body})
        _check(response)
        return response.json().get("html_url", "")


def _check(response: httpx.Response) -> None:
    if response.status_code >= 400:
        detail = response.text[:200].replace("\n", " ")
        raise PublishError(f"GitHub API {response.status_code}: {detail}")
