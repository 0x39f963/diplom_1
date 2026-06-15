"""Сид демо-аккаунта внешнего backend: контрагенты + договоры (богатые данные для real-API демо).

Запуск (из eva-agent-lab, backend поднят, EVA_API_BASE=http в .env):
    PYTHONPATH=src python seed_demo.py --counterparties 60 --contracts 40
Логин - из .env (EVA_LOGIN/EVA_PASSWORD). Скрипт только СОЗДАЕТ демо-данные в твоем демо-аккаунте.
"""

from __future__ import annotations

import argparse
import random
import sys

import httpx

from eva_agent.settings import settings

_PREFIX = ["ООО", "АО", "ПАО"]
_CORE = [
    "Альфа", "Бета", "Вектор", "Горизонт", "Меридиан", "Импульс", "Каскад", "Ореол", "Спектр",
    "Грань", "Контур", "Атлас", "Зенит", "Орбита", "Фактор", "Лидер", "Прайм", "Сириус", "Квант",
    "Новатор", "Эталон", "Прогресс", "Магнит", "Радиус", "Пилот", "Капитал", "Динамо", "Триумф",
]
_SUFFIX = ["Медиа", "Реклама", "Маркетинг", "Диджитал", "Промо", "Групп", "Трейд", "Сервис", "Студия"]


def _inn(index: int) -> str:
    return f"78{index:08d}"


def _login() -> httpx.Client:
    client = httpx.Client(base_url=settings.eva_api_base, timeout=20.0)
    response = client.post(
        "/api/auth/login",
        json={"email": settings.eva_login, "password": settings.eva_password, "remember": True},
    )
    response.raise_for_status()
    token = client.cookies.get("eva_session")  # Secure-cookie по http шлем явным заголовком
    if token:
        client.headers["Cookie"] = f"eva_session={token}"
    return client


def seed(client: httpx.Client, n_counterparties: int, n_contracts: int) -> None:
    counterparty_ids: list[int] = []
    for i in range(n_counterparties):
        name = f"{random.choice(_PREFIX)} {random.choice(_CORE)}-{random.choice(_SUFFIX)}"
        response = client.post(
            "/api/counterparties",
            json={"name": name, "inn": _inn(2000 + i), "legal_type": "ul"},
        )
        if response.status_code in (200, 201):
            counterparty_ids.append(response.json()["id"])
    print(f"контрагентов создано: {len(counterparty_ids)}")

    made = 0
    for i in range(n_contracts):
        if len(counterparty_ids) < 2:
            break
        customer, executor = random.sample(counterparty_ids, 2)
        response = client.post(
            "/api/contracts",
            json={
                "customer_counterparty_id": customer,
                "executor_counterparty_id": executor,
                "contract_type": random.choice(["service", "mediation"]),
                "chain_role": "initial",
                "contract_date": f"2025-{random.randint(1, 9):02d}-{random.randint(1, 28):02d}",
                "contract_number": f"Д-2025/{i + 100}",
                "price_not_stipulated": True,
            },
        )
        if response.status_code in (200, 201):
            made += 1
    print(f"договоров создано: {made}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterparties", type=int, default=60)
    parser.add_argument("--contracts", type=int, default=40)
    args = parser.parse_args()

    if not settings.eva_api_base.startswith("http"):
        print("EVA_API_BASE не http (сейчас 'mock') - нечего сидить. Укажи http://localhost:8082 в .env.")
        return 1

    client = _login()
    seed(client, args.counterparties, args.contracts)
    print("готово. read-only агент теперь видит эти данные через eva_* tools.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
