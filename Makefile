.DEFAULT_GOAL := help
PY := python
RAG_API := http://localhost:8077

# Поиск по закону работает отдельным сервисом (адрес в RAG_API_BASE).
# Подними его рядом перед cli/eval/ui: docker compose up -d && make api

.PHONY: help install smoke cli eval bench ui lint type test gates

help: ## Список целей
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

install: ## Зависимости + RU spaCy-модель
	pip install -r requirements.txt
	$(PY) -m spacy download ru_core_news_lg

smoke: ## Smoke LLM-коннекторов (OpenRouter + локальный Ollama)
	$(PY) -m eva_agent.llm.smoke

cli: ## Агент в консоли: make cli Q="нужен ли ERID для баннера?"
	RAG_API_BASE=$(RAG_API) $(PY) -m eva_agent.cli "$(Q)"

eval: ## Бенчмарк + 3 типа eval + метрики (нужен поднятый retrieval API)
	RAG_API_BASE=$(RAG_API) $(PY) -m evals.run_evals

bench: eval ## Алиас прогона бенчмарка

ui: ## Демо web-чат (Chainlit)
	RAG_API_BASE=$(RAG_API) chainlit run ui/app.py -w

lint: ## ruff
	ruff check src tests evals ui
type: ## mypy
	mypy src evals ui
test: ## pytest
	pytest -q
gates: lint type test ## Все гейты (ruff + mypy + pytest)
