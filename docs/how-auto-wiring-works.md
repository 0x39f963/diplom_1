# Как работает auto-wiring

## Проблема

Планировщик строит todo-лист из шагов. Раньше модель могла связать шаги так:

```json
{"$from": {"step": 2, "path": "contract_id"}}
```

Это хрупко. Модель легко ошибается в номере шага. Еще хуже, нормализация плана может перенумеровать
`order`, и ссылка начинает смотреть не туда. Валидатор видит forward-ссылку, блокирует шаг, и план уходит
в уточнение без полезных инструментов.

## Смысл

Связывание данных переносится в детерминированный код. Модель перечисляет todo и входы, которые уже знает.
Исполнитель сам подставляет входы по машинной карте связей `RELATIONS`.

Правило строгое: если связь не однозначна, план блокируется и просит уточнение. Код не выбирает первый
элемент из списка и не делает догадок.

## Todo id

`todo_id` - стабильное имя todo. Оно не меняется при перенумерации шагов.

Новая явная ссылка выглядит так:

```json
{"$from": {"todo": "get_creative_status", "path": "contract_id"}}
```

Ссылка говорит: взять результат todo `get_creative_status` и достать из него поле `contract_id`.

Старые ссылки вида `$from.step` поддерживаются для совместимости. Перед проверкой плана binding-слой
пытается перевести их в `$from.todo`, если номер шага известен однозначно.

## Auto-wiring

`RELATIONS` описывает машинные связи между инструментами:

- какой инструмент производит значение;
- по какому пути его взять;
- какой инструмент потребляет значение;
- в какой аргумент его поставить;
- нужна ли выборка по selector;
- одно значение ожидается или список.

Если у шага не хватает обязательного аргумента, binding-слой ищет производителя по `RELATIONS`.

Пример:

```text
eva_get_creative_status.contract_id -> eva_get_contract.contract_id
```

Если в плане есть один todo с `eva_get_creative_status` и один todo с `eva_get_contract`, binding вставит:

```json
{
  "$from": {
    "todo": "get_creative_status",
    "path": "contract_id",
    "selector": null,
    "cardinality": "one"
  }
}
```

В trace появится строка:

```text
auto-wire eva_get_contract.contract_id <- get_creative_status.contract_id
```

## Selector и cardinality

Связь `ContractParty->Counterparty` идет через список сторон:

```text
parties[].counterparty_id
```

Если пользователь спросил "заказчик", план содержит `role=customer`. Тогда selector `role` сужает список
до одной стороны:

```text
role=customer -> один counterparty_id -> один вызов eva_get_counterparty
```

Если пользователь спросил "все стороны", selector value нет. При `cardinality=many` исполнитель делает
fan-out: вызывает `eva_get_counterparty` по одному разу на каждое значение.

Fan-out ограничен cap 20. Если значений больше, исполнитель обработает первые 20 и добавит blocker и trace:

```text
fan-out capped at 20 of N
```

Пустой список производителя блокирует план с blocker:

```text
empty producer
```

## Примеры цепочек

Creative->Contract:

1. Todo `get_creative_status` вызывает `eva_get_creative_status(creative_id)`.
2. Результат содержит `contract_id`.
3. Todo `get_contract` не пишет `$from` вручную.
4. Binding вставляет `contract_id` из результата `get_creative_status`.
5. Исполнитель вызывает `eva_get_contract(contract_id)`.

ContractParty->Counterparty by role:

1. Todo `get_contract_parties` вызывает `eva_get_contract_parties(contract_id)`.
2. Результат содержит `parties[]`.
3. Todo `get_counterparty` требует `counterparty_id`.
4. Если есть `role=customer`, selector выбирает одну сторону.
5. Если role нет и нужна обработка всех сторон, срабатывает fan-out.

## Что код не делает

- Не угадывает производителя, если их несколько.
- Не берет первый элемент списка при неоднозначности.
- Не скрывает пустой список производителя.
- Не усекает fan-out молча.
- Не вызывает инструменты во время binding. Binding только меняет план и пишет trace.
