# Камеры → физическая DG5F: SDK/servo интеграция

2026-10-19. Добавлена физическая ветка `live-hand --backend tesollo`.
Это реализация подключения к прежнему аппаратному стеку, проверенная без
движения настоящей руки. Аппаратный прогон ещё нужен: Kinect сейчас не видны
в USB, маршрут к 169.254.186.72 идёт через Wi-Fi, проводного маршрута нет.

## Реализация

```text
ViKi LiveHandEstimator → исходный Hybrid → HandJointTarget (radians)
    → один shared latest slot с исходным timestamp
    → отдельный процесс DG5FHandController
        → исходный Runtime / DemoServoController 60 Hz
        → исходный current/contact/lead guard
        → исходный TesolloDg5fBackend → dg5f_python → DGSDK
```

Код сервоконтроллера, guard, SDK wrapper, калибровка степеней/токов и
fault map не переписаны. Они берутся из read-only portable DG5F проекта.
Новый image `live-hand-hardware` собирает именно его native wrapper и
поставляемый `libDGSDK.so`, сохраняет source hashes. Перед реальным
подключением они сверяются с текущим внешним SDK source; несовпадение
требует пересборки, а не молчаливого запуска другой версии.

ViKi передаёт q_target в радианах; на входе исходного Runtime выполняется
ровно одна конверсия в градусы. Canonical joint order и неисправный
`rj_dg_5_1=0` проверяются. Geometric Hybrid weights и семь fingertip distances
передаются тому же contact guard.

Свежесть считается от source timestamp камеры. Доставка в другой процесс
не обновляет возраст. Runtime использует тот же TTL 250 мс. Неполный кадр,
невалидный target или его истечение приводят к TRACKING_HOLD. Shared slot
надёжно заменяет предыдущий target, включая явный `None`; промежуточные
команды не копятся в очередь для последующего исполнения.

Все вызовы backend выполняет исходный SDK worker. Parent/renderer только
передают targets и явные действия. Отдельный процесс завершает Runtime
также при смерти inference-parent. Ctrl+C обрабатывает parent; SIGTERM
owner вызывает нормальное завершение Runtime, suspend и disconnect.
Ошибки из cleanup SDK worker перехватываются отдельно: одного завершения
Python thread недостаточно для успешного `closed=true`.

В первой версии новой обвязки тест SIGTERM обнаружил deadlock: вызов
`multiprocessing.Event.set()` из signal handler мог прервать `Event.wait()`
под его нерекурсивной блокировкой. Исправлено: signal handler только
устанавливает обычный Python-флаг, завершение идёт из основного цикла.
Отдельная регрессия проверяет SIGINT/SIGTERM и смерть parent.

## Условия исполнения

- По умолчанию `preview`, без hardware. `mock` проверяет прежний servo
  без SDK соединения; только в mock разрешён `--auto-arm`.
- `tesollo` требует `--execute --source cameras`. Записи для физического
  backend этот CLI отклоняет до запуска процессов и соединений.
- После подключения состояние DISARMED. `a` + Enter вызывает прежний ARM:
  проверка готовности и seed из измеренной позы, без home/open команды.
- `d` — DISARM; `r` — явный RECOVER, остающийся DISARMED; `q`, закрытие
  viewer или Ctrl+C — завершение. FAULT не снимается автоматически.
- Штатные параметры сохранены: 60 Гц, 120°/с, 720°/с², прежний guard.
  Это параметры алгоритма, не измеренные пределы механизма. Потеря tracking
  удерживает последний принятый state; возврат допустимых точек продолжает
  следование без нового ARM, как в исходном проекте.

В режиме с controller preview получает **измеренные backend positions**,
а не выдаёт целевые углы за физически исполненные. `targets.jsonl`,
`executor/servo.jsonl` и `preview.jsonl` разделены. Подробный итог сохранён
в `executor/final.json`, признак подтверждённого disconnect — `closed=true`.
Неподтверждённое завершение возвращается как ошибка.

## Повторные прогоны без аппаратуры

Три полные RGB-D записи × два прохода с новым causal perception и
`--backend mock --auto-arm`. Исполнялся исходный Runtime с guard; mock
телеметрия не моделирует усилия и не подтверждает контакт с предметом.

| Эпизод от 2026-10-06 | Проход | Estimates | Valid targets | Hybrid contact frames |
|---|---:|---:|---:|---:|
| 11-00-24 | 1 | 120 | 95 | 84 |
| 11-00-24 | 2 | 119 | 92 | 78 |
| 11-15-00 | 1 | 84 | 69 | 38 |
| 11-15-00 | 2 | 84 | 71 | 40 |
| 11-18-17 | 1 | 81 | 68 | 38 |
| 11-18-17 | 2 | 82 | 73 | 45 |
| Всего | | 570 | 468 | 323 |

Все шесть завершились без control errors, по одному ARM, с подтверждённым
закрытием owner. Всего записано 2787 servo cycles, включая подготовку,
DISARMED и HOLD. Медианный период каждого прогона 16.663–16.668 мс,
p95 16.806–17.288 мс. Это измерение программного mock-контура на этом
компьютере, не частота обновления или точность физической DG5F.

[Числовой протокол](2026-10-19-dg5f-servo-results.json).
Локальные raw logs: `data/live_hand_servo_validation/repeat_1/` и `repeat_2/`.
Видео measured mock-позы:
`data/live_hand_sessions/dbbbbdc4b81c475c938c8c455fad8f73/preview.mp4`.

В native C++ fake SDK прошли проверки telemetry conversion, persistent
thread, stale-target gate, explicit recovery и thermal semantics. Это тест
существующего wrapper без настоящего устройства. Импорт собранного
`dg5f_python` и наличие нужных API проверены без создания DGApi instance.

Новые интеграционные проверки: DISARMED до ARM, source-clock expiry,
HOLD/resume без второго ARM, явный DISARM, RECOVER без ARM, надежный clear
mailbox, завершение по SIGTERM и при смерти parent. Режимы с real backend,
которые могли бы исполнять запись или auto-arm, отклоняются до соединения.

Полный suite: **602 passed, 7 skipped** в аппаратном image; **597 passed,
12 skipped** в базовом `live-test` без внешнего hand runtime. Дополнительно
проверен настоящий CLI на mock: после ACTIVE всей его process group отправлен
SIGINT, результат — exit code 0, `state_before_close=ACTIVE`, `closed=true`,
`armed=false`, без error и до конца записи. Локальный лог:
`data/live_hand_signal_validation/cli.log`.

## Запуск на стенде

[Команды сборки, Ethernet, запуска и ARM](live_dg5f_hand.md#аппаратная-рука).
Сначала нужны подключённые камеры и реальный Ethernet-маршрут к адресу руки.
Затем проверка telemetry/readiness в DISARMED, ARM и медленное следование
раскрытой кисти/одного пальца. Отдельно проверить потерю наблюдения, DISARM,
возврат tracking и лишь затем pinch. Нагрузку/контакт/стабильность физического
захвата эти offline/mock прогоны не подтверждают.
