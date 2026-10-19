# Совмещённый просмотр записанного онлайн-ретаргетинга

Реализована вкладка **Online review**: исходное облако RGB-D, кисть после
онлайн-фильтрации и робот по измеренным суставам MuJoCo показаны в одной сцене
и на одной шкале времени. Это просмотр результатов причинного инференса;
`plan.h5`, `cln.npz` и `rec.npz` не являются его источниками.

## Открыть подготовленный пример

Если сервер выключен:

```bash
docker compose run --rm live-web
```

Открой
[подготовленный прогон](http://localhost:8000/?online_session=27daacc0ad084015b4bb409853ec7fda#online-review).
Reference — `2026-10-06_11-15-00`. Исходные кадры поступали в runtime на
**оригинальной скорости ×1**, perception 15 Гц, исполнитель 50 Гц. Виртуальные
оси robot-base выбраны параллельно сохранённым осям calibration/world; это
конфигурация симуляции, не физическая регистрация настоящего UR.

Нажми ▶. Ползунок и ◄/► позволяют перематывать и идти покадрово. Скорость
просмотра меняет только воспроизведение журнала; уже выполненный инференс
не пересчитывается. `Hand detail` приближает кисть, `Fit robot + hand` возвращает
общий план. Слева включаются облако, кисть, робот и траектории.

- Жёлтый — online filtered hand, только доступные в этот момент оценки.
- Синий — UR10e по `q_observed`, не по отправленным командам.
- Розовый — целевой TCP при активном следовании.
- Голубой — FK TCP по измеренным суставам, в той же номинальной URDF.

Справа показаны состояние, причины остановки, возраст наблюдения, время
source→готовность оценки, номер оценки у perception/исполнителя и joint tracking
residual. Возраст наблюдения и source→estimate — программные временные метрики;
они не являются полной физической camera-to-actuator latency. Отсутствующие и
состарившиеся кисти скрываются. Торможение, unbound и faults не вырезаются.

Масштаб переноса движений в этом прогоне **0.25**. Для совмещения робот один
раз смещается к кисти по первой успешной привязке; масштаб самой модели не
увеличивается и последующие привязки её не телепортируют. Это визуальная
регистрация для анализа относительного движения. Реальное положение базы,
стол, инструмент и collision world этим просмотром не подтверждаются.

## Источники данных и время

`viki.live.RecordedReview` читает:

```text
reference/raw/timestamps.json    исходные sync/native timestamps и offsets
reference/raw/world_anchor.json общий world-display transform, если есть
reference/cloud/*.bin           готовое исходное облако
session/manifest.json            профиль, source speed и clock mapping
session/hand.jsonl               причинные оценки + source/generated time
session/executor/*/execution.h5  измеренные суставы, TCP, цели, state и events
session/session.json            итог и причина завершения
```

Новые `RecordedFrameSource` сохраняют `host_start_us`, `recorded_start_us` и
`playback_speed` до первого наблюдения. Для старых прогонов начало шкалы
восстанавливается только при единственном согласованном решении по native
timestamps всех записанных hand estimates и исходным offsets. Повторённые
исходные кадры учитываются; угадывания по FPS нет. Несогласованные записи
отвергаются с ошибкой.

На выбранном времени берутся:

1. Последний исходный кадр, чей sync tick уже наступил по source clock.
2. Последняя hand estimate, чей `generated_time_us` уже наступил. Её геометрия
   видна только при валидности и source age меньше TTL.
3. Последний measured robot sample до этого времени. Между будущими samples
   ничего не интерполируется; `q_commanded` не используется вместо `q_observed`.

Точки кисти, облако и робот сначала находятся в rig coordinates, затем общий
сохранённый world-display transform применяется к ним одинаково. FK строится
публичным `OnlineRetargeter.scene`, без решения IK, physics replay или RTDE.
Проверяется SHA исходного URDF. В текущей модели отображается только рука UR10e.

Номинальная сетка просмотра 50 Гц позволяет проверять последовательность
состояний. Если браузер не успевает, он пропускает визуальные кадры, сохраняя
скорость общей шкалы; вычисления завершённого инференса от этого не меняются.

## Проверенные прогоны 19 октября

| Параметр | Основной ×1 | Дополнительный ×0.5 |
|---|---:|---:|
| Session ID | `27daacc0ad084015b4bb409853ec7fda` | `9b53a009a7004a8794d755dd029c3636` |
| Hand estimates | 89 | 162 |
| Valid position | 78 | 149 |
| Logged control samples | 304 | 595 |
| Итог | stopped / source_exhausted | stopped / source_exhausted |
| Processing median / p95, ms | 44.02 / 53.61 | 41.59 / 53.15 |
| Source age median / p95, ms | 89.14 / 112.59 | 130.83 / 157.08 |

Основной прогон имеет 255 active-following samples и 258 accepted commands
вместе с торможением. Maximum command speed 0.10115 rad/s, discrete acceleration
0.50000 rad/s²; median/p95 host period 20.163/20.225 ms. Artifact verifier
подтвердил finite commands, command integration, sampled model margins,
закрытый HDF и соответствие числу логов. Это самосогласованность исполнения,
не абсолютная accuracy кисти или физического TCP. `verification.json` лежит
в папке основного прогона.

### Ошибка слежения в основном прогоне

По тем же 255 отсчётам, где `state == active` и `command_accepted`, посчитана
евклидова норма `target_position - tcp_position` в одном control sample.
`tcp_position` — FK по измеренным суставам MuJoCo в номинальной URDF;
обе позиции находятся в системе робота. Unbound и braking samples исключены:
в unbound целевая позиция может подменяться текущим TCP и занижать метрику.

| Метрика | Ошибка TCP, мм |
|---|---:|
| Средняя | 26.70 |
| Медиана | 24.72 |
| RMSE | 31.03 |
| P95 | 58.55 |
| Максимальная | 62.79 |

Максимальная по суставам абсолютная разница между измеренным положением и
предыдущей командой на начало цикла: в среднем 0.177°, p95 0.395°, максимум
0.432°. Это отдельная метрика исполнения суставных команд; она не заменяет
ошибку TCP относительно движущейся цели.

Условия: recorded input ×1, масштаб переноса 0.25, слежение только по позиции.
Это модельная ошибка слежения в пространстве робота, которая включает
динамическое отставание от цели. Она не измеряет абсолютную точность оценки
кисти, точность физического UR или ориентацию. Для разложения ошибки по
задержке, IK и ограничениям динамики нужен отдельный эксперимент.
Числа и SHA-256 исходного HDF сохранены в
[`tracking_error.json`](../data/live_sessions/27daacc0ad084015b4bb409853ec7fda/tracking_error.json).

## Проверка реализации

Hardware-independent tests проверяют availability-time selection, отсутствие
lookahead, скрытие stale/invalid hand, выбор measured joints, fixed registration,
source speed, legacy timestamps с повторёнными кадрами, model hash mismatch,
out-of-range запросы, directory/symlink escapes, world-display transform и
отказ от чтения ещё работающих журналов.

В headful Chrome проверены общая сцена, 21/21 points, active state, реальная
анимация, перемотка и восстановление состояния при обратном seek. Отдельно
подтверждено отсутствие запросов offline geometry/retarget в этом режиме.
[Browser report](../data/online_review_2026-10-19.json),
[снимок](../data/online_review_2026-10-19.png). Legacy review проверен отдельно:
[report](../data/online_review_legacy_2026-10-19.json).

Веб-просмотр использует только GET, hardware interfaces не создаёт. API:
`/api/live/reviews`, `/api/live/reviews/{id}`,
`/api/live/reviews/{id}/frames/{frame}`, `/api/live/reviews/{id}/cloud/{frame}`.
Store принадлежит lifespan приложения, два прогона в bounded cache;
доступные robot FK frames и браузерные cloud/frame caches также ограничены.

Полный `docker compose run --rm live-test`: **535 passed, 7 skipped**, 31.43 с.
`node --check` для новых JS и main.js, `git diff --check` и Compose config
пройдены. Исходный frozen first-run plan сохранил SHA-256
`dafab3957cacf9bef895322f7137d026f2fe7e6d4d335700faa5c97c0ca3eebe`.
