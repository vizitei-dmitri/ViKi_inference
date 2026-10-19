# Проверка реализации live human retarget — 13–14 октября 2026

Реализован отдельный поток `RGB-D → causal hand → relative binding → online IK
→ MuJoCo / UR10e`. Ниже **проверка программного контура и симуляции**.
Реальные камеры не запускались для этих опытов, движения физического UR не было.
Это не оценка абсолютной точности кисти, физического TCP или безопасности стенда.

## Реализованный путь и зависимости

`viki.live.LiveSession` владеет наблюдением; `CameraFrameSource` или
`RecordedFrameSource` выдают только новые группы. `LiveHandEstimator` держит
MediaPipe VIDEO-модель на каждую камеру. Профиль
`live-human-sdk-rays-causal-v1` использует точные SDK color rays, multi-view
triangulation, наблюдаемость/anatomy gates, начальный palm reference, One Euro
для XYZ и причинное SO(3) сглаживание. Будущие кадры, gap interpolation,
centered Savitzky–Golay, episode-wide fusion и batch hand-fit отсутствуют.
Offline frozen profiles и guarded baselines не менялись.

Выход — 21 точка запястья/пальцев, pinch-center, palm orientation, отдельные
validity/quality и source timestamps. Локоть/плечо не оцениваются. Depth-frame Z
не подставляется в color-frame геометрию как depth prior. Single-hand MediaPipe
не подтверждает identity association нескольких людей.

Последняя цель передаётся в bounded mailbox отдельного arm-owner process.
В нём постоянная UR10e/tool0 модель, шестивариантный QP, command continuity,
joint/velocity/acceleration/stopping-distance bounds, модельные floor и все
15 retained self-collision pairs. Концы и середина сегмента проверяются
нелинейно; это sampled model checks, не swept-volume certificate. Для далёких
пар применяется консервативная world AABB lower bound, для близких — signed FCL.
HDF5 журнал пишет отдельный процесс. GUI и ML не владеют циклом исполнения.

CPU images `live`, `live-web`, `live-test` построены через Compose; Kinect SDK
1.4.1, Python 3.10, Pinocchio 4.1.0, MediaPipe 0.10.35, MuJoCo 3.15.0 в финальном
образе. RTMPose/ONNX Runtime CPU доступны для обычного web model registry, но
live-профиль в этих опытах использует MediaPipe. Источник URDF/MJCF и SHA
сохраняются в отчётах. Для RTDE частота явно задаётся constructor/servoJ;
см. [официальный API ur-rtde](https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html).

## Частоты и отрицательные результаты

Начальные опыты 125 Гц показывали типичное IK около 4–7 мс, но редкие выбросы
9–11 мс нарушали бюджет 8 мс. Сначала причиной был GIL журналирования, затем
cyclic GC. Журнал вынесен в отдельный процесс; cyclic GC отключён только внутри
arm-owner после прогрева и восстанавливается после cleanup. Reference counting
и bounded caches/queues остаются.

**По умолчанию исполнитель 50 Гц, nominal dt 20 мс**, perception 15 Гц.
125 Гц остаются explicit experimental с тем же строгим deadline guard.
Это configured rate без hard real-time гарантии Linux/Python. Первый полный
прогон с параллельной нагрузкой сохранил fault
`live simulation exceeded servo period`: 162 samples, peak pre-servo compute
19.25 мс. Артефакт
`data/live_sessions/d1ec628b4abe40c6b2302fab1fccd072/` сохранён. Нельзя заменять
этот результат утверждением, что 50 Гц гарантированы при любой нагрузке.

Последующие полные прогоны без параллельного pytest завершились штатно, включая
независимый native MuJoCo GUI и Chrome/Three UI. Дедлайн остался включён.

## Полная цепочка на записанном RGB-D

Reference: `data/datasets/40mm-pnp/2026-10-06_11-15-00`, Kinect 0/1,
RGB 1920×1080, depth 640×576 NFOV_UNBINNED, requested 30 fps.
Используются только raw images и saved calibration, source speed 0.5.
MP/SDK/model warm-up происходит до создания source clock.
Retarget не читает `rec.npz`, `cln.npz` или `plan.h5`.

| Прогон | Позиция | Позиция + ориентация, native GUI |
|---|---:|---:|
| Session ID | `f648f093bffc432a86b926a3c1c9c1f9` | `2b0ea1d4c1ac44488d717a499ac4fdc5` |
| Hand estimates | 162 | 161 |
| Valid position / orientation | 149 / 149 | 149 / 149 |
| Logged control samples | 596 | 597 |
| Active-following samples | 173 | 176 |
| Accepted commands including braking | 176 | 183 |
| Hand processing median / p95, ms | 48.69 / 63.71 | 52.49 / 66.32 |
| Observation age median / p95, ms | 126.67 / 153.27 | 148.72 / 174.62 |
| IK median / p95 on accepted commands, ms | 6.30 / 9.73 | 7.12 / 11.15 |
| Maximum recorded compute, ms | 14.56 | 15.24 |
| Peak command velocity, rad/s | 0.04578 | 0.12491 |
| Peak discrete command acceleration, rad/s² | 0.50000 | 0.50000 |
| Maximum per-joint excursion, rad | 0.05130 | 0.14558 |

Оба прогона завершены `source_exhausted`, с executor `stopped`, закрытым HDF5
и без logging error. 596/597 cycles включают **unbound наблюдение**, а не
исполнение всей записанной траектории. При потере/невалидности входа runtime
сбрасывает привязку; `--auto-bind` не восстанавливает её автоматически. Для
продолжения после потери нужен ручной Bind. Начальные invalid samples —
9 кадров palm warm-up и пропуски позиции. Нормальное завершение не означает,
что все движения человека были исполнены.

Возраст измеряется от oldest retimed host-arrival исходного кадра до обработки;
это software observation age, не полная физическая camera-to-actuator latency.
Ретайминг recorded source не переименовывает processing time в measurement time.
Default TTL 250 мс проверяется по исходной шкале на управляющем цикле.
У SDK device timestamps и controller timestamps собственные шкалы.

После независимого чтения артефактов проверены nominal-dt command bounds,
`q_next − q = velocity × dt` между последовательными accepted samples,
согласованность числа логов, original source validity и nonnegative saved
model margins. `verification.json` лежит в каждой session directory. Эти
проверки измеряют программную самосогласованность, не physical accuracy.

```bash
docker compose run --rm --entrypoint python live scripts/verify_live_session.py \
  data/live_sessions/2b0ea1d4c1ac44488d717a499ac4fdc5 --require-motion
```

## Потеря входа и повторная привязка

Отдельный native MuJoCo lifecycle probe
`data/live-executor-probes/20261013T155606Z-3cf7991fae38/`: 8.135 с,
404 samples / 111 events, synthetic fresh human input 15 Гц. Лимиты здесь
строже: 0.05 rad/s, 0.1 rad/s². Проверены actual position-actuator motion,
input dropout 0.8 с, bounded brake → unbound, возврат входа без auto-resume,
новый manual bind → active, operator stop. Все четыре lifecycle assertions
пройдены. Peak command speed 0.009576 rad/s, acceleration 0.100000 rad/s²,
IK p95 8.013 мс, peak recorded compute 15.460 мс.

## Браузер и native GUI

Изолированный headless Chrome использовал реальный API и recorded/MuJoCo
runtime. [Отчёт](../data/live_browser_qa_2026-10-14.json) и
[снимок](../data/live_browser_qa_2026-10-14.png): 21/21 points, palm/TCP axes,
две Three views, manual Bind с matching acknowledgement, Clutch, переход на
Viewer и обратно с сохранением сессии и active badge, Stop, восстановление
формы с отключённым physical-motion checkbox. Browser errors/warnings: 0/0.

Первый browser attempt 13 октября не запускал сессию: `min=0.01, step=0.05`
делали default scale 0.25 недопустимым HTML input. Исправлено на `step=0.01`.
Первоначальный failed UI результат не является успешной проверкой; положительный
отчёт относится к повторному прогону 14 октября.

В browser run — 213 control samples / 53 observations, 21 accepted commands,
штатный operator stop. Browser software rendering увеличил perception p95 до
108.63 мс и observation age p95 до 205.92 мс; peak compute 17.51 мс. Выполнены
те же artifact checks. Низкий rendering FPS не задаёт servo dt, но общая нагрузка
на хост всё равно может повлиять на scheduling.

Native `--gui --orientation` прогон выше завершился штатно. GUI отображает
actual simulated joints отдельным процессом; physics и IK остаются у исполнителя.
Menagerie example base placement повёрнут относительно URDF: live runtime
использует identity base placement и compiled gravity compensation. Номинальные
Menagerie attachment site и URDF tool0 после этого всё равно расходятся примерно
на 17–18 мм. Это **различие моделей**, не ошибка измеренного физического TCP;
relative binding использует одну и ту же Pinocchio FK для начала и целей.
Controller RTDE TCP логируется отдельно.

## Что ещё не подтверждено

Live camera acquisition на текущем rig и физический UR10e не тестировались.
UR требует explicit enable, IP, подтверждённую rig→robot rotation, плоскость пола
и matching `source_calibration_id`; управляющий скрипт загружается только при
manual bind к stationary measured joints. Никаких auto-home/reset/auto-retry.
Replay/live используют общий robot ownership lock.

Текущий model scope — bare arm / configured floor / retained self-pairs.
Реальная стойка, стол, окружающие предметы, установленная клещевая модель и
человек не представлены как полный collision world. Hand driver, force/grasp
feedback, синхронные robot-execution camera observations, policy checkpoints,
Diffusion Policy/RL inference и обучение остаются отдельными этапами.

После последнего исправления полный `docker compose run --rm live-test`:
**525 passed, 7 skipped**, 30.86 с. Лог `/tmp/viki-live-final-tests.log`;
`node --check` для `live.js`, `live_scene.js`, `main.js`, Compose config и
`git diff --check` пройдены. Live-specific tests покрывают causality,
calibration identity, command bounds, model/SDK faults, startup cancellation,
logs, operator acknowledgements и cleanup.

Исходный архив SHA-256
`49fee7cb46d722d66bbb09ee590278b5712dbe07a7e1c84adaef2171a4ff0017`
и frozen first-run plan
`dafab3957cacf9bef895322f7137d026f2fe7e6d4d335700faa5c97c0ca3eebe`
повторно сверены без изменений. Для первого offline plan по-прежнему нужен
`--speed-scale 0.02`; дефолтные 0.1 этот конкретный plan отвергает по ускорению.
Это штатный guard, а не разрешение ускорять проверенный первый запуск.
