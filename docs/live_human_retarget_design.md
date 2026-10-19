# Онлайн-повторение движения человека на UR10e

Дата: 2026-10-13. Этот документ сохраняет **проект до реализации** и исходные
измерения. После исследования реализована отдельная live-ветка с профилем
`live-human-sdk-rays-causal-v1`, 21 точкой по умолчанию и исполнителем 50 Гц.
Текущие интерфейсы и отличия: [запуск](live_human_run.md),
[проверка реализации](2026-10-13-live-validation.md). Утверждения «ещё нет» ниже
относятся к моменту проектирования, а не к текущему коду.

Первый режим — камеры наблюдают руку человека, ViKi причинно оценивает её
движение и формирует новые цели UR10e. Сначала цели исполняются в MuJoCo.
Автономный inference обученной политики — следующий этап; для этого режима
checkpoint Diffusion Policy не требуется. Этот документ не подтверждает
безопасность движения на физическом стенде.

## Что уже есть и что придётся добавить

Сейчас камеры работают онлайн: `CameraManager` хранит ограниченные буферы,
ставит `Frame.host_timestamp_us` при получении кадра и предоставляет
`latest_frame`, `nearest_frame`, `recent_frames`. Основная обработка
`raw/ → rec.npz → cln.npz → plan.h5` остаётся офлайн.

Полезные существующие части:

- `viki/cameras/manager.py`: владение камерами и снимки буферов;
- `viki/perception/backends/mediapipe.py`: один долгоживущий VIDEO-detector на
  поток, 2-D landmarks в полном цветном изображении;
- `viki/perception/triangulate.py`: покадровая геометрия, угловые и
  reprojection gates, depth как дополнительное свидетельство;
- `viki/perception/hand_angles.py`: palm frame из WRIST, INDEX_MCP,
  MIDDLE_MCP, PINKY_MCP;
- `viki/retarget/frames.py`: явные rig/calibration/robot transforms;
- `viki/retarget/solver.py:986`: Pinocchio-модель, физические arm angles,
  Jacobians, floor support и self-collision margins;
- `viki/replay/ur.py`: identity/start checks, единый RTDE owner, watchdog,
  `servoJ`, отмена, telemetry и cleanup;
- `viki/replay/simulation.py`: проверенный офлайн MuJoCo preview. Потокового
  симулятора этот модуль пока не предоставляет.

Новые `LiveSession`, causal profile, frame-local public API, online IK,
streaming executor, live MuJoCo backend и UI управления сессией **предлагаются**.
Существующие `retarget_episode` и `RobotDriver.execute` не являются такими API.
`server/jobs.py` пригоден для запуска/остановки задач, но HTTP FIFO не должен
служить очередью управляющих целей.

## Короткий путь с существующими зависимостями

Для первой версии использовать MediaPipe → SDK-correct color rays →
triangulation семи опор → causal position target → incremental QP → joint servo.
Сохраняются NumPy/SciPy, `pin-pink>=4`, `pin>=3.5`, QP solvers и `ur-rtde`.
Модели и factory calibration загружаются один раз перед запуском сессии.
Дополнительная CUDA/SAM/episode hand-fit обработка для этой версии не нужна.

Предлагаемый набор `tri7`: WRIST(0), THUMB_CMC(1), THUMB_TIP(4), INDEX_MCP(5),
INDEX_TIP(8), MIDDLE_MCP(9), PINKY_MCP(17). Он содержит palm anchors и pinch
centre; MediaPipe продолжает выдавать 21 точку, но геометрическое refinement
исполняется только для необходимых семи. При выборе другого target anchor
состав опор фиксируется в новом профиле.

Для Kinect лучи должны строиться из **цветной** factory calibration через
backend-owned SDK projector, не из depth K и не путём повторной OpenCV
undistortion уже исправленных лучей. Цветные pixels остаются в полном
изображении; после crop/resize требуется точное обратное преобразование.
Глубина после SDK color→depth mapping — необязательное мягкое свидетельство,
не замена двухкамеральной геометрии. SDK-обращения остаются в `cameras/`;
планируется публичный интерфейс получения луча/проекции, который perception
получает через Protocol из `contracts.py`.

MediaPipe VIDEO mode использует tracking и требует возрастающих timestamps;
LIVE_STREAM mode асинхронен и может игнорировать новый вход при занятости.
Для первого implementation предпочтителен существующий синхронный VIDEO
backend в отдельном perception worker, с собственным latest-frame отбором.
[Официальный Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

Нельзя пересоздавать detector каждый кадр. Повторные/старые кадры отвергаются
**перед** вызовом backend: текущий `max(ts_ms, last_ts_ms + 1)` обеспечивает
совместимость API, но не доказывает новизну наблюдения. После переподключения
камеры начинается новая camera generation и detector state сбрасывается.
Handedness score не считать per-landmark confidence: текущий MediaPipe backend
его сознательно не использует. Смена наблюдаемой руки или неоднозначность
нескольких рук прекращает binding, а не переключает управление на другую руку.
Это требование нового режима, а не готовая возможность single-hand backend:
сейчас `num_hands=1` и он не возвращает сведения о второй руке. Для первой
версии нужна явно ограниченная сцена с одной рукой; обнаружение неоднозначности
потребует отдельного multi-hand/association API и проверки нового профиля.

## Отдельный причинный профиль

Предлагаемое имя: `live-human-tri7-position-v1`. Его ещё нет в registry.
Добавляется новая immutable recipe; существующие frozen profiles и защищённые
артефакты не меняются. Manifest фиксирует model hash, anchors, detector mode,
angular geometry gates, calibration, filter, timestamp и loss policies.

Порядок обработки нового frame group:

1. Проверить camera generation, sequence, возраст, calibration hash и
   cross-camera skew; выбрать только уже поступившие frames.
2. Детектировать 2-D опоры, восстановить полный pixel coordinate system.
3. Построить SDK color rays; проверить finite/depth/evidence masks и
   triangulation geometry по каждой нужной опоре.
4. Получить наблюдаемый target anchor и отдельный palm orientation candidate.
5. Обновить причинный filter только новым валидным наблюдением.
6. Применить session binding, workspace mapping и rate/quality guards.
7. Опубликовать цель с исходным timestamp и сроком действия.

Не использовать future gap filling, centered Savitzky–Golay,
episode-wide confidence normalization, whole-episode bone fitting или
пересчёт прошлых результатов после прихода будущего кадра. Проверка
`stable_palm_orientation_mask` тоже требует замены: она использует median по
эпизоду и может ретроспективно инвалидировать предыдущую ориентацию.

Для baseline выбрать causal One Euro на позиции, с `dt` между валидными
наблюдениями, параметрами в метрах и явным reset после потери binding.
Статический jitter и задержка измеряются раздельно; параметры не наследуются
молча от offline SG. [Авторский One Euro](https://gery.casiez.net/1euro/).
Альтернативный контроль — stateful Butterworth с переносом `zf → zi` между
кадрами. [SciPy sosfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html).

Маски `observed`, `geometry_valid`, `position_valid`, `orientation_valid`,
`filtered` и `stale` различаются. Конечное число после filter не превращает
пропуск в observed data. Первой версии достаточно двух свежих камер;
монокулярное продолжение движения не включается автоматически. После
пропуска разрешён только ограниченный по времени hold/deceleration, без
продолжительной экстраполяции движения человека.

## Часы, дедупликация и freshness

Host monotonic clock — единая шкала live runtime. Camera device timestamp и
RTDE controller timestamp сохраняются дополнительно; их epochs напрямую не
сравниваются. Для robot packets нужны controller timestamp **и** host receive
time. В существующем replay log controller time есть, отдельного packet-arrival
host timestamp и camera correlation ещё нет.

Каждое наблюдение несёт `(camera_id, generation, sequence, device_time,
host_receive_time)`. Каждая цель несёт source frame IDs, oldest source time,
generated-at, sequence, calibration generation и valid-until. Повторное чтение
последнего кадра не продлевает TTL. Filtering/IK/render также не обновляют
время источника. Переупорядоченные результаты старых inference jobs не
перезаписывают более новые цели.

Arrival timestamps дают практическую синхронизацию, но не точное время
экспозиции: USB/SDK buffering создаёт дополнительную задержку. В отчёте
отдельно хранить device/arrival skew и известную оценку exposure-to-arrival.
Без её измерения observation age на host — только доступная оценка возраста.

При аппаратно синхронных Kinect использовать физическую синхронизацию и
измеренный residual skew. Для свободных камер можно выбирать ближайшие
**доступные** frames с ограничением skew; нельзя вызывать offline
`_time_align_uv`, использующий будущие samples. Более сложная фиксированная
задержка/ray prediction допустима лишь как отдельная версия профиля с
измеренной ошибкой. Missing pixels и непроверенная область — unknown.

## Первое управление: относительная позиция и clutch

Сначала только позиция. При ручном bind/clutch сохранить свежую filtered human
anchor `p_H0`, фактическую robot joint configuration `q_R0`, robot task-point
reference `p_R0/R_R0`, выбранный tool и calibration generation. Не исполнять `home_q` и не
использовать `q_approach` из offline plan.

```text
p_R_target = p_R0 + scale * R_RH * (p_H_filtered - p_H0)
R_R_target = R_R0
```

`R_RH` — проверенное преобразование направления rig→robot; `scale` — явная
безразмерная настройка. Relative mode требует правильных осей, workspace и
tool model, даже если не требует совпадения абсолютных human/robot origins.
Начальная reference должна совпадать с FK той же модели при измеренной
`q_R0`, которая используется в IK; frozen human reference — с тем же filtered
sample. Тогда нулевое human displacement не создаёт initial correction.
Active UR TCP также записывается и сверяется с model FK. Если calibrated UR
FK и generic Pinocchio FK расходятся, нельзя взять actual TCP за target0 и
заставить другую модель компенсировать это расхождение движением при bind.
Нужна согласованная factory/tool calibration либо явно ограниченный relative
model-space прототип с измеренным FK disagreement; абсолютная метрическая
accuracy в этом случае не заявляется.
Clutch release прекращает следование; новый bind снова якорится в фактической
позе робота и текущем положении человека. Возвращение руки в кадр после
occlusion не возобновляет движение автоматически и не догоняет накопившееся
смещение. Нужен ручной rebind.

Для bare arm task point — явно выбранная flange/installed TCP pose. Виртуальный
Robotiq из offline retarget не является измеренным bare-arm TCP. Не смешивать
динамический gripper contact midpoint с `getActualTCPPose`; действующая
pendant TCP definition фиксируется при запуске. Физическая модель, tool,
payload и calibration должны соответствовать исполнению.

Предлагаемый `ToolSpec` явно задаёт flange→TCP offset и model/installed-tool
identity. Для первой версии выбирается bare-arm/reference TCP. Если виртуальная
клешня сохраняется в визуализации, она помечается model-only. Её opening из
человеческого pinch не выдаётся за actual tool state и не двигает
оптимизируемый TCP аппаратного bare-arm режима. В assembled Pinocchio-модели
все passive/tool DOFs фиксируются явной конфигурацией; неизвестная геометрия
физического инструмента блокирует модельные claims для этого инструмента.

Factory UR kinematics calibration, camera extrinsics/rig anchor и измеренная
регистрация robot base относительно rig — три разных преобразования.
Загрузка заводской UR calibration не измеряет camera→robot registration.
Текущие default base position/RPY не считать измеренной регистрацией стенда;
live binding не подтверждает правильность осей или физического floor plane.

Ориентацию включать отдельным этапом после position trials. Все четыре palm
anchors должны иметь свежие геометрические свидетельства, достаточную длину и
невырожденный MCP spread. Сохраняется MCP-based frame, не thumb-based normal.
Запрещены identity fallback и смена знака нормали как будто это новая
наблюдаемая команда. Геодезическое приращение
`Log(R_previous.T @ R_observed)` проверяется по фактическому `dt`, затем
фильтруется и ограничивается на SO(3); Euler components независимо не
сглаживаются. Relative orientation задаётся проверенным conjugated human
increment относительно `R_R0`, с фиксированной palm/tool correspondence.
При invalid orientation сначала прекращать ориентационное следование; policy
для продолжения position фиксируется отдельно, без скрытого fallback.

## Incremental IK и явная граница по времени

Выбор для минимального пути — Pink/Pinocchio/QP, уже используемые проектом,
с собственным публичным `OnlineRetargeter.step` в `retarget/`. Это **новый API**.
Модель, collision data и solver workspace держатся в памяти.
[Pink differential IK](https://github.com/pink-kinematics/pink/blob/main/pink/solve_ik.py)
возвращает velocity и по умолчанию учитывает configuration/velocity limits;
[AccelerationLimit](https://github.com/pink-kinematics/pink/blob/main/pink/limits/acceleration_limit.py)
добавляется явно и требует истории интеграции. Custom floor и physical UR
angle constraints не возникают из default solver автоматически.

`solve_sequential_baseline` не является готовой заменой: он передаёт `T=1`
в batch solver; матрица временных разностей пуста и ограничение перехода от
предыдущей q отсутствует (`solver.py:559`, `:648`, `:889`). Post-hoc SG затем
может нарушить floor/collision/velocity margins.

Различать `dt_obs` для фильтра наблюдений, `dt_ik` для retarget update и
`dt_servo` для фактически отправленных команд. Ускорения нельзя оценивать
разностью двух IK knots, а считать доказанными для interpolated servo stream.
Состояние QP включает свежую measured `q_m`, последнюю **отправленную**
`q_c`, её command velocity `u_prev`, actual qd, send time/planned apply time и
актуальную tool configuration. Точное время применения нельзя приписывать
команде только по host send timestamp. При запуске stationary arm задаёт
`q_c=q_m`, `u_prev=0`.

Пример конечного шага; неизвестная `u` — velocity между command samples:

```text
q_next = q_c + dt * u
delta_from_measured = q_next - q_m

minimize  ||W_p * (J_p(q_m) * delta_from_measured - k_p*dt*e_p)||²
        + ||W_R * (J_R(q_m) * delta_from_measured - k_R*dt*e_R)||²
        + lambda_v * ||u||² + lambda_a * ||u-u_prev||²

subject to
  q_min + joint_margin <= q_next <= q_max - joint_margin
  -v_max <= u <= v_max
  -a_max*dt <= u-u_prev <= a_max*dt
  floor(q_m) + J_floor(q_m)*delta_from_measured >= required_floor_margin
  distance(q_m) + J_distance(q_m)*delta_from_measured >= required_pair_margin
```

Первый position stage удерживает robot orientation с ограниченным весом;
при несовместимой цели QP не должен компенсировать её произвольной сменой
ветви IK. Reference posture — local continuity, не тяга к hardware home.
Угол каждого UR joint остаётся абсолютным физическим углом: нельзя wrap
через ±π или использовать Pinocchio cosine/sine embedding как actuator vector.
Близость к singularity диагностируется по Jacobian; большие joint velocities
и conditioning приводят к уменьшению цели/остановке, а не увеличению gain.

QP linearization не доказывает collision-free nonlinear step. После решения
проверить точные floor/self-collision margins и собственно **исполняемый**
segment, применить ограниченное feasibility-preserving backtracking либо
отказ. Floor support распространяется на movable arm/tool collision bodies,
а не только TCP. Существующий список self-collision pairs сохраняет свой
модельный scope; окружение и движущийся человек отдельной моделью пока не
проверяются. Уже нарушенные measured margins — fault, не повод рассчитывать
автоматический recovery trajectory.

При IK реже servo нужен ограниченный joint segment с начальными `q_c/u_prev`,
конечными состояниями и временами. Executor проверяет его финальные samples
на своих `dt_servo` и не вставляет линейный knot, создающий скачок скорости.
Изменение target прерывает только ещё не исполняемую часть и начинается из
текущего command state. Упрощённый pseudocode показывает правила, не готовый
код управления:

```text
on_new_camera_group:
    reject duplicate, stale, mixed-generation or excessive-skew sources
    detect → SDK rays → tri7 → evidence masks → causal filter
    if not bound or source invalid: publish invalid/loss event
    else: publish mapped target, source IDs/time, expiry, sequence

executor_tick:
    now, measured = fresh_robot_state()
    dt = checked_elapsed_command_time(now)  # no catch-up burst
    reject stale telemetry, unsafe runtime, discontinuity or wrong generation
    goal = newest_valid_target()            # reading again does not renew TTL
    if goal absent/expired/clutch released:
        bounded_deceleration(); clear_binding(); require_manual_rebind()
    else:
        segment = newest_validated_candidate_or_active_segment()
        reject candidate with stale boundary/generation/planned start
        check next sample continuity, age, v/a and tracking
        send next six-joint sample; update command history only on acceptance
    publish telemetry to bounded log ring; no GUI/disk I/O in tick

retarget_update:  # independent scheduled work; never queue unlimited targets
    snapshot latest goal, measured q and executor boundary/command state
    compute bounded_incremental_qp with explicit planned start/boundary
    check exact model margins and final servo samples
    publish candidate segment with source expiry and boundary identity
```

Протокол commit будущего boundary state реализуется до аппаратного режима.
Если за время вычисления boundary изменился, candidate отбрасывается, а не
сдвигается в новую позу без пересчёта. В отсутствие нового candidate executor
исполняет только уже проверенную часть до её срока действия, затем замедляется.
Другой допустимый первый вариант — ограниченный локальный IK внутри executor,
но только после измерения его worst-case budget на 8 ms period; произвольный
блокирующий solve в servo tick недопустим.

## Исполнитель и состояние сессии

Предлагаемый `viki/live` оркестрирует stages через `contracts` и их публичные
`__init__`. Frame primitives, causal preparation и online retarget пока не
экспортируются публично; их выделение — отдельная будущая задача. Импорт
соседских `_Cam`, private floor helpers или `_robot_lock` из live запрещён.
Камеры/SDK принадлежат `cameras`, solver — `retarget`, общий UR controller
owner/lock — `replay`; supervisor/server получают public session interface.

Один процесс владеет RTDEControl и receive state, а perception работает
независимо. Retarget вычисления могут иметь отдельного worker, но command
history и окончательное решение о подаче принадлежат executor. Camera
buffer→perception и perception→retarget используют bounded latest mailbox;
новая цель заменяет необработанную старую. Command/observation logging имеет
собственный ограниченный ring и writer. Переполнение recorder отмечается в
логе, не блокирует actuator tick. GUI получает downsampled snapshots.
Нельзя запускать независимый CameraManager на одном устройстве рядом с
существующим web manager; camera owner выбирается при запуске сессии.

State machine: `disconnected → observing → ready_unbound → bound → tracking`.
Clutch release/observation loss ведёт через controlled stop в
`ready_unbound`; fault требует явного устранения причины и новой сессии или
rebind по разрешённому переходу. Объект `LiveSession` собирается в lifespan;
handlers через DI запускают/останавливают сессию и читают статус. Они не
вычисляют frames/IK и не выполняют отдельные RTDE-вызовы.

Предлагаемые DTO в `contracts.py`: `HumanTarget`, `RobotObservation`,
`TimedJointSegment`, `LiveStatus`. В них нужны units/frame names,
source IDs/timestamps, masks/quality, command sequence, deadline,
tool/calibration/model/profile hashes. Arm state и tool state разделены.
Для будущей гуманоидной руки tool commands — именованный `(D,)` vector с
per-channel semantics/limits/availability; не фиксированное `6+1`.
В первой версии физический инструмент не управляется и не считается
измеренным: unavailable state остаётся unknown, не нулём.
UR RTDE action всегда содержит шесть **физических arm angles** в каноническом
порядке shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3. Tool
actuator vector идёт отдельному владельцу driver, когда он появится.

RobotObservation должен быть согласованным снимком одного RTDE packet или
проверенной receive-cache generation. Независимые getters без контроля
generation могут смешать q и timestamp соседних пакетов. Packet fields,
controller timestamp и host receive time публикуются вместе.
Сейчас high-level getters не дают подтверждённого packet-arrival callback.
Для cache adapter можно проверить controller timestamp до/после чтения полей
и сохранить host snapshot-read time; это не измеренное packet-arrival time.
Точное arrival stamping потребует соответствующего receive transport API.

RTDE reuse сохраняет read-only identity/remote/stationary checks, controller
safety gates, watchdog, tracking residual, independent cleanup и общую
блокировку replay/live. `RTDEControlInterface` не thread-safe, а параметры
`servoJ` speed/acceleration сейчас не используются, поэтому пределы обязаны
обеспечиваться самим command stream.
[SDU API](https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html).

В отличие от offline replay, runtime должен читать `speed_scaling` и
`target_speed_fraction`, учитывать фактическое замедление и не догонять
wall-time trajectory burst. Первая политика — остановить human tracking при
pause/нулевом scaling или недопустимом lag и потребовать rebind; управление
slider автоматически не повышается. Небольшое допустимое scaling учитывается
в прогрессе segment, без накопления human history. Детальная формула scaling
требует UR trial; она не считается уже реализованной.
[Официальный UR scaled controller](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_controllers/doc/index.html).

Missed deadline не исправляется быстрым исполнением пропущенных samples.
Stale **human goal** при свежей robot telemetry вызывает bounded deceleration
и unbind. Потеря robot telemetry/контроллерного соединения — fault с
предусмотренными `servoStop/stopScript` и controller watchdog; исполнение
нового рассчитанного braking segment при неизвестном состоянии запрещено.
Программная остановка и модельные bounds не являются сертифицированным
hardware safety controller.

## Выбор backend и зависимости

| Вариант | Для первой версии | Дополнительная работа |
|---|---|---|
| Pink/Pinocchio + direct RTDE | Предлагаемый короткий путь; совместим с текущими Python packages и UR guard/log logic | Causal API, physical angle/command-history constraints, floor/self-collision nonlinear checks, independent executor |
| UR ROS 2 Driver + MoveIt Servo | Альтернативный backend при нужде в PlanningScene, других роботах и более сложной collision интеграции | ROS version/container, URDF/SRDF, kinematics config, External Control, factory calibration, status/timeout integration |

MoveIt Servo предоставляет joint/Cartesian inputs, singularity/collision
checking и smoothing. Эти модельные функции полезны, но не заменяют
правильную perception freshness и регистрацию окружающих объектов.
[MoveIt Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html).
UR forward controllers требуют достижимых команд от клиента; нельзя
переносить поведение scaled trajectory controller на arbitrary streaming.
[UR controllers](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/usage/controllers.html).

ROS migration сейчас не обязательна. Если выбирается этот вариант, закрепить
дистрибутив и соответствующую ветку драйвера: current main docs относятся к
Rolling, не подтверждают наличие всех возможностей в Humble.
[UR installation](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/installation.html).
Для geometric TCP tracking нужна также заводская калибровка конкретного UR;
generic модель может иметь centimetre-scale расхождения.
[UR calibration](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/robot_setup.html).

## Измеренные компоненты и предлагаемые бюджеты

Покадровой end-to-end производительности live pipeline ещё нет. Отдельный
recorded-input benchmark MediaPipe выполнен; incremental IK, live acquisition
и поведение под camera/render load ещё требуют измерения.

Локальный probe на 80 сохранённых 2-D observations измерил **только**
frame-local V4 triangulation:

| Опоры | Median | p95 | p99 | Max |
|---|---:|---:|---:|---:|
| Все 21 | 19.39 ms | 40.98 ms | 68.99 ms | 74.47 ms |
| tri7 | 5.92 ms | 6.35 ms | 7.77 ms | 11.90 ms |

Источник: локальный
[`data/live_geometry_profile_2026-10-13.json`](../data/live_geometry_profile_2026-10-13.json).
Исключены камера, detector, depth sampling, filters, IK, servo и rendering;
это не доказательство 15/30 FPS полной системы и не сравнение с ground truth.
Software EGL render probe дал примерно 5–7 FPS на текущем пути. Такой render
не должен задавать servo или perception frequency; GUI вынесен из цикла.

Дополнительный benchmark полного frame-local perception на RGB-D записи:
два persistent MediaPipe VIDEO detector, preparation, SDK depth sampling и
V4 triangulation; 10 warmup +50 measured paired groups. Для all21 median
41.82 ms / p95 52.61 ms; для arm7 median 31.28 ms / p95 40.52 ms.
Обе ветки решили все anchors на 50/50 группах. Исключены disk decode/read,
sensor latency, live sync, causal filters, hand fit, IK, servo и rendering.
Это component coverage/latency, не ground truth или гарантия online 20 Hz.
Повторные прогоны дали arm7 p95 около40–46 ms.
Скрипт: [`benchmark_live_perception.py`](../scripts/benchmark_live_perception.py).
Данные: [`VIDEO report`](../data/live_perception_benchmark_2026-10-13.json).

В выбранной записи камеры были запрошены как `NFOV_UNBINNED`, 30 fps;
median интервал accepted sync ticks — 33.409 ms. Это свидетельство о записи,
а не замер камер текущей live сессии. Ограничение 15 Hz для WFOV_UNBINNED
не относится к этим NFOV данным.

Стартовые **цели**, подлежащие проверке, а не достигнутые гарантии:

| Участок | Цель первого prototype |
|---|---|
| Camera acquisition | Измерить delivered fps выбранного режима; запись была NFOV_UNBINNED / requested30, live frequency ещё не подтверждена |
| Perception worker | Tick 20 Hz с latest-frame selection; обработка только нового кадра, без повторного inference |
| Target observation | Не больше min(delivered camera rate, sustainable perception rate); drop старых frames при перегрузке |
| Retarget updates | 25–50 Hz для MuJoCo profiling, persistent model; частоту выбрать по p99 budget |
| UR executor | 125 Hz, 8 ms command period; 500 Hz не включать до отдельного timing trial |
| Perception+geometry | Целевой processing budget <50 ms для20Hz; measured arm7 p95≈40.5ms без filters/IK, оставшегося бюджета ещё недостаточно для гарантии |
| Executor work | p99 полезной работы целиться <4 ms при period 8 ms, с отдельным max/deadline guard |
| Human source TTL | Например 200 ms для первого prototype; зафиксировать, проверить stop latency и обосновать по measured tails |
| Camera skew | Verified wired pair — native residual tolerance; для свободных камер, например≤20ms, после проверки timing/coverage |

При неподходящем p99 менять архитектуру/нагрузку, а не объявлять соответствие
только по среднему FPS. Контроллер e-Series поддерживает RTDE до 500 Hz, но
может пропускать output packets. Это не hard-real-time гарантия Python/GPU
runtime. [Текущий UR RTDE guide](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html).
Low-latency/PREEMPT_RT kernel и приоритеты уменьшают jitter; параметры среды
фиксируются в benchmark report.
[SDU real-time setup](https://sdurobotics.gitlab.io/ur_rtde/pages/core_concepts/guides/realtime_setup.html).

## Этапы реализации и критерии приёмки

1. **Recorded causal replay.** Воспроизвести raw frames/observations в порядке
   arrival без доступа к будущему; сравнить tri7 geometry/coverage с офлайн
   артефактами, измерить jitter, phase lag и время компонентов. Offline
   результат — сравнение методов, не физический ground truth.
2. **Live MuJoCo position mode.** Камеры, detector и relative binding работают
   реально; joint backend — симулятор. Ввести clutch, loss/rebind, bounded
   segments и независимый recorder. Начать без ориентационного следования.
3. **UR read-only observations.** Проверить identity, actual q/qd/TCP,
   timestamps, packet freshness, scaling, installed tool и robot calibration;
   control script и motion пока не запускать.
4. **Staged arm motion.** После review физического workspace — stationary bind,
   малые relative displacement, ограниченные duration/velocity/acceleration,
   отмена и возврат в unbound. Затем orientation и расширение workspace как
   отдельные проверяемые этапы.
5. **Policy producer позже.** Подключить checkpoint к тем же observation/action
   contracts и executor после отдельного обучения/экспорта. Логи human raw и
   robot execution сохраняют различную provenance.

Для будущей автономной политики дополнительно проверить camera viewpoint и
scene conditioning: текущие человеческие демонстрации и teleoperation videos
могут содержать человека, тогда как автономное исполнение его не содержит.
Совпадение формата RGB не устраняет этот distribution mismatch. Camera
observations во время robot execution записываются отдельно и синхронизируются
с actual states/commands; старый human raw не переименовывается в robot view.

Обязательные checks для первой реализации:

- Prefix invariance: выход до `t` одинаков при любой замене/удалении будущих
  frames; ни filter, ни gate не меняют уже опубликованные команды.
- Повторы не продлевают TTL; out-of-order jobs, camera reset и mixed calibration
  generations не управляют рукой; очереди остаются ограниченными.
- Новый человек/рука, occlusion и clutch release прекращают binding; возврат
  цели не создаёт catch-up motion до ручного rebind.
- При bind и неподвижном human target commanded q остаётся начальной measured
  q: различие model/controller FK не порождает скрытую Cartesian correction.
- При потере input с валидной telemetry исполняется измеряемое ограниченное
  замедление; при stale robot state срабатывает fault path, не расчёт движения
  на старой q. Проверить fault-injection disconnect, reversed timestamps,
  failed command acknowledgement, infeasible QP и missed deadline.
- Все финальные samples соблюдают абсолютные q bounds и дискретные v/a limits
  на recorded command times, включая старт, replacement segment и stop.
  Переход через ±π не меняет winding.
- Exact floor/self-collision model checks выполняются на исполняемом segment;
  отрицательный measured margin, degraded geometry и singularity приводят к
  указанному reject/decelerate/fault результату. Scope фиксируется в отчёте.
- Render/log writer stalls и нагрузка камер не блокируют executor. Измерить
  p50/p95/p99/max observation age, stage processing latency, servo jitter,
  dropped frames, camera skew, tracking residual и stop latency.
- Replay и live не могут одновременно владеть UR; shutdown/cancellation
  выполняются владельцем RTDE. Любая handled failure сохраняет partial log.
- Log связывает source frames, filtered target, QP result, sent command,
  acknowledgement и measured state. Unknown tool feedback не выдаётся за
  measurement; motion trial не объявляется grasp success.

Предлагаемые, **ещё недоступные** интерфейсы: `viki live --backend mujoco`,
`viki live --backend ur10e`, `/api/live/start`, `/api/live/bind`,
`/api/live/clutch`, `/api/live/stop`, `/api/live/status`. Их точные параметры
должны появиться вместе с реализацией и тестами; запускать их сейчас нельзя.
