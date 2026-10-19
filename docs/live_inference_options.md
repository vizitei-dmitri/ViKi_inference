# Готовые компоненты для живой имитации и последующей policy inference

Дата проверки источников: **13 октября 2026 года**.

После исследования реализован первый human-following режим. Текущие
[инструкции](live_human_run.md) и [проверки](2026-10-13-live-validation.md)
описаны отдельно; сравнение ниже сохраняет исходную исследовательскую оценку.

**Статус: исследование и рекомендация, не реализация online-режима.** Первый
запрошенный режим — человек двигает рукой перед камерами, ViKi оценивает её
позу, а UR10e следует за ограниченной роботной целью. Обученная политика,
checkpoint и обучение для этого режима не требуются. Diffusion Policy/RL —
последующий источник действий для того же исполнителя.

Текущее устройство и зависимости разобраны в
[аудите проекта](2026-10-13-live-project-audit.md). Предлагаемые контракты,
порядок обработки и этапы проверки описаны в
[проекте живого ретаргетинга](live_human_retarget_design.md).

## Выбор для первого режима

Рекомендуется сохранить camera SDK и геометрию ViKi, использовать существующий
MediaPipe backend в `VIDEO`-режиме, добавить причинный фильтр и инкрементальный
IK на уже используемых Pinocchio/Pink/QP-зависимостях. Исполнитель — отдельный
владелец UR RTDE, принимающий ограниченные цели с временными метками.
Существующий replay исполняет законченный план; приём новых целей во время
исполнения ещё нужно реализовать.

Частоты perception, IK и servo задаются отдельно. Конкретные рабочие значения
выбираются после измерения задержек, пропусков кадров и длительности IK на этом
компьютере. Частота viewer не является измерением частоты controller.

| Компонент | Готовая возможность | Решение для ViKi |
|---|---|---|
| [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) | Tracking в `VIDEO` и `LIVE_STREAM`; временные метки | Сначала `VIDEO` в отдельном perception worker; API уже есть в backend |
| [1€ Filter](https://gery.casiez.net/1euro/) / [SciPy `sosfilt`](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.signal.sosfilt.html) | Фильтрация по текущим и прошлым отсчётам, сохраняемое состояние | Сравнить jitter и lag на записанном потоке; не переносить целый offline smoother |
| [Pink](https://github.com/pink-kinematics/pink) | Differential IK, QP, ограничения configuration/velocity | Использовать уже доступные зависимости и геометрию; новый online solver оформить отдельно |
| [Ruckig](https://docs.ruckig.com/tutorial.html) | Генерация перехода с ограничениями velocity, acceleration, jerk | Возможный дополнительный слой между IK и RTDE; не обязательная зависимость первого прототипа |
| [ur-rtde](https://sdurobotics.gitlab.io/ur_rtde/pages/examples/high_frequency_servoing/servoj_example.html) | `servoJ`, `initPeriod`/`waitPeriod`, чтение состояния | Ближайший путь: расширить существующий UR executor для потока целей |
| [UR ROS 2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) + [MoveIt Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html) | Готовый hardware driver, servo IK, planning scene и проверки | Альтернатива при необходимости полного ROS motion stack; больше интеграционной работы |
| [Stanford Diffusion Policy RealEnv](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/real_world/real_env.py) | Независимые observation/control loops, timed actions, shared memory | Архитектурный образец пригоден для живой имитации уже сейчас; policy необязательна |
| [LeRobot](https://huggingface.co/docs/lerobot/v0.6.1/inference) / [robomimic](https://robomimic.github.io/docs/modules/algorithms.html) | Загрузка обученной модели, rollout, запись/обучение | Позднее подключить как замену источника целей |

## Hand tracking и причинная обработка

MediaPipe `detect_for_video(image, timestamp_ms)` блокирует вызывающий поток,
но использует tracking между кадрами. Это совместимо с текущим синхронным
`HandPoseBackend.detect()`; название `VIDEO` не требует сначала записать файл.
Нужен один экземпляр на camera stream и возрастающие timestamps.
`LIVE_STREAM` использует `detect_async()` и callback; когда task занят, новые
кадры могут быть отброшены. Поэтому callback-результат нужно связывать с
исходным timestamp, а не считать, что каждая отправка вернула результат.
[Официальное описание режимов](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker).

Для первого режима предпочтителен `VIDEO` worker с ограниченным входным
буфером: этот вариант сохраняет текущий backend-контракт. `LIVE_STREAM`
имеет смысл после измерения задержек, если отдельная асинхронная выдача даёт
преимущество. Оба варианта требуют обработки потери наблюдения и устаревания;
tracking SDK сам по себе не разрешает продолжать движение при пропавшей руке.

1€ Filter меняет cutoff в зависимости от скорости: подавляет jitter при
медленном движении и уменьшает lag при быстром. Автор предоставляет реализации
и процедуру настройки; параметры зависят от единиц и характера сигнала.
Это кандидат для сравнения, не подтверждённое улучшение качества ViKi.
[Страница авторов](https://gery.casiez.net/1euro/).

Альтернатива без новой зависимости — SciPy IIR через `sosfilt(..., zi=state)`
с передачей возвращённого `zf` в следующий вызов. Для фиксированных
коэффициентов нужна согласованная временная сетка. Положение, ориентация и
gripper opening требуют своих представлений; линейное сглаживание девяти
элементов rotation matrix не гарантирует корректную rotation.
[SciPy 1.11.4 API](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.signal.sosfilt.html).

Текущий `viki/dsp.py` вызывает `savgol_filter` над последовательностью, а
prepare интерполирует пробелы между наблюдениями. Центрированное окно и
интерполяция с известным правым концом используют будущие данные. Их полное
поведение нельзя сохранить в причинном online-режиме без дополнительной
задержки. `mode="interp"` — режим краёв массива, не переключатель причинности.
[SciPy Savitzky–Golay API](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.signal.savgol_filter.html),
[локальная обработка](../viki/dsp.py), [prepare](../viki/prepare/README.md).

## IK, ограничения движения и выбор исполнителя

Pink предоставляет differential IK: вычисление скорости из текущей
configuration и задач, затем интеграция на шаг времени. Configuration и
velocity limits берутся из robot model; дополнительные limits/barriers нужно
явно проверить для выбранной версии и модели. В ViKi Pink/Pinocchio уже
используются для configuration и collision geometry, но нынешний whole-trajectory
solver не становится online controller от изменения одного флага.
[Pink README](https://github.com/pink-kinematics/pink),
[текущий solver](../viki/retarget/solver.py).

Рекомендация: сохранять robot model и кэш FK/Jacobian, обновлять цель и
решать ограниченную локальную задачу. Начальное состояние должно учитывать
измеренную позу руки робота. Недостижимая цель, потеря hand observations или
неуспех QP должны давать определённый переход к остановке. Ограничения пола,
суставов и self-collision не описывают реальные стол и предметы без их моделей.

Ruckig рассчитан на переходы между состояниями с ограничениями скорости,
ускорения и jerk. Можно рассмотреть локальную Community state-to-state
генерацию. В документированном Community режиме intermediate waypoints
переключают расчёт на cloud API, который не подходит для realtime control;
отдельный предиктивный Tracking interface доступен в Pro. Ruckig не следует
считать collision checker: его вход описывает состояния и кинематические
пределы, а проверка геометрии остаётся отдельной задачей. После его сглаживания
нужно проверять полученный путь, а не только исходные IK targets.
[Официальный tutorial](https://docs.ruckig.com/tutorial.html).

Для ближайшего шага прямой RTDE требует меньше изменений: есть UR10e driver,
проверки старта, watchdog и журналы. Их нужно сохранить и отделить
высокочастотную отправку `servoJ` от perception/IK, чтобы вычисление камеры не
блокировало servo period. Только один процесс владеет control connection;
остальные передают цели. Это предлагаемый новый контракт поверх
[текущего replay](../viki/replay/README.md), не уже готовая функция.
`ur-rtde` документирует управление period через `initPeriod`/`waitPeriod`.
[ServoJ example](https://sdurobotics.gitlab.io/ur_rtde/pages/examples/high_frequency_servoing/servoj_example.html).

ROS 2 + MoveIt Servo полезны, когда нужен planning scene, готовая проверка
singularities/collisions и стандартные robot controllers. Rolling Servo
принимает joint velocities, EE twist и EE pose; collision checks зависят от
включённых опций и содержимого planning scene. Это не автоматическая
модель препятствий стенда. [MoveIt Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html).

При выборе ROS необходимо согласовать distro, branch, controllers и версии
MoveIt: у UR driver отдельные `humble`, `jazzy` и другие ветки, а `main`
соответствует Rolling. Наличие старого Humble Docker image не подтверждает
совместимость с API из текущего Rolling tutorial. Заводскую кинематическую
калибровку конкретного UR рекомендуется извлечь и проверить; отдельно нужна
регистрация камеры/доски относительно физического robot base и реального TCP.
[UR release matrix](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver#release-status),
[UR calibration](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/jazzy/doc/ur_robot_driver/ur_calibration/doc/index.html).

## Что переиспользовать у Diffusion Policy сейчас

В Stanford `RealEnv` методы `get_obs()` и `exec_actions(actions, timestamps)`
разделены. Камеры и robot states согласуются на временную сетку; действия
назначаются на будущие моменты, просроченные точки отбрасываются. Controller
работает в отдельном процессе, использует shared-memory buffers и
интерполирует цели независимо от producer. Точно такое разделение применимо
к human-mimic producer без обучения нейросети.
[Pinned RealEnv](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/real_world/real_env.py),
[pinned controller](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/real_world/rtde_interpolation_controller.py).

Переносить следует архитектуру, не аппаратные defaults. Upstream использует
MultiRealSense и UR5 CB3, Cartesian `servoL`, на старте может вызвать `setTcp`
и опционально `moveJ`. `eval_real_robot.py` — пример planar Push-T, где
action изменяет только XY и область clip задана в коде. Для UR10e + Azure
Kinect + текущего joint `servoJ` нужны свои адаптеры и проверенная регистрация.
[Controller source](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/real_world/rtde_interpolation_controller.py),
[evaluation source](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/eval_real_robot.py).

## Позднее: обученная policy, async chunks и RL

LeRobot v0.6.1 предоставляет model loading, observation/action processors,
rollout и отдельные `PolicyServer`/`RobotClient` для async inference. Но в
built-in robot adapters этого релиза нет UR10e; Azure Kinect также требует
своего адаптера. Полная интеграция реализует `observation_features`,
`action_features`, `get_observation`, `send_action` и lifecycle. Для ViKi
проще сначала использовать policy-only adapter, сохранив владельца robot
control в нашем runtime.
[Hardware guide](https://huggingface.co/docs/lerobot/v0.6.1/integrate_hardware),
[built-in robots](https://github.com/huggingface/lerobot/tree/v0.6.1/src/lerobot/robots),
[async guide](https://huggingface.co/docs/lerobot/v0.6.1/async).

У LeRobot `DiffusionPolicy.select_action()` хранит историю наблюдений и очередь
действий; `predict_action_chunk()` выдаёт последовательность целей.
Новый inference может блокировать вызывающий поток, поэтому его место —
policy worker, не servo loop. Нормализация, RGB transforms, размеры истории,
action units и joint order определяются конкретным checkpoint.
[Pinned DP implementation](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/diffusion/modeling_diffusion.py).

| Метод исполнения chunks | Что даёт | Ограничение |
|---|---|---|
| [Async inference](https://huggingface.co/docs/lerobot/v0.6.1/async) | Следующий chunk вычисляется, пока выполняется текущий | Нужны time alignment и поведение при исчерпании очереди; async сам не согласует любую смену стратегии |
| [ACT temporal aggregation](https://github.com/tonyzhaozh/act) | Объединение предсказаний одного момента времени из нескольких chunks | Averaging нельзя считать безопасным для разных мод движения; нужен общий action convention |
| [RTC](https://arxiv.org/abs/2506.07339) | Новый chunk учитывает уже committed префикс предыдущего во время denoising | Поддержка алгоритма в статье шире конкретного готового backend |

В LeRobot v0.6.1 RTC engine явно проверяет `supports_rtc()` и сигнатуру с
`inference_delay`/`prev_chunk_left_over`. Обычный `DiffusionPolicy` принимает
`batch, noise` и не предоставляет этот RTC-контракт. Готовый RTC путь относится
к поддерживающим его политикам, например Pi0/Pi0.5/SmolVLA; это не универсальный
переключатель для каждого DP checkpoint.
[RTC compatibility gate](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/rollout/inference/rtc.py),
[RTC guide](https://huggingface.co/docs/lerobot/v0.6.1/rtc).

robomimic — альтернативный backend для DP/BC baselines: загрузка через
`policy_from_checkpoint`, wrapper `RolloutPolicy`, `start_episode()` и
`policy(obs)`. Он даёт модельный интерфейс, но не заменяет временное
согласование камер и hardware executor.
[Algorithm API](https://robomimic.github.io/docs/modules/algorithms.html),
[Diffusion Policy tutorial](https://robomimic.github.io/docs/tutorials/training_diffusion_policy.html).

Современный LeRobot нельзя просто добавить в существующий image: **релиз
v0.6.1 требует Python ≥3.12**, Torch ≥2.7,<2.12 и NumPy ≥2,<2.3; ViKi images
используют Python 3.10. Original Stanford DP real environment закрепляет ещё
другой стек: Python 3.9, Torch 1.12.1 и CUDA 11.6. Поэтому для будущей policy
рекомендуется отдельный GPU Docker service с закреплённым ML окружением и
локальным IPC; SDK/IK/RTDE остаются в своих окружениях. Загрузка этих
зависимостей не нужна для первого human-mimic режима.
[LeRobot v0.6.1 dependencies](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml),
[Stanford DP environment](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/conda_environment_real.yaml),
[ViKi image](../docker/Dockerfile), [UR image](../docker/Dockerfile.ur).

Для DP/RL следует переиспользовать observation history, timed target queue,
ограничения, единственного владельца control и execution recorder. Human
retarget producer позже заменяется policy worker. Recorder должен сохранять
изображения во время исполнения робота, измеренные текущие states, реально
отправленные commands, timestamps и преобразования координат; RL дополнительно
потребует reward, завершение, outcome и intervention metadata. Текущий
[LeRobot exporter](../viki/export/README.md) использует human video и следующий
attained state как action, не ресэмплирует replay по fps и исключает arm-only
логи. Эти данные нельзя выдавать за уже согласованный robot policy dataset.
Для arm-only исследований нужен отдельный честный schema без фиктивного
измеренного gripper и без обхода нынешнего screening gate. Физический tool/hand
позже добавляется отдельным action contract и драйвером.
