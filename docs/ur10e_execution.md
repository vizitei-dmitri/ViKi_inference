# Первый запуск ViKi на UR10e

Состояние на 2026-10-12: программный путь исполнения шести суставов добавлен,
аппаратная проверка ещё не проведена. На этом этапе физический захват отсутствует;
клещевой URDF остаётся частью офлайн-планирования. Драйвер ничего не командует
захвату и сохраняет его измерения как NaN. Замена на антропоморфную руку потребует
отдельного драйвера актуаторов и нового контракта наблюдений/действий.

## Что в проекте уже есть

ViKi — офлайн-конвейер с устойчивыми артефактами эпизода. Существующие записи
можно использовать без повторного capture: `rec.npz → cln.npz → plan.h5`.
Batch IK использует зарегистрированную модель робота, физический
адаптер и gripper TCP, joint/velocity limits, self-collision и плоскость пола.
Сцена SAM/object-model нужна для cube-aware режима; hand-only ретаргетинг
запускается отдельно через `--no-object-grasp`.

Критичные для реального исполнения особенности:

- По умолчанию выбран **UR10**, а нужен **UR10e**. Старый UR10-план надо заново
  решить для UR10e, даже если оба имеют шесть суставов.
- `q` — физические углы суставов в радианах; порядок RTDE:
  shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3.
  Pinocchio continuous-joint embedding и автоматический wrap углов здесь
  неприемлемы: переход через 2π может означать полный оборот реального сустава.
- План содержит `q_approach` из зарегистрированного home, а не текущего
  состояния робота. Replay проверяет первый кадр подхода и не достраивает
  непроверенный переход из произвольной позы.
- `RETARGET_ROBOT_BASE_POSITION/RPY` — положение базы относительно калибровки
  ChArUco, а не измеренная автоматически регистрация робота. Значение
  `[-0.7, 0, 0]` и адаптер 11 мм из пилота нельзя считать геометрией нового стенда.
- `raw/world_anchor.json` переносит rig в calibration frame один раз.
  Плановый gripper TCP — динамический центр контактных панелей. RTDE TCP —
  активный TCP установки UR; эти два определения нельзя смешивать.
- Collision с реальным окружением, сплошная проверка пути между кадрами и
  физическая успешность захвата пока не реализованы. Сохранённые геометрические
  margins — проверка модели в дискретных точках, а не ground truth.

## Выбранный план и предварительные проверки

Выбран эпизод `2026-10-06_11-15-00`. Отдельная копия проверенного плана:
`data/ur10e_first_run/episode/plan.h5`. Не запускай retarget поверх этой копии.
[Протокол сравнения десяти эпизодов](ur10e_first_run_selection.md) содержит
измерения и ограничения отбора. В MuJoCo используется голая UR10e, поскольку
этот тест исполняет только руку; виртуальный клещевой захват остаётся в IK.

При `--speed-scale 0.03` полный подход и демонстрация занимают **314.16 с**;
максимальная скорость команд **0.04115 rad/s (2.36°/с)**, дискретное ускорение
**0.2352 rad/s²**. Старт: **`[0, -90, 0, -90, 0, 0]` градусов** в порядке
base / shoulder / elbow / wrist 1 / wrist 2 / wrist 3.
SHA-256 плана:

```text
dafab3957cacf9bef895322f7137d026f2fe7e6d4d335700faa5c97c0ca3eebe
```

Все команды ниже выполняются из корня проекта:

```bash
cd /home/yoba/Documents/work/ViKi_inference
EP=data/ur10e_first_run/episode
```

## 1. Предпросмотр в MuJoCo без робота

Образы `viki-ur` и `viki-preview` уже собраны на этой машине; на новой машине
используй `docker compose build ur preview`. Для просмотра и исполнения
готового плана полный perception-образ с camera SDK не нужен.
Модель Menagerie уже скачана; revision и hashes файлов лежат в
`models/mujoco/ur10e/source.json`. Проверить её кэш или скачать при отсутствии:

```bash
docker compose run --rm --entrypoint python3 preview scripts/fetch_ur10e_mjcf.py
```

Открыть интерактивный просмотр:

```bash
xhost +SI:localuser:root
docker compose run --rm preview preview "$EP" \
  --model models/mujoco/ur10e/scene.xml \
  --mode position --speed-scale 0.03 --gui --playback-speed 10
```

Окно открывается **на паузе**. `Space` запускает/приостанавливает, мышь вращает
камеру и меняет масштаб; закрытие окна заканчивает просмотр. Полный путь
при просмотре 10× занимает около 32 с. `--playback-speed` меняет только темп
просмотра: динамика моделируется при `speed-scale 0.03`. Для реального темпа
поставь `--playback-speed 1`.

Готовое видео: `data/ur10e_first_run/preview.mp4`. Перегенерировать без окна:

```bash
docker compose run --rm -e MUJOCO_GL=egl preview preview "$EP" \
  --model models/mujoco/ur10e/scene.xml --mode position \
  --speed-scale 0.03 --playback-speed 10 \
  --video data/ur10e_first_run/preview.mp4
```

Просмотри **весь подход**, поворот запястья и финальную позу, сопоставь их со
свободным пространством вокруг реальной руки. МуJoCo проверяет модель руки
и плоскость пола; стол, стойка, кабели, инструмент и объекты туда не добавлены.
Положительный отчёт модели не подтверждает безопасность реального окружения.
`position` использует позиционные приводы; `kinematic` задаёт позы напрямую
и не проверяет динамику. Отчёты и traces: `$EP/intermediates/mujoco/`.

## 2. Ethernet и адрес компьютера

Подключи Ethernet компьютера к Ethernet-порту **контроллера UR**, включи
контроллер. На пульте проверь статический адрес **`192.168.0.100`** и маску
`255.255.255.0`. Для прямого кабеля gateway не нужен.

Проводной интерфейс этой машины — `enp49s0`. Назначь `192.168.0.10/24`,
если `.10` не занят. Создай отдельный профиль один раз:

```bash
sudo nmcli connection add type ethernet ifname enp49s0 \
  con-name ur10e-local ipv4.method manual \
  ipv4.addresses 192.168.0.10/24 ipv4.never-default yes \
  ipv6.method disabled
sudo nmcli connection up ur10e-local
ip -brief address show enp49s0
ip route get 192.168.0.100
ping -c 3 192.168.0.100
```

Если профиль уже существует, используй только `connection up`;
адрес можно исправить через `nmcli connection modify ur10e-local ...`.
Маршрут должен показать **`dev enp49s0 src 192.168.0.10`**.
До подключения адрес робота уходил через VPN `amn0`; проводной `/24` должен
получить приоритет. Wi-Fi и default gateway этим профилем не меняются.
Отключить профиль после тестов: `sudo nmcli connection down ur10e-local`.

## 3. Контроллер и чтение состояния

На пульте включи Remote Control в настройках и выбери Remote в меню режима.
Включи руку и отпусти тормоза штатными действиями на пульте. Останови текущую
программу и другие управляющие драйверы. Payload/CoG и TCP задаются для
**фактического** инструмента; масса виртуального Robotiq из IK не является
payload голой руки. ViKi эти настройки не меняет и protective stop не сбрасывает.

Диагностика только читает Dashboard/RTDE, без управляющего скрипта:

```bash
docker compose run --rm ur ur-status --host 192.168.0.100
```

Драйвер рассчитан на UR10e с PolyScope 5, Dashboard 29999 и RTDE 30004.
Dashboard может назвать UR10e `UR10`; проверяется также версия PolyScope.
Таймаут — повод проверить линк, маску и маршрут; ошибку Remote/тормозов
исправляй на пульте. Диагностика не включает питание и не отпускает тормоза.

## 4. Стартовая поза и короткий тест

Проверить выбранный план офлайн:

```bash
docker compose run --rm ur replay "$EP" --check \
  --arm-only --speed-scale 0.03
```

На пульте переведи руку в **`[0, -90, 0, -90, 0, 0]` градусов**, выбирая
свободный маршрут под наблюдением. ViKi не строит переход из текущей позы
и не выполняет автоматический moveJ в home. Сохраняется абсолютный winding:
не заменяй угол последнего сустава эквивалентным ±360°.
Если переключал Local для позиционирования, верни Remote и останови движение.
Проверить отклонение старта, без движения:

```bash
docker compose run --rm ur replay "$EP" --inspect \
  --host 192.168.0.100 --arm-only --speed-scale 0.03
```

Нужны `at_start: true`, неподвижная рука, normal/reduced safety mode и
остановленная программа. Допуск старта 0.01 rad, около 0.57° на сустав.
Перед исполнением проверка повторяется.

После просмотра пути и проверки свободного пространства следующая команда
**начинает физическое движение**, ограниченное десятью секундами подачи команд:

```bash
docker compose run --rm ur replay "$EP" \
  --driver ur10e --host 192.168.0.100 --arm-only \
  --speed-scale 0.03 --stop-after 10
```

Первые 10 с при таком замедлении — только начало подхода. Ожидается небольшое
движение и остановка. Нормальный результат: `test-stop`, причина
`test_duration_limit`, `completed=false`. Это не завершённый replay эпизода.
Проверяй command-rate `execution.h5`: summary демонстрации ещё не содержит
её кадров. `Ctrl+C` запрашивает программную остановку; при проблеме физического
движения используй штатную остановку UR. После неожиданной остановки разбери
причину по журналу, прежде чем повторять запуск.

## 5. Полный тест руки

После штатного короткого теста **снова верни руку на пульте в старт** и повтори
`--inspect`. Каждый запуск начинается с начала плана; продолжения с места
остановки нет. Если путь свободен и короткий тест прошёл штатно:

```bash
docker compose run --rm ur replay "$EP" \
  --driver ur10e --host 192.168.0.100 --arm-only --speed-scale 0.03
```

Прогон займёт около 5 мин 14 с и оставит руку в финальной позе; возврата нет.
При завершённом исполнении ожидается `arm-only`, без заявления об успешном
захвате. Оставь command bounds по умолчанию: 0.25 rad/s, 0.5 rad/s².

После read-only проверок подключается RTDE Control, проверяются архивные
waypoints относительно установленных controller safety limits и включается
watchdog 10 Hz. Команды servoJ подаются через initPeriod/waitPeriod. Есть
контроль safety mode, связи, программы, свежести RTDE, расписания и tracking
error. Tracking residual — max |q_actual − q_command| в rad, предел 0.05.
Опоздание свыше 40 мс вызывает остановку, без ускоренного догоняния.
Это программные проверки; они не заменяют системы безопасности UR.

При завершении, обработанной ошибке и Ctrl+C выполняются servoStop,
stopScript и disconnect, каждый шаг независимо. Автоматического reset/retry
нет. Hard kill/потеря питания не гарантируют запись журнала.
Файловый lock в общем `data/.robot-locks/` исключает два ViKi-процесса на одном
IP; сторонний драйвер и копию проекта с другим data он не блокирует.

HTTP использует те же функции: `POST /api/replay/check`, `/api/replay/inspect`,
`/api/replay`. Для исполнения:

```json
{
  "episode": "data/ur10e_first_run/episode",
  "driver": "ur10e",
  "host": "192.168.0.100",
  "options": {"arm_only": true, "speed_scale": 0.03, "stop_after": 10}
}
```

`POST /api/replay/jobs/<job-id>/stop` отменяет ожидающую задачу либо передаёт
остановку исполняющему потоку. Он делает cleanup и сохраняет частичный журнал;
другой поток не вызывает RTDE Control. Shutdown сервера запрашивает остановку
активных задач. CLI устанавливает обработчики SIGINT/SIGTERM и тогда, когда
shell передал игнорирование SIGINT.

## Артефакты и последующая политика

Каждая попытка создаёт собственный
`intermediates/replay/<run-id>/execution.h5`. Старые попытки сохраняются.
Журнал содержит SHA-256 исходного плана, identity робота, параметры запуска,
полный подход, commands и измерения с частотой управления. `q_observed`
считывается перед command; `q_attained`, `qd_attained` и активный TCP — после
периода. Есть host monotonic command time и controller timestamp; getters
читают SDK receive cache, а не общий атомарный пакет камер и proprioception.

`replay.h5` — краткая сводка только кадров демонстрации: первый control sample
на/после времени исходного кадра. `source_dt` сохраняет оригинальный шаг, `dt`
учитывает замедление. Успешный запуск без инструмента получает `arm-only`,
`hardware=true`, `gripper_measured=false`, NaN вместо фиктивных показаний
захвата. Нормальная остановка `--stop-after` сохраняет `test-stop`, `completed=false`;
обработанная ошибка сохраняет частичный лог, verdict reject и
`completed=false`; старые replay/export status инвалидируются.

Для Diffusion Policy/BC нужны observation–command пары с согласованной
частотой, представлением действия, horizon и нормализацией. Для последующего
RL также нужны outcome/reward и reset. Следующие этапы:

1. Снять роботную RGB-D запись одновременно с исполнением и сопоставить её
   host timestamps с RTDE. Исходное `raw/` — запись человека; переносить её
   как реальные наблюдения исполнения робота нельзя.
2. Добавить драйвер установленной руки, измерения её актуаторов и модель/TCP
   в retarget. Нормализованное opening сейчас имеет одну размерность;
   полноценная рука потребует векторной команды и state.
3. Задать action как действительно отправленную команду, а не следующий
   достигнутый state; определить frame, единицы, частоту и задержки.
4. Доработать LeRobot video/resampling/object-relative ветку и отдельную
   оценку успеха. Сейчас этот экспорт частичный и исключает dryrun/arm-only.

Все 10 загруженных эпизодов обработаны; известен IP `192.168.0.100`.
Аппаратная успешность и точность не заявлены: результатов physical replay нет.

Протокол управления сверялся с первичными источниками:
[UR remote interfaces](https://www.universal-robots.com/articles/ur/interface-communication/remote-control-via-tcpip/),
[UR Dashboard commands](https://www.universal-robots.com/manuals/EN/HTML/SW5_23/Content/prod-dashboard/Dashboard_table.htm),
[ur-rtde servoJ example](https://sdurobotics.gitlab.io/ur_rtde/pages/examples/high_frequency_servoing/servoj_example.html),
[ur-rtde API](https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html),
[MuJoCo Python viewer](https://mujoco.readthedocs.io/en/stable/python.html),
[UR10e Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/e4049d0a3bfd58d2a3081614e6777d4007e3f86a/universal_robots_ur10e).
