# Симплекс, внутренняя точка и DLN на задачах распределения ресурсов

Воспроизводимый эксперимент для статьи на VII МНПК «Цифровая трансформация – шаг в будущее» (ЭФ БГУ, 2026).
Сравниваются три метода решения ЛП $\max c^\top x,\ Ax\le b,\ x\ge0$ в одной среде (C#, Native AOT, одни и те же
плотные структуры данных и векторные ядра):

- **табличный симплекс** с правилами Данцига и Бланда (одна реализация, различается только входящая переменная);
- **прямо-двойственный метод внутренней точки** Мехротры (нормальные уравнения + Холецкий);
- **градиентный спуск на диагональной линейной сети (DLN)** по Wang, Ghosal, Mazumder, [arXiv:2310.02535](https://arxiv.org/abs/2310.02535).

Эталонные оптимумы $f^\star$ получены HiGHS и хранятся в `experiments/manifest.csv`; в C#-прогонах внешний решатель
не вызывается.

| Документ | Содержание |
|---|---|
| [`docs/article.md`](docs/article.md) | текст статьи (аннотация, ключевые слова, методика, таблица, рисунок, выводы, литература) |
| [`docs/theory-and-plan.md`](docs/theory-and-plan.md) | разбор теории, сверка со статьёй про DLN, гипотезы, план |
| [`docs/decisions.md`](docs/decisions.md) | журнал решений: версии, параметры, правило подбора DLN, диагностика |
| [`results/cloud-xeon/`](results/cloud-xeon/) | результаты облачного прогона: сырые CSV, сводки, рисунки, отчёт о среде |

## Быстрый старт

Нужны **.NET SDK 10**, **uv** (менеджер Python) и, для Native AOT, системный C/C++-линкер. Python отдельно ставить не
нужно — `uv` скачает подходящий сам.

### Windows (PowerShell)

```powershell
winget install Microsoft.DotNet.SDK.10
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Native AOT: Visual Studio 2022 или Build Tools с нагрузкой «Разработка классических приложений на C++»
#   https://aka.ms/vs/17/release/vs_BuildTools.exe   (без неё используйте ключ -NoAot)

git clone <url-репозитория> lp-methods; cd lp-methods
powershell -ExecutionPolicy Bypass -File scripts\run.ps1            # быстрый режим, ~10–20 мин
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Mode full # полный эксперимент, несколько часов
```

### Linux / macOS (bash)

```bash
# .NET 10: https://dotnet.microsoft.com/download  (или: curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 10.0)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Native AOT: Ubuntu/Debian — sudo apt install clang zlib1g-dev;  macOS — xcode-select --install

git clone <url-репозитория> lp-methods && cd lp-methods
scripts/run.sh                 # быстрый режим
scripts/run.sh --mode full     # полный эксперимент
```

Результаты появятся в `results/<имя-машины>/` (имя меняется ключом `-Label` / `--label`).

### Что делает скрипт

1. Генерирует задачи по seed (`uv run --project python lpgen`) и **сверяет SHA-256 каждого файла с манифестом** —
   если хэш не совпал, вы решаете не те задачи, что в статье. Эталон $f^\star$ берётся из манифеста.
2. Запускает юнит-тесты (`dotnet test`).
3. Собирает Native AOT-бинарник под вашу ОС и процессор.
4. Пишет отчёт о среде `env.txt`: CPU, ядра, ОЗУ, ОС, версия рантайма, ширина `Vector<double>`, поддержка AVX2/AVX-512/NEON,
   замедление арифметики на субнормальных числах.
5. Прогоны: иллюстративная задача, основной набор (`raw.csv`), исследование $\lambda$ для DLN (`lambda.csv`).
6. Анализ: `summary.md`/`summary.csv` (min/медиана/p90/max по классам и методам), `validation.md`, `fig_time.*`,
   `fig_lambda.*`, `illustrative.md`.

| Режим | Задачи | Повторы | Время на современном ПК |
|---|---|---|---|
| `quick` | иллюстративная, S и M по 10 на структуру, $\lambda$-исследование по 5 задач | 1 + 3 | ~10–20 мин |
| `full` | все 570 задач основного набора (S/M/L/XL), $\lambda$-исследование по 20 задач | 1 + 5 (XL и DLN на M+ — 1) | ~3–5 ч |

## Запуск на других процессорах

Числа **зазор, невязка, итерации и статус от процессора не зависят**: все лимиты DLN заданы в итерациях, задачи
одинаковы по хэшу. Отличаться будут только время и, на уровне последних бит, результаты из-за разной ширины векторов.
Поэтому сравнивать машины имеет смысл по времени, а точность — проверять на совпадение.

Набор инструкций для Native AOT задаётся свойством `LpIsa` (передаётся в `IlcInstructionSet`):

| Ключ скрипта | Что получится | Когда использовать |
|---|---|---|
| `native` (по умолчанию) | инструкции процессора, **на котором идёт сборка** | замеры на своей машине; бинарник не переносить |
| `x86-64-v3` | AVX2/FMA/BMI, `Vector<double>` = 4 | переносимый бинарник для Intel/AMD с 2015 г. |
| `x86-64-v4` | + AVX-512 (ширина `Vector<T>` остаётся 256 бит) | серверные Intel, AMD Zen 4+ |
| `apple-m1` | ARMv8.5 + NEON, `Vector<double>` = 2 | Apple Silicon (`osx-arm64`) |
| `""` (пустая строка) | базовый уровень платформы, `Vector<double>` = 2 на x64 | максимальная переносимость |

```powershell
.\scripts\run.ps1 -Isa x86-64-v3          # Windows
```

```bash
scripts/run.sh --isa x86-64-v3            # Linux/macOS
scripts/run.sh --isa ""                   # без явного набора инструкций
```

Бинарник, собранный с `native` на машине с AVX-512, может не запуститься на процессоре без AVX-512 — собирайте на той
машине, где меряете. Ширина векторов попадает в `env.txt`; при сравнении машин проверяйте, что она одинакова, или
указывайте её в таблице.

Если AOT-тулчейна нет, `-NoAot` / `--no-aot` соберёт обычный JIT-бинарник (в `env.txt` будет `native_aot: False`).
JIT использует Dynamic PGO, поэтому такие времена с AOT-временами не смешивайте.

### Советы для стабильного времени

- Windows: схема питания «Высокая производительность», ноутбук от сети, закрыть браузеры/IDE/антивирусное сканирование.
- Linux: `taskset -c 2 artifacts/linux-x64/lpbench run ...` — фиксирует ядро; режим `performance` у cpufreq.
- Не запускайте два прогона одновременно.

## Команды по отдельности

```bash
uv run --project python lpgen --classes ILL,S,M        # только часть задач
uv run --project python lpgen --verify                 # только проверка хэшей
dotnet test -c Release
dotnet publish src/LpBench.Cli -c Release -r linux-x64 -p:LpIsa=native -o artifacts/linux-x64   # win-x64, osx-arm64, ...

B=artifacts/linux-x64/lpbench                           # на Windows: artifacts\win-x64\lpbench.exe
$B env --out results/my/env.txt
$B solve --file data/illustrative.lpb --fstar 150 --class ILL
$B run  --set main --classes S,M --max-tasks 10 --out results/my/raw.csv
$B grid --set main --classes S --structures dense --lambdas 0.01,0.005,0.002 --modes plain,log --out results/my/lambda.csv
uv run --project python lpanalyze --run results/my
```

`lpbench --help` перечисляет все опции. Параметры методов, лимиты и таймауты — в `experiments/params.ini`
(ключи вида `ключ.КЛАСС` и `ключ.метод.КЛАСС` переопределяют общие).

## Структура репозитория

```
src/LpBench.Core/       решатели: TableauSimplex, MehrotraIpm, DlnSolver; DenseMath (общие векторные ядра); метрики; формат LPB1
src/LpBench.Cli/        консольное приложение lpbench (env, solve, run, grid), Native AOT
tests/LpBench.Tests/    xUnit: ручные задачи, иллюстративная задача, пример Била (зацикливание Данцига), статусы, I/O
python/lpbench_py/      generate.py (генератор + HiGHS), analyze.py (таблицы и рисунки)
experiments/            params.ini (замороженные параметры), manifest.csv (seed, SHA-256, f*)
scripts/                run.ps1 (Windows), run.sh (Linux/macOS)
results/<машина>/       результаты прогонов
data/                   сгенерированные задачи (≈ 0.85 ГБ, в git не хранятся)
```

## Ограничения

- Симплекс — учебная плотная таблица, IPM — плотные нормальные уравнения; это не промышленные решатели, а
  одинаково написанные реализации для сравнения методов в равных условиях.
- Все матрицы плотные, поэтому разреженность ускорения не даёт; разреженные задачи отличаются только численно.
- DLN использует редукцию из Remark 1.1 статьи (ограничение суммы и сдвиг стоимостей), так как статья требует
  $c>0$; детали и проверка условий — `docs/theory-and-plan.md`, п. 2.
- Облачные времена в `results/cloud-xeon/` справочные (виртуальная машина с общим железом).
