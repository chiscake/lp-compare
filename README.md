# Симплекс, внутренняя точка и DLN на задачах распределения ресурсов

Воспроизводимый эксперимент для статьи на VII МНПК «Цифровая трансформация – шаг в будущее» (ЭФ БГУ, 2026).
Сравниваются три метода решения ЛП $\max c^\top x,\ Ax\le b,\ x\ge0$ в одной среде (C#, Native AOT, одни и те же
плотные структуры данных и векторные ядра):

- **табличный симплекс** с правилами Данцига и Бланда (одна реализация, различается только входящая переменная);
- **прямо-двойственный метод внутренней точки** Мехротры (нормальные уравнения + Холецкий);
- **градиентный спуск на диагональной линейной сети (DLN)** по Wang, Ghosal, Mazumder, [arXiv:2310.02535](https://arxiv.org/abs/2310.02535).

Эталонные оптимумы $f^\star$ получены HiGHS (двойственный симплекс и IPM с crossover; берётся решение с лучшим
прямо-двойственным сертификатом, $\le10^{-12}$) и хранятся в `experiments/manifest.csv`; в C#-прогонах внешний
решатель не вызывается.

| Документ | Содержание |
|---|---|
| [`docs/article.md`](docs/article.md) | текст статьи (аннотация, ключевые слова, методика, таблица, рисунок, выводы, литература) |
| [`docs/theory-and-plan.md`](docs/theory-and-plan.md) | разбор теории, сверка со статьёй про DLN, гипотезы, план |
| [`docs/decisions.md`](docs/decisions.md) | журнал решений: версии, параметры, правило подбора DLN, диагностика |
| [`results/cloud-xeon/`](results/cloud-xeon/) | результаты облачного прогона: сырые CSV, сводки, рисунки, отчёт о среде |

## Главные результаты (облачная ВМ, Intel Xeon с AVX-512, 1 поток)

Медианы по задачам класса (все три структуры): время, мс (итерации). Полные таблицы — [`summary.md`](results/cloud-xeon/summary.md),
проверка точности — [`validation.md`](results/cloud-xeon/validation.md).

| Класс | Данциг | Бланд | IPM | HiGHS ДС | DLN | Зазор DLN |
|---|---|---|---|---|---|---|
| S 10×20 | 0.0017 (7) | 0.0025 (14) | 0.022 (8) | 0.15 (6) | 12.7 (43 628) | 4.5e-6 |
| M 100×200 | 0.72 (123) | 7.7 (1442) | 4.3 (13) | 4.5 (86) | 2878 (лимит) | 0.13 |
| L 500×1000 | 403 (2102) | 12 188 (63 539) | 883 (19) | 170 (485) | 6653 (лимит) | 1.0 |
| XL 1000×2000 | 5433 (6800) | 157 569 (лимит в 67 %) | 9001 (22) | 900 (1152) | 7191 (лимит) | 0.98 |

- Оба правила симплекса совпадают с эталоном до $3\cdot10^{-13}$; IPM решает 100 % задач с зазором $\le1.3\cdot10^{-8}$.
- Правило Данцига быстрее нашего IPM почти везде (на плотных — в 19–37 раз, на разреженных/вырожденных XL — в 1.4–1.5 раза).
  Однопоточный HiGHS (двойственный симплекс, разреженная алгебра) на разреженных/вырожденных L и XL быстрее таблицы
  в 2.4–8.5 раза, на плотных L и XL — медленнее в 3.5–7 раз. Бланд делает в 7–32 раза больше пивотов начиная с M.
- DLN решает с качеством $10^{-3}$, $10^{-6}$, $10^{-8}$ (зазор при невязке $\le10^{-6}$) 70, 45 и 6 % задач S и ни одной
  задачи M–XL ([`quality.md`](results/cloud-xeon/quality.md)). Разложение по точному решению регуляризованной задачи
  ([`dln_decomposition.md`](results/cloud-xeon/dln_decomposition.md)): на L смещение регуляризации — 1.0 при ошибке
  оптимизации $10^{-4}$, на M они сопоставимы (0.075 и 0.076), на S преобладает ошибка оптимизации (κ, лимит).
  Нужные $\lambda$ лежат ниже окна double; DLN с $\ln u$ за $3\cdot10^6$ итераций на M остаётся на зазоре 0.3–0.8
  ([`lambda_long.md`](results/cloud-xeon/lambda_long.md)).

## Быстрый старт

Нужны **.NET SDK 10**, **uv** (менеджер Python) и, для Native AOT, системный C/C++-линкер. Python отдельно ставить не
нужно — `uv` скачает подходящий сам.

### Windows (PowerShell)

```powershell
winget install Microsoft.DotNet.SDK.10
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Native AOT: Visual Studio 2022 или Build Tools с нагрузкой «Разработка классических приложений на C++»
#   https://aka.ms/vs/17/release/vs_BuildTools.exe   (без неё используйте ключ -NoAot)

git clone https://origin.cursor.com/git/chis/lp-compare.git lp-methods; cd lp-methods
powershell -ExecutionPolicy Bypass -File scripts\run.ps1            # быстрый режим, ~10–20 мин
powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Mode full # полный эксперимент, несколько часов
```

### Linux / macOS (bash)

```bash
# .NET 10: https://dotnet.microsoft.com/download  (или: curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 10.0)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Native AOT: Ubuntu/Debian — sudo apt install clang zlib1g-dev;  macOS — xcode-select --install

git clone https://origin.cursor.com/git/chis/lp-compare.git lp-methods && cd lp-methods
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
5. Прогоны: иллюстративная задача, основной набор (`raw.csv`), траектории DLN и DLN с $\ln u$ (`trace.csv`, время до
   заданного зазора), однопоточный HiGHS как промышленный ориентир по времени (`highs.csv`), исследования $\lambda$
   (`lambda.csv`; в полном режиме также L — `lambda_large.csv` и длинные прогоны ниже окна double — `lambda_long.csv`) и $\kappa$.
6. Анализ: `summary.md`/`summary.csv` (min/медиана/p90/max и разброс повторов), `validation.md`, `quality.md` (доля задач,
   решённых с зазором $\le10^{-3},10^{-6},10^{-8}$ при допустимости, и время до этого), `dln_decomposition.md` (смещение
   регуляризации и ошибка оптимизации DLN по точному решению регуляризованной задачи на каждой задаче),
   `lambda_long.md`, `kappa.md`, `fig_time.*`, `fig_lambda.*`, `illustrative.md`.

| Режим | Задачи | Повторы | Время на современном ПК |
|---|---|---|---|
| `quick` | иллюстративная, S и M по 10 на структуру, $\lambda$-исследование по 5 задач | 1 + 3 | ~10–20 мин |
| `full` | все 570 задач основного набора (S/M/L/XL), траектории DLN, HiGHS, $\lambda$-исследование по 20 задач (L — по 5) и длинные log-прогоны | 1 + 5 (XL и DLN на M+ — 0/1 + 3, Бланд на XL — 1) | ~8–10 ч последовательно; в облаке 4 части на 4 ядрах ≈ 4 ч |

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
$B run  --set main --classes XL --out results/my/raw_xl.csv   # части можно гонять параллельно на разных ядрах; lpanalyze объединяет raw*.csv
$B grid --set main --classes S --structures dense --lambdas 0.01,0.005,0.002 --modes plain,log --out results/my/lambda.csv
$B grid --set main --classes M --structures dense --max-tasks 10 --lambdas 0.0005 --modes log --max-iter 3000000 --out results/my/lambda_long.csv
$B trace --set main --classes S,M --methods dln,dln-log --out results/my/trace.csv   # траектории DLN: время до зазора
uv run --project python lphighs --classes S,M --out results/my/highs.csv             # HiGHS, 1 поток, 3 повтора
uv run --project python lpanalyze --run results/my
```

`lpbench --help` перечисляет все опции. Параметры методов, лимиты и таймауты — в `experiments/params.ini`
(ключи вида `ключ.КЛАСС` и `ключ.метод.КЛАСС` переопределяют общие).

## Структура репозитория

```
src/LpBench.Core/       решатели: TableauSimplex, MehrotraIpm, DlnSolver; DenseMath (общие векторные ядра); метрики; формат LPB1
src/LpBench.Cli/        консольное приложение lpbench (env, solve, run, grid, trace), Native AOT
tests/LpBench.Tests/    xUnit: ручные задачи, иллюстративная задача, пример Била (зацикливание Данцига), статусы, I/O
python/lpbench_py/      generate.py (генератор + HiGHS), highs_bench.py (время HiGHS), analyze.py (таблицы, рисунки, точное
                        решение регуляризованной задачи)
experiments/            params.ini (замороженные параметры), manifest.csv (seed, SHA-256, f*)
scripts/                run.ps1 (Windows), run.sh (Linux/macOS)
results/<машина>/       результаты прогонов
data/                   сгенерированные задачи (≈ 0.85 ГБ, в git не хранятся)
```

## Ограничения

- Симплекс — учебная плотная таблица, IPM — плотные нормальные уравнения; это не промышленные решатели, а
  одинаково написанные реализации для сравнения методов в равных условиях. Как ориентир сильной реализации
  приведено время однопоточного HiGHS (модифицированный двойственный симплекс и IPM с разреженной алгеброй и presolve);
  оно измерено из Python (`highspy`, таймер вокруг `Highs.run()`), поэтому на S включает накладные расходы вызова.
- Все матрицы плотные, поэтому разреженность ускорения не даёт; разреженные задачи отличаются только численно.
- DLN использует редукцию из Remark 1.1 статьи (ограничение суммы и сдвиг стоимостей), так как статья требует
  $c>0$; детали и проверка условий — `docs/theory-and-plan.md`, п. 2.
- Облачные времена в `results/cloud-xeon/` справочные (виртуальная машина с общим железом).
