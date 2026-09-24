#!/usr/bin/env python3
"""
doa.py — Визначення напрямку на ціль (Direction of Arrival).

═══════════════════════════════════════════════════════════════════
⚠️ ЧОМУ РАДАР ПОКАЗУВАВ НЕПРАВИЛЬНИЙ КУТ
═══════════════════════════════════════════════════════════════════

БАГ №1 (головний) — будь-яка помилка мовчки перетворювалась на 90°.
    Старий get_usb_doa() мав `except Exception: return 90`. Якщо
    утиліта не знайдена, впала, або вивід не розпарсився — радар
    отримував рівно 90° і вважав це правдою. Саме тому кут «зазвичай
    неправильний»: він фактично сталий.
    → Тепер при збої повертається None і радар пише «н/д», а причина
      помилки друкується один раз.

БАГ №2 — парсинг ламався на реальному форматі виводу.
    `ast.literal_eval("0.273 3.688 2.993 2.993")` кидає SyntaxError:
    literal_eval розуміє лише "[0.273, 3.688]" з дужками і комами.
    xvf_host зазвичай друкує значення ЧЕРЕЗ ПРОБІЛ → виняток →
    див. баг №1 → завжди 90°.
    → Тепер значення витягуються регулярним виразом і формат не має
      значення (пробіли, коми, дужки — будь-що).

БАГ №3 — брався випадковий промінь: `vals[-1]`.
    AEC_AZIMUTH_VALUES повертає азимут КОЖНОГО з чотирьох променів AEC.
    Останній у списку не є «обраним» — це просто четвертий промінь.
    → Тепер вибирається найбільша узгоджена група променів (у прикладі
      2.993 повторюється двічі — це і є домінантне джерело), або
      конкретний індекс, якщо його задано у radar_calibration.json.

БАГ №4 — субпроцес запускався у головному аудіоциклі.
    Один виклик xvf_host.py блокує потік надовго. Це блокувало читання
    аудіо → переповнення буфера → пропущені шматки звуку → гірша детекція.
    → Тепер опитування йде у власному потоці, аудіоцикл лише читає кеш.

    ⚠️ ЦИФРА ТУТ БУЛА ВЗЯТА ЗІ СТЕЛІ. Раніше цей коментар стверджував
    «200–500 мс», і на цю цифру потім спирались оцінки навантаження на
    Pi. ВИМІРЯНО на цільовому залізі (Raspberry Pi 5, Python у venv,
    `python diagnose.py xvf`, n=20):

        порожній запуск інтерпретатора   mean  18.3 мс   (p95  20.8)
        xvf_host.py AEC_AZIMUTH_VALUES   mean 128.6 мс   (p95 137.2)

    Тобто реальна вартість — 129 мс, а не 200–500, і майже вся вона
    належить самому xvf_host.py, а не старту Python (18 мс). При
    interval = 0.35 с це 27% зайнятості одного потоку і ~7% усього CPU
    чотириядерного Pi 5, ПОСТІЙНО — включно з часом, коли цілі немає.

    Це помітна, але НЕ головна стаття витрат: інференс Hailo займає 29 мс
    із 33.3 мс кадрового бюджету (87%), і саме він обмежує камеру.

БАГ №5 — не було жодної прив'язки до фізичної орієнтації масиву.
    Нуль градусів мікрофона майже ніколи не збігається з нулем на екрані.
    → Додано doa_offset_deg / doa_invert у radar_calibration.json
      (визначаються через `python calibrate.py doa`).

БАГ №6 — трекер намертво залипав на першому куті.
    SmartTracker запам'ятовував кут на початку захоплення і ігнорував
    усе, що відхилялось більше ніж на 50°. Якщо перший кут був хибним
    (див. баг №1 — а він був хибним завжди), ціль уже ніколи не могла
    «перетягнути» напрямок.
    → DOATracker тепер згладжує кут циркулярним середнім і перескакує
      на новий напрямок, якщо той підтверджується кілька разів поспіль.

Крім апаратного DOA реалізовано власний SRP-PHAT по сирих каналах —
він працює, якщо масив віддає 4 непроцесовані мікрофони.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from bearing_frame import CanonicalBearing, source_convention

SPEED_OF_SOUND = 343.0   # м/с при +20 °C


# ═══════════════════════════════════════════════════════════════
#  Результат вимірювання
# ═══════════════════════════════════════════════════════════════

@dataclass
class DOAReading:
    """
    Одне вимірювання напрямку.

    ⚠️ `angle_deg` після DOAProvider._canonicalise() уже КАНОНІЧНИЙ
    (bearing_frame). До нього — сирий кут джерела. Ці два стани розрізняються
    полем `calibrated`/`convention`, які заповнює саме канонізація.
    """
    angle_deg: float | None      # None = напрямок невідомий
    confidence: float = 0.0      # 0..1
    source: str = "none"         # "usb" | "srp" | "none"
    ambiguous: bool = False      # True = можлива дзеркальна неоднозначність
    raw: list[float] = field(default_factory=list)
    error: str | None = None
    #: Конвенцію джерела виміряно повністю (і нуль, і напрямок обертання).
    calibrated: bool = False
    #: Людський опис конвенції — показується, коли вона неповна.
    convention: str = ""
    #: ІДЕНТИЧНІСТЬ ФІЗИЧНОГО ВИМІРУ, а не момент читання.
    #:
    #: ⚠️ Кешоване значення читається БАГАТО РАЗІВ. HardwareDOA опитує DSP
    #: раз на ~0.35 с (плюс 200-500 мс на запуск субпроцесу), а аудіоцикл
    #: читає кеш кожні 0.25 с — тобто ОДИН вимір потрапляє у трекер 2-4
    #: рази, а поки кеш живий (max_age=2.0 с) — до 8 разів. Без цього поля
    #: трекер рахував ВИКЛИКИ, а не виміри: три «незалежні підтвердження»
    #: стрибка складались з одного-єдиного зчитування, і восьмимісна
    #: історія заповнювалась однією й тією ж цифрою, після чого зважене
    #: циркулярне середнє більше нічого не усереднювало.
    #:
    #: 0 = вимір НЕ МАЄ ідентичності і завжди свіжий. Саме такий SRP-PHAT:
    #: він рахується наново з кожного аудіоблоку, тому дедуплікувати його
    #: не можна і не треба.
    seq: int = 0

    @property
    def ok(self) -> bool:
        return self.angle_deg is not None


# ═══════════════════════════════════════════════════════════════
#  Розбір виводу xvf_host
# ═══════════════════════════════════════════════════════════════

_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_azimuth_values(text: str, key: str = "AEC_AZIMUTH_VALUES"
                         ) -> list[float]:
    """
    Витягує числа з виводу xvf_host.

    Приймає будь-який із форматів:
        AEC_AZIMUTH_VALUES: [0.273, 3.688, 2.993, 2.993]
        AEC_AZIMUTH_VALUES 0.273 3.688 2.993 2.993
        aec_azimuth_values = 0.273,3.688,2.993,2.993

    Старий код використовував ast.literal_eval і падав на всьому,
    крім першого варіанта.
    """
    for line in text.splitlines():
        if key.lower() not in line.lower():
            continue
        tail = line[line.lower().index(key.lower()) + len(key):]
        values = [float(m) for m in _FLOAT_RE.findall(tail)]
        if values:
            return values
    return []


def azimuths_to_degrees(values: list[float]) -> list[float]:
    """
    Приводить азимути до градусів 0–359.

    Прошивки віддають або радіани (0..2π), або одразу градуси.
    Розрізняємо за діапазоном: якщо всі |значення| ≤ 2π — це радіани.
    """
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    if np.max(np.abs(arr)) <= 2.0 * math.pi + 0.2:
        arr = np.degrees(arr)
    return [float(a % 360.0) for a in arr]


def select_azimuth(degrees: list[float], beam_index: int = -1,
                   cluster_deg: float = 25.0,
                   prefer_deg: float | None = None
                   ) -> tuple[float | None, float, bool]:
    """
    Обирає напрямок з чотирьох променів.

    Повертає (кут, впевненість, неоднозначно).

    beam_index >= 0 → беремо саме цей промінь.
    beam_index < 0  → шукаємо найбільшу узгоджену групу променів:
                      якщо кілька променів дивляться приблизно в один бік,
                      там і є домінантне джерело.

    ═══════════════════════════════════════════════════════════════
    ⚠️ ВИПРАВЛЕНО: НІЧИЯ МІЖ ГРУПАМИ ВИРІШУВАЛАСЬ ПОРЯДКОМ ПРОМЕНІВ
    ═══════════════════════════════════════════════════════════════

    Старий код брав першу групу, що досягла максимуму (`len > len`).
    Якщо DSP віддавав два промені на захід і два на схід, перемагала та,
    що трапилась раніше у списку — а порядок променів DSP не гарантує.
    Виміряно: ті самі чотири азимути у чотирьох порядках дають ДВІ
    ПРОТИЛЕЖНІ відповіді, і в усіх випадках впевненість 0.50, тобто
    нічия була невидимою для решти системи.

    Далі DOATracker бачив три «підтвердження» нового напрямку поспіль і
    чесно перескакував на протилежний бік — саме те, що на екрані
    виглядає як «ціль на заході, а стрілка іноді повертає на схід».

    Тепер нічия позначається прапорцем `ambiguous`, а кут береться як
    циркулярне середнє ВСІХ променів найбільшого розміру групи, тобто
    детермінований і не залежить від порядку. Трекер на неоднозначних
    вимірах не перескакує.
    """
    if not degrees:
        return None, 0.0, False

    if 0 <= beam_index < len(degrees):
        return degrees[beam_index], 0.5, False

    groups: list[list[float]] = []
    for centre in degrees:
        members = [d for d in degrees
                   if abs((d - centre + 180.0) % 360.0 - 180.0) <= cluster_deg]
        groups.append(members)

    best_size = max(len(g) for g in groups)
    winners = [g for g in groups if len(g) == best_size]

    # Групи навколо різних променів того самого джерела — це та сама група.
    # Різними вважаються лише ті, чиї центри рознесені більше за кластер.
    distinct: list[list[float]] = []
    for group in winners:
        centre = circular_mean(group)
        if not any(angular_diff(centre, circular_mean(d)) <= cluster_deg
                   for d in distinct):
            distinct.append(group)

    ambiguous = len(distinct) > 1

    # ═══════════════════════════════════════════════════════════
    # ⚠️ ПЛЮРАЛЬНІСТЬ БЕЗ БІЛЬШОСТІ — ТАКИЙ САМИЙ ФАНТОМ, ЯК НІЧИЯ
    # ═══════════════════════════════════════════════════════════
    #
    # Промінь, що перестав оновлюватись, тримає старе значення. Два
    # таких промені утворюють групу з двох, яка перемагає два поодинокі
    # промені на реальному джерелі — і робила це БЕЗ прапорця, бо
    # `ambiguous` спрацьовував лише на РІВНИХ групах.
    #
    # Виміряно на станції (XVF3800, calibrate.py check / diagnose beams):
    #     [39.4, 72.1, 4.7, 4.7]      -> 4.7   (50%), ambiguous=False
    #     [91.8, 245.1, 281.2, 245.1] -> 245.1 (50%), ambiguous=False
    # У другому випадку джерело було на ~272°, тобто помилка ~180°.
    #
    # Без прапорця такий вимір обходив УСІ запобіжники одразу:
    # `prefer_deg` читається лише в гілці нижче, а DOATracker гілкує свій
    # захист по `reading.ambiguous` — тож фантом міг САМОТУЖКИ захопити
    # ціль і слугувати підтвердженням стрибка.
    #
    # ⚠️ ДУБЛІКАТ САМ ПО СОБІ НЕ Є ДОКАЗОМ ПРОТУХЛОСТІ. DSP квантує
    # азимут із кроком ~0.3°, тому два ЖИВІ промені на одне джерело
    # регулярно дають біт-у-біт однакове число (виміряно: рядки
    # 269.1/269.1/269.1 під час справного стеження). Критерій тут —
    # НЕ «однакові значення», а «за напрямок голосує менше половини
    # променів»: такий вимір ненадійний незалежно від причини.
    if best_size * 2 <= len(degrees):
        ambiguous = True

    if ambiguous:
        # ═══════════════════════════════════════════════════════════
        # ⚠️ НІЧИЯ 2-ПРОТИ-2 СИСТЕМАТИЧНО ВИГРАВАЛАСЬ МЕНШИМ АЗИМУТОМ
        # ═══════════════════════════════════════════════════════════
        #
        # Ключ сортування був `(_spread_deg, circular_mean)`. При розкладі
        # 2-проти-2 обидві групи мають практично однаковий розкид (0..2°),
        # тому рішення щоразу падало на другий елемент ключа — тобто на
        # ВЕЛИЧИНУ КУТА. Виміряно виконанням: [300,302,60,62] → 61°,
        # [350,352,170,172] → 171°, [10,12,190,192] → 11°; у всіх шести
        # відтворених нічиїх перемагав кластер із меншим кутом, а там, де
        # кластери рознесені на ~180°, це РІВНО дзеркальна помилка.
        #
        # Жодного акустичного сенсу «менший азимут» не має: у
        # AEC_AZIMUTH_VALUES немає ані рівнів, ані ширини променя — самі
        # кути. Тобто ІНФОРМАЦІЇ, щоб обрати між двома рівними кластерами,
        # у цьому вимірі просто НЕМАЄ, і вигадувати її не можна.
        #
        # Тому вибір спирається на єдиний доказ, який існує поза цим
        # виміром: напрямок, у якому масив уже стабільно чув ціль
        # (`prefer_deg` — останній ОДНОЗНАЧНИЙ вимір, у тому самому сирому
        # кадрі). Нічия більше не тягне систему в бік менших кутів — вона
        # утримує вже підтверджений бік, а прапорець `ambiguous` і надалі
        # каже решті системи, що цей вимір ненадійний.
        #
        # Без `prefer_deg` (холодний старт) обґрунтованого вибору не існує
        # взагалі. Тоді лишається детермінований порядок — але сам по собі
        # він нічого не доводить, і саме тому DOATracker не дозволяє
        # неоднозначному виміру захопити ціль поодинці.
        if prefer_deg is None:
            distinct.sort(key=lambda g: (_spread_deg(g), circular_mean(g)))
        else:
            distinct.sort(key=lambda g: (angular_diff(circular_mean(g),
                                                      prefer_deg),
                                         _spread_deg(g), circular_mean(g)))
    best_members = distinct[0]

    angle = circular_mean(best_members)
    confidence = len(best_members) / len(degrees)
    if ambiguous:
        # Нічия — це НЕ така сама впевненість, як одностайність.
        confidence *= 0.5
    return angle, confidence, ambiguous


def _spread_deg(degrees: list[float]) -> float:
    """Максимальне відхилення групи кутів від її циркулярного середнього."""
    if len(degrees) < 2:
        return 0.0
    centre = circular_mean(degrees)
    return max(angular_diff(d, centre) for d in degrees)


# ═══════════════════════════════════════════════════════════════
#  Циркулярна арифметика
# ═══════════════════════════════════════════════════════════════

def circular_mean(degrees: list[float],
                  weights: list[float] | None = None) -> float:
    """Середнє кутів. Звичайне арифметичне тут неправильне: (350°+10°)/2 = 180°."""
    if not degrees:
        return 0.0
    w = np.ones(len(degrees)) if weights is None else np.asarray(weights, float)
    rad = np.radians(degrees)
    deg = np.degrees(np.arctan2(np.sum(w * np.sin(rad)),
                                np.sum(w * np.cos(rad))))
    # Округлення прибирає похибку float: без нього середнє(350°, 10°)
    # дає 359.99999999999994 замість рівно 0°.
    return float(np.round(deg, 6) % 360.0)


def angular_diff(a: float, b: float) -> float:
    """Найкоротша різниця між кутами, 0..180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def apply_orientation(angle: float, offset_deg: float,
                      invert: bool) -> float:
    """Переводить кут масиву у кут установки (екран / компас)."""
    if invert:
        angle = -angle
    return (angle + offset_deg) % 360.0


def unapply_orientation(angle: float, offset_deg: float,
                        invert: bool) -> float:
    """
    Зворотне до apply_orientation: кут установки → ВЛАСНИЙ кут масиву.

    Потрібно всюди, де кут треба віддати назад залізу, а не людині.
    Найважливіший приклад — світлодіодне кільце ReSpeaker: воно фізично
    закріплене на масиві, тому світитись має світлодіод у власній системі
    координат масиву. Якщо подати туди кут із екрана, кільце показуватиме
    хибний напрямок рівно на doa_offset_deg — і помилку буде видно лише
    на реальному залізі.

    Конвенція тримається тут, поруч із прямим перетворенням, щоб її не
    можна було випадково продублювати з іншим знаком.
    """
    angle = (angle - offset_deg) % 360.0
    if invert:
        angle = (-angle) % 360.0
    return angle


# ═══════════════════════════════════════════════════════════════
#  Апаратний DOA через xvf_host (XVF3800)
# ═══════════════════════════════════════════════════════════════

#: Каталоги, які ніколи не містять драйвера масиву, але можуть містити
#: сотні тисяч файлів. Пропускаються під час пошуку xvf_host.py.
_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", "AppData", "Library", "Windows",
    "Program Files", "Program Files (x86)", "OneDrive", "venv", ".venv",
    "site-packages", "dist-packages", "snap", "Downloads",
})

_XVF_CANDIDATES = [
    "/home/pi/mk/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py",
    "~/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py",
    "~/mk/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py",
    "/opt/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py",
]


def find_xvf_host(max_depth: int = 4,
                  max_seconds: float = 2.0) -> Path | None:
    """
    Шукає xvf_host.py. Шлях можна задати змінною середовища XVF_HOST_PATH.

    Старий код мав ОДИН зашитий шлях; якщо він не збігався, радар мовчки
    працював з фіктивним кутом 90°.

    Args:
        max_depth:   максимальна глибина пошуку в домашньому каталозі
        max_seconds: бюджет часу на пошук (див. коментар нижче)
    """
    env = os.environ.get("XVF_HOST_PATH")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser()

    for candidate in _XVF_CANDIDATES:
        p = Path(candidate).expanduser()
        if p.exists():
            return p

    # ── Останній шанс: пошук у домашніх каталогах ──
    #
    # ⚠️ ВИПРАВЛЕНО (інтеграція): тут було
    #     for p in home.glob("**/python_control/xvf_host.py")
    # тобто РЕКУРСИВНИЙ обхід УСЬОГО домашнього каталогу без обмежень.
    # Виміряно на реальній машині: понад 120 секунд (перервано, не
    # завершилось). Обхід зачіпає node_modules, .git, кеші, змонтовані
    # мережеві диски. Це блокувало ЗАПУСК радара — DOAProvider
    # створюється у setup-фазі аудіо, тобто мікрофон не починав слухати,
    # доки обхід не завершиться.
    #
    # Тепер: обмежена глибина (4 рівні) + бюджет часу. Типова установка
    # ~/reSpeaker_.../python_control/xvf_host.py знаходиться на 2-3 рівні,
    # тому реальні випадки як знаходились, так і знаходяться.
    # Пошук у ширину з перевіркою бюджету на КОЖНОМУ каталозі.
    # Path.glob() тут не годиться: навіть із обмеженою глибиною один
    # виклик glob сканує весь рівень, перш ніж повернути керування, тому
    # перевіряти час між викликами марно (виміряно: 23 с на глибині 4).
    deadline = time.monotonic() + max_seconds
    roots = [h for h in (Path.home(), Path("/home"))
             if h is not None and h.is_dir()]
    queue: list[tuple[Path, int]] = [(r, 0) for r in roots]

    while queue:
        if time.monotonic() > deadline:
            return None
        directory, depth = queue.pop(0)

        candidate = directory / "python_control" / "xvf_host.py"
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass

        if depth >= max_depth:
            continue
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if time.monotonic() > deadline:
                        return None
                    # Приховані каталоги й типові «важкі» кеші не містять
                    # драйвера масиву, але містять сотні тисяч файлів.
                    if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            queue.append((Path(entry.path), depth + 1))
                    except OSError:
                        continue
        except (OSError, PermissionError):
            continue
    return None


class HardwareDOA:
    """
    Фонове опитування азимута у DSP мікрофонного масиву.

    Аудіоцикл ніколи не чекає на субпроцес — він читає останнє
    закешоване значення.
    """

    def __init__(self, script_path: str | Path | None = None,
                 interval: float = 0.35, beam_index: int = -1,
                 timeout: float = 3.0):
        self.script_path = (Path(script_path).expanduser() if script_path
                            else find_xvf_host())
        self.interval = interval
        self.beam_index = beam_index
        self.timeout = timeout

        self._lock = threading.Lock()
        self._reading = DOAReading(None, error="ще не опитано")
        self._stamp = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._reported_error: str | None = None
        self.fail_count = 0
        #: Лічильник УСПІШНИХ вимірів. Зростає рівно раз на кожен азимут,
        #: реально отриманий від DSP, і проставляється у DOAReading.seq,
        #: щоб трекер міг відрізнити новий вимір від повторного читання
        #: того самого кешу. Див. DOAReading.seq.
        self._seq = 0
        #: Останній ОДНОЗНАЧНИЙ сирий азимут і його час — єдина підстава,
        #: за якою можна розв'язати нічию 2-проти-2 (див. select_azimuth).
        #: Кадр той самий, сирий, тому конвенція тут не потрібна.
        self._stable_raw: float | None = None
        self._stable_stamp = 0.0

    #: Скільки живе «стабільний бік» для розв'язання нічиї. Збігається з
    #: DOATracker.RELEASE_SEC: якщо трек уже відпущено, спиратись на нього
    #: більше немає підстав.
    STABLE_RAW_SEC = 6.0

    # ── Життєвий цикл ──────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self.script_path is not None and self.script_path.exists()

    def start(self) -> bool:
        if not self.available:
            with self._lock:
                self._reading = DOAReading(
                    None, error="xvf_host.py не знайдено "
                                "(задайте XVF_HOST_PATH)")
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            reading = self._query_once()
            with self._lock:
                self._reading = reading
                # ⚠️ BUG-004: monotonic, never wall clock. A Raspberry Pi has
                # no battery-backed RTC and steps its clock when NTP first
                # syncs after boot, often by hours. With time.time() a
                # forward step made every cached azimuth look older than
                # max_age, silently dropping the array to the SRP fallback
                # while it was answering normally; a backward step served
                # arbitrarily stale azimuths as fresh.
                self._stamp = time.monotonic()
            if reading.error and reading.error != self._reported_error:
                # Друкуємо кожну НОВУ помилку рівно один раз —
                # мовчазний провал і був причиною сталого кута 90°.
                print(f"\n⚠️  DOA (USB): {reading.error}")
                self._reported_error = reading.error
            self._stop.wait(self.interval)

    # ── Один запит ─────────────────────────────────────────────

    def _query_once(self) -> DOAReading:
        try:
            out = subprocess.run(
                [sys.executable, str(self.script_path), "AEC_AZIMUTH_VALUES"],
                capture_output=True, text=True, timeout=self.timeout,
                # ⚠️ xvf_host.py читає свої мапи команд відносно власної
                # директорії — без cwd він падає при запуску збоку.
                cwd=str(self.script_path.parent),
            )
        except subprocess.TimeoutExpired:
            self.fail_count += 1
            return DOAReading(None, error="таймаут запиту до масиву")
        except OSError as exc:
            self.fail_count += 1
            return DOAReading(None, error=f"не вдалось запустити: {exc}")

        text = (out.stdout or "") + "\n" + (out.stderr or "")
        values = parse_azimuth_values(text)

        if not values:
            self.fail_count += 1
            snippet = " ".join(text.split())[:120]
            return DOAReading(
                None, error=f"не вдалось розібрати вивід: «{snippet}»")

        degrees = azimuths_to_degrees(values)

        # Нічия розв'язується на користь боку, який масив уже стабільно
        # чув, а не на користь меншого кута. Підказка протухає разом із
        # треком, щоб не тягнути за собою давно неактуальний напрямок.
        now = time.monotonic()
        prefer = (self._stable_raw
                  if (self._stable_raw is not None
                      and now - self._stable_stamp <= self.STABLE_RAW_SEC)
                  else None)
        angle, conf, ambiguous = select_azimuth(degrees, self.beam_index,
                                                prefer_deg=prefer)
        if angle is not None and not ambiguous:
            # Тільки ОДНОЗНАЧНИЙ вимір має право ставати опорою для
            # наступної нічиї. Інакше одна нічия закріплювала б сама себе.
            self._stable_raw, self._stable_stamp = float(angle), now

        self.fail_count = 0
        # Новий фізичний вимір — новий номер. Кеш може бути прочитаний
        # скільки завгодно разів, але номер у нього залишиться цей.
        self._seq += 1
        return DOAReading(angle, confidence=conf, source="usb", raw=degrees,
                          ambiguous=ambiguous, seq=self._seq)

    # ── Читання кешу ───────────────────────────────────────────

    def read(self, max_age: float = 2.0) -> DOAReading:
        with self._lock:
            reading, stamp = self._reading, self._stamp
        if reading.ok and time.monotonic() - stamp > max_age:
            return DOAReading(None, error="дані застаріли")
        return reading


# ═══════════════════════════════════════════════════════════════
#  Власний SRP-PHAT по сирих каналах
# ═══════════════════════════════════════════════════════════════

class ArrayDOA:
    """
    SRP-PHAT: перебирає напрямки і шукає той, за якого затримки між
    мікрофонами найкраще узгоджуються з взаємною кореляцією.

    Працює, лише якщо звукова карта віддає СИРІ канали мікрофонів.
    Якщо масив уже все змікшував у оброблене стерео (типовий режим
    XVF3800 «2 канали»), канали будуть майже ідентичними — це виявляється
    і метод чесно повідомляє, що не може працювати.

    Точність з масивом 43 мм на 16 кГц — приблизно ±15°. Це грубо,
    але це РЕАЛЬНИЙ напрямок, а не константа.
    """

    def __init__(self, mic_positions: list[list[float]],
                 sample_rate: int,
                 band: tuple[float, float] = (200.0, 3500.0),
                 n_angles: int = 72, interp: int = 8):
        self.mics = np.asarray(mic_positions, dtype=np.float64)
        self.sample_rate = sample_rate
        self.band = band
        self.interp = interp
        self.n_mics = len(self.mics)

        self.angles = np.arange(0, 360, 360 // n_angles, dtype=np.float64)
        self.pairs = [(i, j) for i in range(self.n_mics)
                      for j in range(i + 1, self.n_mics)]

        # Максимальна можлива затримка (у семплах) для найдальшої пари
        max_dist = max(
            (np.linalg.norm(self.mics[i] - self.mics[j]) for i, j in self.pairs),
            default=0.0)
        self.max_lag = int(np.ceil(max_dist / SPEED_OF_SOUND * sample_rate)) + 2

        # Очікувані затримки для кожної пари та кожного напрямку.
        # tau_ij(θ) = -( (p_i - p_j) · u(θ) ) / c
        rad = np.radians(self.angles)
        unit = np.stack([np.cos(rad), np.sin(rad)], axis=1)     # [A, 2]
        self.expected_tau = np.stack(
            [-(self.mics[i] - self.mics[j]) @ unit.T / SPEED_OF_SOUND
             for i, j in self.pairs])                            # [P, A]

    # ── Допоміжне ──────────────────────────────────────────────

    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.sample_rate)
        spec[(freqs < self.band[0]) | (freqs > self.band[1])] = 0.0
        return np.fft.irfft(spec, n=len(x))

    def _gcc_phat(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Взаємна кореляція з фазовим вирівнюванням (PHAT), інтерпольована
        у `interp` разів. Повертає вектор довжини 2*max_shift+1,
        центрований на нульовій затримці.
        """
        n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
        spec = np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n))
        # PHAT: лишаємо тільки фазу — робить пік вужчим і стійким до
        # спектральної форми джерела (реверберація, вітер).
        spec /= np.abs(spec) + 1e-12
        cc = np.fft.irfft(spec, n * self.interp)

        shift = self.max_lag * self.interp
        return np.concatenate((cc[-shift:], cc[:shift + 1]))

    @staticmethod
    def _sample_at(cc: np.ndarray, centre: int, idx: np.ndarray) -> np.ndarray:
        """Лінійна інтерполяція кореляції у дробових позиціях."""
        pos = np.clip(centre + idx, 0, len(cc) - 2)
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        return cc[lo] * (1.0 - frac) + cc[lo + 1] * frac

    # ── Основний метод ─────────────────────────────────────────

    def usable_pairs(self, audio: np.ndarray,
                     threshold: float = 0.999) -> list[tuple[int, int]]:
        """
        Пари мікрофонів, які цього блоку несуть корисну інформацію.

        ═══════════════════════════════════════════════════════════════
        ⚠️ ВИПРАВЛЕНО: СЛІПА СМУГА ±90° І ДОВІЧНЕ ВИМКНЕННЯ SRP-PHAT
        ═══════════════════════════════════════════════════════════════

        Старий channels_are_distinct() порівнював ЛИШЕ канали 0 і 1.
        Мікрофони 0 і 1 у квадратному масиві мають однакову координату y,
        тож джерело на осі +y доходить до них ОДНОЧАСНО — канали стають
        ідентичними, і метод відмовлявся працювати, хоча мікрофони 2 і 3
        у цей момент дають бездоганний часовий зсув.

        Виміряно на синтетичній хвилі: напрямки 89°, 90°, 91° і 270°
        відхилялись повністю, тобто масив мав сліпу смугу шириною кілька
        градусів рівно перпендикулярно базі 0–1.

        Гірше: DOAProvider бачив у тексті помилки слово «ідентичні» і
        назавжди виставляв _array_disabled_reason. Тобто дрон, який один
        раз пройшов через 90°, вимикав SRP-PHAT до кінця сеансу.

        Тепер вироджені пари просто виключаються з суми — саме так це і
        має працювати: SRP-PHAT підсумовує кореляції ПАР, і пара з нульовою
        затримкою не несе інформації про напрямок, зате несе шум, який
        зміщує пік. Оброблене стерео (усі пари вироджені) як і раніше
        чесно відхиляється.
        """
        n_ch = min(audio.shape[1], self.n_mics)
        if n_ch < 2:
            return []
        centred = [audio[:, c] - audio[:, c].mean() for c in range(n_ch)]
        norms = [float(np.linalg.norm(c)) for c in centred]

        good: list[tuple[int, int]] = []
        for i, j in self.pairs:
            if i >= n_ch or j >= n_ch:
                continue
            denom = norms[i] * norms[j]
            if denom < 1e-9:
                continue
            if abs(float(centred[i] @ centred[j]) / denom) < threshold:
                good.append((i, j))
        return good

    def channels_are_distinct(self, audio: np.ndarray,
                              threshold: float = 0.999) -> bool:
        """
        Чи можна взагалі рахувати напрямок по цьому блоку.

        True, якщо хоча б одна пара мікрофонів не вироджена. Оброблене
        стерео від XVF3800 має майже ідентичні канали — по ньому DOA
        неможливий, і про це краще сказати, ніж видавати випадкові кути.
        """
        return bool(self.usable_pairs(audio, threshold))

    def estimate(self, audio: np.ndarray) -> DOAReading:
        """
        Args:
            audio: ndarray [samples, channels] — сирі канали мікрофонів
        """
        n_ch = min(audio.shape[1], self.n_mics)
        if n_ch < 2:
            return DOAReading(None, error="потрібно щонайменше 2 канали")

        good = set(self.usable_pairs(audio))
        if not good:
            return DOAReading(
                None, error="канали ідентичні — масив віддає оброблене "
                            "стерео, SRP-PHAT неможливий")

        channels = [self._bandpass(audio[:, c].astype(np.float64))
                    for c in range(n_ch)]

        scores = np.zeros(len(self.angles))
        used = 0
        for k, (i, j) in enumerate(self.pairs):
            if i >= n_ch or j >= n_ch:
                continue
            # Вироджена пара (джерело на її перпендикулярі) не несе
            # інформації про напрямок, зате додає у суму шум і зміщує пік.
            if (i, j) not in good:
                continue
            cc = self._gcc_phat(channels[i], channels[j])
            centre = (len(cc) - 1) // 2
            idx = self.expected_tau[k] * self.sample_rate * self.interp
            scores += self._sample_at(cc, centre, idx)
            used += 1

        if used == 0:
            return DOAReading(None, error="немає придатних пар мікрофонів")

        best = int(np.argmax(scores))
        peak, floor_ = float(scores[best]), float(np.median(scores))
        spread = float(scores.max() - scores.min())
        confidence = 0.0 if spread < 1e-9 else min(
            1.0, max(0.0, (peak - floor_) / spread))

        # З двома мікрофонами напрямок визначається з точністю до
        # дзеркального відображення відносно осі масиву.
        ambiguous = n_ch < 3

        return DOAReading(float(self.angles[best]), confidence=confidence,
                          source="srp", ambiguous=ambiguous)


# ═══════════════════════════════════════════════════════════════
#  Згладжування та утримання напрямку
# ═══════════════════════════════════════════════════════════════

class DOATracker:
    """
    Згладжує потік вимірювань кута.

    Замість старої логіки «запам'ятати перший кут і відкидати все,
    що далі 50°» (яка намертво залипала на хибному початковому куті)
    тут:
      • поточний кут = зважене циркулярне середнє останніх вимірювань;
      • одиничний викид ігнорується;
      • але якщо новий напрямок підтверджується JUMP_CONFIRMATIONS разів
        поспіль — трек переходить на нього (ціль реально перелетіла
        або початкове захоплення було хибним);
      • без свіжих даних упевненість згасає і трек звільняється.
    """

    HISTORY = 8
    OUTLIER_DEG = 35.0
    JUMP_CONFIRMATIONS = 3
    RELEASE_SEC = 6.0

    def __init__(self):
        self._history: deque[tuple[float, float]] = deque(maxlen=self.HISTORY)
        self._pending: list[float] = []
        self._last_update = 0.0
        self.angle: float | None = None
        self.confidence = 0.0
        self.ambiguous = False
        self.source = "none"
        #: Provenance of the readings currently in the history. The tracker
        #: now smooths CANONICAL angles only (see DOAProvider._canonicalise),
        #: so these describe the frame the smoothed angle is already in.
        self.calibrated = False
        self.convention = ""
        #: (seq, source) останнього ЗАРАХОВАНОГО фізичного виміру — щоб
        #: повторне читання того самого кешу не рахувалось ще раз.
        self._last_measurement: tuple[int, str] | None = None

    def reset(self) -> None:
        self._history.clear()
        self._pending.clear()
        self.angle = None
        self.confidence = 0.0
        # ⚠️ ПРАПОРЕЦЬ НЕОДНОЗНАЧНОСТІ ТЕЖ СКИДАЄТЬСЯ. Раніше він тут не
        # скидався, тому наступна — зовсім інша — ціль успадковувала
        # «неоднозначність» від попередньої, яку відпустили шість секунд
        # тому. Неоднозначність є властивістю ВИМІРУ, а не трекера, тому
        # вона не може пережити той трек, у якому виникла.
        self.ambiguous = False
        self.source = "none"
        self.calibrated = False
        self.convention = ""
        self._last_measurement = None

    def update(self, reading: DOAReading, now: float | None = None) -> None:
        # ⚠️ BUG-003: monotonic by default. This value is only ever used as
        # `now - self._last_update` against RELEASE_SEC, i.e. a DURATION, and
        # target_state.py states the project rule: wall clock can jump, and a
        # backward jump makes the difference negative so the release never
        # fires and a dead track is held for ever. Callers may still inject a
        # clock for deterministic tests.
        now = time.monotonic() if now is None else now

        if not reading.ok or reading.confidence <= 0.0:
            # Немає даних — упевненість повільно згасає
            if self.angle is not None and now - self._last_update > self.RELEASE_SEC:
                self.reset()
            return

        # ── ОДИН ФІЗИЧНИЙ ВИМІР — НЕ БІЛЬШЕ ОДНОГО ОНОВЛЕННЯ ──
        #
        # ⚠️ Аудіоцикл читає КЕШ HardwareDOA частіше, ніж DSP встигає його
        # оновити (0.25 с проти 0.35 с + 200-500 мс на субпроцес), тому те
        # саме зчитування поверталось 2-4 рази, а при пригальмованому
        # опитуванні — до 8 разів (max_age = 2.0 с). `_pending` рахував
        # ВИКЛИКИ, отже JUMP_CONFIRMATIONS = 3 «незалежні підтвердження»
        # стрибка могли надійти з ОДНОГО зчитування, і трек перекидався на
        # протилежний бік від одного хибного виміру. Історія страждала так
        # само: вісім слотів заповнювались однією цифрою.
        #
        # Тепер повторне читання того самого виміру не є ані підтвердженням,
        # ані новою точкою історії, ані приводом омолодити трек: воно просто
        # ігнорується, а трек продовжує старіти від СПРАВЖНЬОГО останнього
        # виміру. seq = 0 означає «ідентичності немає» (SRP-PHAT рахується
        # заново з кожного блоку) — такі виміри проходять завжди.
        if reading.seq:
            fingerprint = (reading.seq, reading.source)
            if fingerprint == self._last_measurement:
                return
            self._last_measurement = fingerprint

        angle = float(reading.angle_deg)
        self.source = reading.source
        self.calibrated = reading.calibrated
        self.convention = reading.convention

        if self.angle is None:
            # ── ЗАХОПЛЕННЯ ЦІЛІ ──
            #
            # ⚠️ НЕОДНОЗНАЧНИЙ ВИМІР НЕ ЗАХОПЛЮЄ ЦІЛЬ САМОТУЖКИ. Правило
            # «нічия не рухає трек» було реалізоване лише для викидів, і
            # порожній трек приймав будь-що. Через це ПЕРШИЙ же вимір після
            # тиші — навіть нічия 2-проти-2, у якій select_azimuth обрав
            # кластер лише за тим, що його кут менший, — задавав напрямок
            # треку. Далі неоднозначні виміри на тому самому (хибному) боці
            # проходили як «у межах ±35°» і підживлювали його, а такі ж
            # виміри на правильному боці відкидались як викиди. Трек
            # залипав на хибному боці.
            #
            # Тепер неоднозначне захоплення вимагає такої самої серії
            # узгоджених ПІДТВЕРДЖЕНЬ, як і стрибок. Однозначний вимір
            # захоплює ціль одразу, як і раніше. Це зберігає роботу на
            # 2-мікрофонному SRP-PHAT, де `ambiguous` істинний ЗАВЖДИ
            # (ambiguous = n_ch < 3): там трек просто з'явиться після трьох
            # узгоджених вимірів замість одного.
            if reading.ambiguous:
                centre = self._confirm(angle)
                if centre is None:
                    return
                self._accept(centre, reading.confidence, now, ambiguous=True)
                return
            self._pending.clear()
            self._accept(angle, reading.confidence, now, ambiguous=False)
            return

        if angular_diff(angle, self.angle) <= self.OUTLIER_DEG:
            self._pending.clear()
            self._accept(angle, reading.confidence, now,
                         ambiguous=reading.ambiguous)
            return

        # ── Викид: чекаємо підтвердження, перш ніж перескакувати ──
        #
        # ⚠️ НЕОДНОЗНАЧНИЙ ВИМІР НЕ Є ПІДТВЕРДЖЕННЯМ. Коли DSP віддає
        # нічию між двома протилежними групами променів, серія таких
        # вимірів виглядає точно як «ціль стабільно перелетіла» — і трекер
        # чесно перескакував на протилежний бік за 1.5 с. Це і був
        # механізм, через який на екрані ціль із заходу «іноді» опинялась
        # на сході. Нічия не рухає трек: вона лише не оновлює його.
        if reading.ambiguous:
            return
        centre = self._confirm(angle)
        if centre is not None:
            # Новий напрямок стабільний — переходимо на нього
            self._history.clear()
            self._accept(centre, reading.confidence, now, ambiguous=False)

    def _confirm(self, angle: float) -> float | None:
        """
        Накопичує виміри, що не збігаються з поточним треком.

        Повертає центр серії, щойно надійшло JUMP_CONFIRMATIONS УЗГОДЖЕНИХ
        між собою вимірів, інакше None. Кожен виклик — це окремий фізичний
        вимір: дублікати кешу відсіяні вище за `seq`.
        """
        self._pending.append(angle)
        if len(self._pending) < self.JUMP_CONFIRMATIONS:
            return None
        recent = self._pending[-self.JUMP_CONFIRMATIONS:]
        centre = circular_mean(recent)
        self._pending.clear()
        if all(angular_diff(a, centre) <= self.OUTLIER_DEG for a in recent):
            return centre
        return None

    def _accept(self, angle: float, weight: float, now: float,
                ambiguous: bool = False) -> None:
        # ⚠️ Прапорець неоднозначності належить ПРИЙНЯТОМУ виміру. Раніше
        # він виставлявся на самому початку update(), ще до всіх перевірок,
        # тому відкинутий викид усе одно вмикав «±mirror», а відкинутий
        # ОДНОЗНАЧНИЙ викид — гасив його. Прапорець описував вимір, якого
        # трек навіть не бачив.
        self.ambiguous = bool(ambiguous)
        self._history.append((angle, max(weight, 0.05)))
        angles = [a for a, _ in self._history]
        weights = [w for _, w in self._history]
        self.angle = circular_mean(angles, weights)
        self.confidence = float(np.mean(weights))
        self._last_update = now

    def format(self) -> str:
        if self.angle is None:
            return "н/д"
        text = f"{self.angle:5.1f}°"
        if self.ambiguous:
            text += "±"          # дзеркальна неоднозначність (2 мікрофони)
        return text


# ═══════════════════════════════════════════════════════════════
#  Об'єднаний провайдер
# ═══════════════════════════════════════════════════════════════

class DOAProvider:
    """
    Обирає найкраще доступне джерело напрямку:
      1. апаратний азимут XVF3800 (найточніший — там 4 сирі мікрофони);
      2. власний SRP-PHAT по сирих каналах;
      3. якщо нічого не працює — чесне «н/д».
    """

    #: Скільки блоків поспіль канали мають бути виродженими, перш ніж
    #: SRP-PHAT вимикається на весь сеанс. Один блок — це не доказ:
    #: ціль могла просто пройти через перпендикуляр до бази мікрофонів.
    ARRAY_DISABLE_AFTER = 20

    def __init__(self, cfg: dict, sample_rate: int, n_channels: int):
        # ⚠️ КОЖНЕ ДЖЕРЕЛО МАЄ ВЛАСНУ КОНВЕНЦІЮ. Раніше один
        # doa_offset_deg/doa_invert застосовувався і до USB, і до
        # SRP-PHAT. Але напрямок обертання SRP-PHAT ВІДОМИЙ з коду
        # (проти годинникової, це atan2 по mic_positions_m), а в USB DSP
        # він невідомий. Спільне калібрування гарантує, що правильним
        # може бути щонайбільше одне з двох джерел. Див. bearing_frame.
        self.conventions = {
            "usb": source_convention("usb", cfg),
            "srp": source_convention("srp", cfg),
        }

        self.hardware = HardwareDOA(beam_index=int(cfg.get("doa_beam_index", -1)))
        self.hardware_ok = self.hardware.start()

        self.array = ArrayDOA(cfg.get("mic_positions_m"), sample_rate)

        # ═══════════════════════════════════════════════════════
        # ⚠️ SRP-PHAT IS NOW GATED ON PHYSICAL VALIDITY (finding H5)
        # ═══════════════════════════════════════════════════════
        #
        # This used to be `self.array_ok = n_channels >= 2`, which armed
        # SRP-PHAT on any stereo device. On this station that is exactly
        # the wrong condition, because the XVF3800 delivers TWO PROCESSED
        # channels — its own beamformer output, not microphones.
        #
        # SRP-PHAT is a geometric method. It computes, for each candidate
        # direction, the inter-microphone delays that direction would
        # produce, and scores them against the measured cross-correlation.
        # Every term in that requires the channels to be microphones AT
        # KNOWN POSITIONS. Feed it two beamformed channels and the delays
        # it solves for do not correspond to any physical baseline: the
        # output is a smooth, plausible-looking angle with no relationship
        # to where the sound came from.
        #
        # Nothing downstream could catch this. The degeneracy check only
        # rejects channels that are nearly IDENTICAL, and the measured
        # correlation between the two processed channels was 0.8632 —
        # comfortably below the 0.999 threshold — so `usable_pairs()`
        # accepted the pair and the result was flagged merely `ambiguous`.
        #
        # Two independent conditions must BOTH hold, and BOTH require an
        # explicit attestation that only a hardware test can justify:
        #
        #   1. THE CHANNELS MUST BE VERIFIED RAW MICROPHONES —
        #      `mic_channels_verified: true` PLUS a non-empty
        #      `mic_channels`.
        #   2. THE GEOMETRY MUST BE VERIFIED —
        #      `mic_geometry_verified: true` PLUS `mic_positions_m`
        #      different from the unmeasured default.
        #
        # ⚠️ NEITHER A CHANNEL COUNT NOR A CHANNEL LIST IS EVIDENCE
        # (forensic review defect D5). An earlier version of this gate
        # accepted `n_channels >= 4 or mic_channels`. Both disjuncts were
        # wrong:
        #
        #   * a device can expose four PROCESSED channels — the XVF3800's
        #     own 6-channel mode is documented in audio_io as 4 mics plus
        #     2 references, which is itself an assumption, and a 4-channel
        #     mode could be 2 beams plus 2 references. Counting channels
        #     does not reveal their semantics;
        #   * `mic_channels: [0, 1]` is an operator typing a tuple into
        #     JSON. It is an assertion, not a measurement, and the whole
        #     point of finding H5 was that the two processed XVF3800
        #     channels are NOT microphones at known positions.
        #
        # Requiring a separate, explicit verified-flag means arming
        # SRP-PHAT is a deliberate act that records "a human ran the tap
        # test", rather than a side effect of filling in a config field.
        #
        # Refusing is safe. The USB DSP azimuth is the primary source and
        # is unaffected; losing the fallback means the station reports
        # "н/д" when the USB path fails, which is the honest answer, rather
        # than a number derived from a geometry it does not have.
        from calibration import DEFAULTS as _CAL_DEFAULTS

        channels_listed = bool(cfg.get("mic_channels"))
        channels_attested = bool(cfg.get("mic_channels_verified"))
        geometry_differs = (cfg.get("mic_positions_m")
                            != _CAL_DEFAULTS.get("mic_positions_m"))
        geometry_attested = bool(cfg.get("mic_geometry_verified"))

        raw_channels_verified = channels_listed and channels_attested
        geometry_verified = geometry_differs and geometry_attested

        self._array_disabled_reason: str | None = None
        if not raw_channels_verified:
            self._array_disabled_reason = (
                f"сирі мікрофонні канали НЕ ПІДТВЕРДЖЕНО "
                f"({n_channels} кан., mic_channels"
                f"{'' if channels_listed else ' не'} задано, "
                f"mic_channels_verified="
                f"{str(channels_attested).lower()}) — ні кількість каналів, "
                f"ні перелік каналів не доводять, що це фізичні мікрофони, "
                f"а не оброблені промені XVF3800; потрібен тест HW-5")
        elif not geometry_verified:
            self._array_disabled_reason = (
                f"геометрію масиву НЕ ПІДТВЕРДЖЕНО (mic_positions_m "
                f"{'відрізняється від типової' if geometry_differs else '— ЗНАЧЕННЯ ЗА ЗАМОВЧУВАННЯМ, квадрат 43 мм'}, "
                f"mic_geometry_verified={str(geometry_attested).lower()}) — "
                f"SRP-PHAT без виміряної бази дає стабільно неправильний "
                f"кут; потрібен тест HW-6")

        self.array_ok = self._array_disabled_reason is None
        self._degenerate_streak = 0

        self.tracker = DOATracker()
        #: Останній канонічний пеленг — те, що споживають радар, LED і
        #: камера. Жоден із них не бачить сирого кута джерела.
        self.canonical = CanonicalBearing()

    def describe(self) -> str:
        parts = []
        parts.append(f"USB DOA: {'✅ ' + str(self.hardware.script_path)}"
                     if self.hardware_ok else "USB DOA: ❌ недоступний")
        # ⚠️ The reason is printed, not just the state. "вимкнено" alone
        # invites an operator to go looking for a switch to turn back on;
        # the reason says what physical fact would have to change first.
        if self.array_ok:
            parts.append("SRP-PHAT: доступний")
        else:
            parts.append(f"SRP-PHAT: вимкнено — {self._array_disabled_reason}")
        for conv in self.conventions.values():
            parts.append(conv.describe())
        return " | ".join(parts)

    def update(self, audio: np.ndarray | None = None) -> DOAReading:
        """
        Args:
            audio: [samples, channels] — потрібно лише для SRP-PHAT
        """
        reading = self.hardware.read() if self.hardware_ok else DOAReading(None)

        if not reading.ok and self.array_ok and audio is not None \
                and self._array_disabled_reason is None:
            reading = self.array.estimate(audio)
            if not reading.ok and reading.error and "ідентичні" in reading.error:
                # ⚠️ Один вироджений блок — НЕ привід вимикати SRP-PHAT
                # назавжди. Раніше саме так і було: дрон, що один раз
                # пройшов перпендикулярно базі мікрофонів, гасив власний
                # DOA до кінця сеансу. Вимикаємо лише після стійкої серії,
                # яку дає справді оброблене стерео.
                self._degenerate_streak += 1
                if self._degenerate_streak >= self.ARRAY_DISABLE_AFTER:
                    self._array_disabled_reason = reading.error
                    print(f"\n⚠️  DOA (SRP-PHAT): {reading.error} "
                          f"({self._degenerate_streak} блоків поспіль)")
            elif reading.ok:
                self._degenerate_streak = 0

        # ⚠️ BUG-005. КАНОНІЗАЦІЯ ВІДБУВАЄТЬСЯ **ДО** ЗГЛАДЖУВАННЯ.
        #
        # Раніше трекер згладжував СИРІ кути, а конвенція застосовувалась до
        # результату — і бралась та, чиє джерело відповіло останнім. Оскільки
        # update() перемикається між USB і SRP-PHAT поблочно, у одному
        # циркулярному середньому опинялись кути у ДВОХ різних системах
        # координат, а потім до цієї суміші застосовувалась одна конвенція.
        # Виправити такий пеленг неможливо жодним калібруванням.
        #
        # Тепер кожен вимір переводиться у канонічний кадр СВОЄЮ конвенцією
        # одразу, і трекер згладжує вже однорідні числа. Побічний ефект:
        # перемикання джерела більше не рухає трек стрибком.
        canonical_reading = self._canonicalise(reading)
        self.tracker.update(canonical_reading)
        self.canonical = self._to_canonical(canonical_reading)
        return canonical_reading

    def _canonicalise(self, reading: DOAReading) -> DOAReading:
        """
        Один сирий вимір → той самий вимір у КАНОНІЧНОМУ кадрі.

        Єдине місце, де застосовується конвенція джерела.
        """
        if not reading.ok:
            return reading

        conv = self.conventions.get(reading.source)
        if conv is None:
            # ⚠️ BUG-009. Невідоме джерело раніше означало «викинути кут».
            # Кут при цьому був цілком дійсним — невідомою була лише його
            # конвенція. Тепер він проходить далі як НЕКАЛІБРОВАНИЙ: краще
            # показати напрямок і сказати, що його система координат
            # невідома, ніж мовчки не показати нічого.
            conv = source_convention(reading.source, {})
            self.conventions[reading.source] = conv
            print(f"\n⚠️  DOA: невідоме джерело «{reading.source}» — "
                  f"кут показується як НЕКАЛІБРОВАНИЙ")

        return replace(reading,
                       angle_deg=conv.to_canonical(reading.angle_deg),
                       calibrated=conv.calibrated,
                       convention=conv.describe())

    def _to_canonical(self, reading: DOAReading) -> CanonicalBearing:
        """
        Згладжений (уже канонічний) кут трекера → CanonicalBearing.

        Тут перетворень більше НЕМАЄ — тільки пакування. Радар, LED і
        камера далі виконують лише власні вихідні перетворення.
        """
        if self.tracker.angle is None:
            return CanonicalBearing(reason="напрямок не виміряно")

        return CanonicalBearing(
            deg=self.tracker.angle,
            confidence=self.tracker.confidence,
            source=self.tracker.source or reading.source or "none",
            calibrated=self.tracker.calibrated,
            ambiguous=self.tracker.ambiguous or reading.ambiguous,
            reason="" if self.tracker.calibrated else self.tracker.convention)

    def hold(self) -> CanonicalBearing:
        """
        Один блок БЕЗ вимірювання: трек лише старіє.

        ⚠️ BUG-019. Раніше radar.py робив це вручну, викликаючи приватний
        `_to_canonical` і сам присвоюючи `canonical` — тобто зовнішній
        модуль тримав у руках внутрішній інваріант цього класу. Тепер
        «блок без виміру» є частиною ПУБЛІЧНОГО контракту провайдера,
        поруч з update(), і обидва шляхи оновлюють стан однаково.
        """
        empty = DOAReading(None)
        self.tracker.update(empty)
        self.canonical = self._to_canonical(empty)
        return self.canonical

    def stop(self) -> None:
        self.hardware.stop()


# ═══════════════════════════════════════════════════════════════
#  Самотест
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("🧭 doa.py — самотест")
    print("=" * 66)

    print("\n1. Розбір виводу xvf_host (те, на чому падав старий код):")
    samples = [
        "AEC_AZIMUTH_VALUES: [0.273, 3.688, 2.993, 2.993]",
        "AEC_AZIMUTH_VALUES 0.273 3.688 2.993 2.993",
        "aec_azimuth_values = 0.273,3.688,2.993,2.993",
        "AEC_AZIMUTH_VALUES:  0.273  3.688  2.993  2.993  ",
    ]
    for s in samples:
        vals = parse_azimuth_values(s)
        deg = azimuths_to_degrees(vals)
        angle, conf, amb = select_azimuth(deg)
        print(f"   {s[:46]:<48s} → {angle:6.1f}°  (conf {conf:.2f}"
              f"{', НЕОДНОЗНАЧНО' if amb else ''})")

    import ast
    print("\n   Для порівняння — стара логіка ast.literal_eval:")
    for s in samples:
        try:
            vals = ast.literal_eval(s.split(":", 1)[1].strip())
            print(f"   {s[:46]:<48s} → {math.degrees(float(vals[-1])) % 360:.1f}°")
        except Exception as exc:
            print(f"   {s[:46]:<48s} → ❌ {type(exc).__name__} → "
                  f"старий код повертав 90°")

    print("\n2. Циркулярне середнє (звичайне середнє тут помиляється):")
    print(f"   середнє(350°, 10°) = {circular_mean([350.0, 10.0]):.1f}° "
          f"(арифметичне дало б 180°)")

    print("\n3. SRP-PHAT на синтетичній хвилі (квадрат 43 мм, 4 мікрофони):")
    sr = 16000
    mics = [[-0.0215, 0.0215], [0.0215, 0.0215],
            [0.0215, -0.0215], [-0.0215, -0.0215]]
    srp = ArrayDOA(mics, sr)
    rng = np.random.default_rng(0)

    for true_angle in (0.0, 45.0, 130.0, 250.0, 315.0):
        n = sr  # 1 секунда
        src = rng.standard_normal(n + 200)
        u = np.array([math.cos(math.radians(true_angle)),
                      math.sin(math.radians(true_angle))])
        channels = []
        for pos in mics:
            # Мікрофон, ближчий до джерела (p·u > 0), чує звук РАНІШЕ:
            # t_i = -p_i·u / c, тобто x_i[n] = s(n + p_i·u/c·sr).
            delay = np.dot(pos, u) / SPEED_OF_SOUND * sr
            idx = np.arange(n) + 100 + delay
            channels.append(np.interp(idx, np.arange(len(src)), src))
        audio = np.stack(channels, axis=1)
        audio += 0.01 * rng.standard_normal(audio.shape)

        r = srp.estimate(audio)
        err = angular_diff(r.angle_deg, true_angle) if r.ok else float("nan")
        print(f"   істина {true_angle:5.1f}° → оцінка {r.angle_deg:5.1f}°  "
              f"похибка {err:4.1f}°  conf={r.confidence:.2f}")

    print("\n4. Виявлення обробленого стерео (однакові канали):")
    mono = rng.standard_normal(16000)
    fake_stereo = np.stack([mono, mono], axis=1)
    r = ArrayDOA(mics[:2], sr).estimate(fake_stereo)
    print(f"   {r.error}")

    print("\n5. DOATracker — перескок на новий напрямок замість залипання:")
    tr = DOATracker()
    seq = [90.0] * 4 + [200.0] * 5
    for i, a in enumerate(seq):
        tr.update(DOAReading(a, confidence=0.8, source="test"), now=i * 0.5)
        print(f"   вимір {a:5.1f}° → трек {tr.format()}")
    print()
