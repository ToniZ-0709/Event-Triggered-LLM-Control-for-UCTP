"""Parse one official ITC 2019 problem XML without loading its external DTD.

The order of ``classes`` is stable and defines the chromosome gene order. Every
file is an independent problem instance; a subset is a list of input paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import xml.etree.ElementTree as ET


_TYPES = {
    "SameStart", "SameTime", "DifferentTime", "SameDays", "DifferentDays",
    "SameWeeks", "DifferentWeeks", "SameRoom", "DifferentRoom", "Overlap",
    "NotOverlap", "SameAttendees", "Precedence", "WorkDay", "MinGap",
    "MaxDays", "MaxDayLoad", "MaxBreaks", "MaxBlock",
}
_PARAMETERS = {"WorkDay": 1, "MinGap": 1, "MaxDays": 1, "MaxDayLoad": 1,
               "MaxBreaks": 2, "MaxBlock": 2}
_TYPE_RE = re.compile(r"^([A-Za-z]+)(?:\((\d+(?:,\d+)*)\))?$")


@dataclass(frozen=True, slots=True)
class Time:
    days: str
    weeks: str
    start: int
    length: int
    penalty: int = 0
    day_mask: int = field(init=False, repr=False)
    week_mask: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "day_mask", int(self.days, 2))
        object.__setattr__(self, "week_mask", int(self.weeks, 2))

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True, slots=True)
class Room:
    id: str
    capacity: int | None
    unavailable: tuple[Time, ...]
    travel: dict[str, int]


@dataclass(frozen=True, slots=True)
class Class:
    id: str
    course_id: str
    config_id: str
    subpart_id: str
    parent_id: str | None
    limit: int | None
    times: tuple[Time, ...]
    room_options: tuple[tuple[str, int], ...]
    no_room: bool


@dataclass(frozen=True, slots=True)
class Config:
    id: str
    subparts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Course:
    id: str
    configs: tuple[Config, ...]


@dataclass(frozen=True, slots=True)
class Distribution:
    kind: str
    params: tuple[int, ...]
    required: bool
    penalty: int
    classes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Student:
    id: str
    courses: tuple[str, ...]


@dataclass(slots=True)
class Instance:
    name: str
    nr_days: int
    nr_weeks: int
    slots_per_day: int
    weights: tuple[int, int, int, int]
    rooms: dict[str, Room]
    classes: tuple[Class, ...]
    courses: dict[str, Course]
    distributions: tuple[Distribution, ...]
    students: tuple[Student, ...]
    class_index: dict[str, int]

    def travel(self, room_a: str | None, room_b: str | None) -> int:
        if not room_a or not room_b or room_a == room_b:
            return 0
        return self.rooms[room_a].travel.get(room_b, 0)


def _required(element: ET.Element, attribute: str) -> str:
    value = element.get(attribute)
    if value is None:
        raise ValueError(f"<{element.tag}> missing @{attribute}")
    return value


def _integer(element: ET.Element, attribute: str, default: int | None = None) -> int | None:
    raw = element.get(attribute)
    if raw is None:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"<{element.tag}> @{attribute} must be nonnegative")
    return value


def _time(element: ET.Element, days: int, weeks: int, slots: int) -> Time:
    day_bits = _required(element, "days")
    week_bits = _required(element, "weeks")
    if (len(day_bits) != days or len(week_bits) != weeks or
            set(day_bits) - {"0", "1"} or set(week_bits) - {"0", "1"} or
            "1" not in day_bits or "1" not in week_bits):
        raise ValueError(f"Invalid days/weeks bit strings in <{element.tag}>")
    start = _integer(element, "start")
    length = _integer(element, "length")
    if start is None or length is None or length < 1 or start + length > slots:
        raise ValueError(f"Invalid time interval in <{element.tag}>")
    return Time(day_bits, week_bits, start, length, _integer(element, "penalty", 0) or 0)


def load_instance(path: str | Path) -> Instance:
    path = Path(path)
    root = ET.parse(path).getroot()
    if root.tag != "problem":
        raise ValueError(f"{path}: expected <problem>, found <{root.tag}>")
    name = _required(root, "name")
    nr_days = int(_required(root, "nrDays"))
    nr_weeks = int(_required(root, "nrWeeks"))
    slots = int(_required(root, "slotsPerDay"))
    if min(nr_days, nr_weeks, slots) < 1:
        raise ValueError("nrDays, nrWeeks and slotsPerDay must be positive")
    opt = root.find("optimization")
    if opt is None:
        raise ValueError("Missing <optimization>")
    weights = tuple(int(_required(opt, key)) for key in ("time", "room", "distribution", "student"))
    if any(weight < 0 for weight in weights):
        raise ValueError("Optimization weights must be nonnegative")

    rooms_element = root.find("rooms")
    courses_element = root.find("courses")
    distributions_element = root.find("distributions")
    students_element = root.find("students")
    if any(x is None for x in (rooms_element, courses_element, distributions_element, students_element)):
        raise ValueError("Problem must contain rooms, courses, distributions and students")

    rooms: dict[str, Room] = {}
    for element in rooms_element.findall("room"):
        room_id = _required(element, "id")
        if room_id in rooms:
            raise ValueError(f"Duplicate room id {room_id}")
        travels = {_required(t, "room"): int(_required(t, "value")) for t in element.findall("travel")}
        rooms[room_id] = Room(room_id, _integer(element, "capacity"),
                              tuple(_time(u, nr_days, nr_weeks, slots) for u in element.findall("unavailable")),
                              travels)
    # ITC lists each symmetric travel edge once.
    for room in rooms.values():
        for other, value in tuple(room.travel.items()):
            if other not in rooms or value < 0:
                raise ValueError(f"Invalid travel edge {room.id} -> {other}")
            reverse = rooms[other].travel.get(room.id)
            if reverse is not None and reverse != value:
                raise ValueError(f"Asymmetric travel edge {room.id} <-> {other}")
            rooms[other].travel[room.id] = value

    classes: list[Class] = []
    courses: dict[str, Course] = {}
    class_index: dict[str, int] = {}
    for course_element in courses_element.findall("course"):
        course_id = _required(course_element, "id")
        if course_id in courses:
            raise ValueError(f"Duplicate course id {course_id}")
        configs: list[Config] = []
        for config_element in course_element.findall("config"):
            config_id = _required(config_element, "id")
            subparts: list[tuple[int, ...]] = []
            for subpart_element in config_element.findall("subpart"):
                subpart_id = _required(subpart_element, "id")
                indices: list[int] = []
                for element in subpart_element.findall("class"):
                    class_id = _required(element, "id")
                    if class_id in class_index:
                        raise ValueError(f"Duplicate class id {class_id}")
                    no_room = element.get("room", "true").lower() == "false"
                    room_options = tuple((_required(r, "id"), _integer(r, "penalty", 0) or 0)
                                         for r in element.findall("room"))
                    if (not no_room and not room_options) or (no_room and room_options):
                        raise ValueError(f"Invalid room domain for class {class_id}")
                    if any(room_id not in rooms for room_id, _ in room_options):
                        raise ValueError(f"Unknown room in class {class_id}")
                    times = tuple(_time(t, nr_days, nr_weeks, slots) for t in element.findall("time"))
                    if not times:
                        raise ValueError(f"Class {class_id} has no time options")
                    class_index[class_id] = len(classes)
                    indices.append(len(classes))
                    classes.append(Class(class_id, course_id, config_id, subpart_id,
                                         element.get("parent"), _integer(element, "limit"),
                                         times, room_options, no_room))
                if not indices:
                    raise ValueError(f"Empty subpart {subpart_id} of course {course_id}")
                subparts.append(tuple(indices))
            if not subparts:
                raise ValueError(f"Empty config {config_id} of course {course_id}")
            configs.append(Config(config_id, tuple(subparts)))
        if not configs:
            raise ValueError(f"Course {course_id} has no configs")
        courses[course_id] = Course(course_id, tuple(configs))

    for cls in classes:
        if cls.parent_id is not None:
            if cls.parent_id not in class_index:
                raise ValueError(f"Unknown parent {cls.parent_id} of class {cls.id}")
            if classes[class_index[cls.parent_id]].course_id != cls.course_id:
                raise ValueError(f"Parent of class {cls.id} belongs to another course")

    distributions: list[Distribution] = []
    for element in distributions_element.findall("distribution"):
        raw_type = _required(element, "type")
        match = _TYPE_RE.fullmatch(raw_type)
        if match is None or match.group(1) not in _TYPES:
            raise ValueError(f"Unsupported distribution type: {raw_type}")
        kind = match.group(1)
        params = tuple(map(int, match.group(2).split(","))) if match.group(2) else ()
        if len(params) != _PARAMETERS.get(kind, 0):
            raise ValueError(f"Incorrect parameters for distribution {raw_type}")
        ids = tuple(_required(c, "id") for c in element.findall("class"))
        if any(class_id not in class_index for class_id in ids):
            raise ValueError(f"Unknown class in distribution {raw_type}")
        required = element.get("required", "false").lower() == "true"
        distributions.append(Distribution(kind, params, required,
                                          _integer(element, "penalty", 0) or 0,
                                          tuple(class_index[class_id] for class_id in ids)))

    students: list[Student] = []
    student_ids: set[str] = set()
    for element in students_element.findall("student"):
        student_id = _required(element, "id")
        if student_id in student_ids:
            raise ValueError(f"Duplicate student id {student_id}")
        student_ids.add(student_id)
        requests = tuple(_required(c, "id") for c in element.findall("course"))
        if any(request not in courses for request in requests):
            raise ValueError(f"Unknown course requested by student {student_id}")
        students.append(Student(student_id, requests))

    return Instance(name, nr_days, nr_weeks, slots, weights, rooms, tuple(classes),
                    courses, tuple(distributions), tuple(students), class_index)
