"""
Flask Web App for Multi-Class AI Timetable Scheduling.
Uses timetable_scheduler.py as the backend engine.
"""

import contextlib
import io
from collections import defaultdict
from typing import Any

from flask import Flask, jsonify, render_template, request, make_response

from timetable_scheduler import (
    Class, Room, Subject, Teacher, TimetableScheduler, 
    TimeSlot, GeneticScheduler, TimetableAnalytics
)

app = Flask(__name__)

ALL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MAX_GENERATION_RETRIES = 5


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_code(value: Any) -> str:
    return clean_text(value).upper()


def parse_int(value: Any, default: int, minimum: int = 1, maximum: int = 10_000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_code_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = raw.split(",")
    else:
        values = []
    out: list[str] = []
    seen = set()
    for item in values:
        code = normalize_code(item)
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def parse_break_periods(raw: Any, periods_per_day: int) -> list[int]:
    if not isinstance(raw, list):
        return []
    out = set()
    for item in raw:
        if isinstance(item, int):
            p = item
        elif isinstance(item, dict):
            p = parse_int(item.get("after"), 0)
        else:
            try:
                p = int(item)
            except (TypeError, ValueError):
                continue
        if 1 <= p <= periods_per_day:
            out.add(p)
    return sorted(out)


def build_scheduler_from_payload(payload: dict[str, Any], scheduler_class=TimetableScheduler):
    institution_name = clean_text(payload.get("institution_name")) or "Institution"
    days_per_week = parse_int(payload.get("days_per_week"), default=5, minimum=1, maximum=7)
    periods_per_day = parse_int(payload.get("periods_per_day"), default=8, minimum=1, maximum=12)
    # BUG FIX 1: Recesses are purely visual display separators — they are NOT scheduling
    # break slots.  Period 4 is still a full teaching period even if there is a recess
    # shown after it.  Passing recess "after" values as scheduler breaks removed those
    # periods from all_slots, making the timetable impossible to generate with any load.
    break_periods: list[int] = []          # always empty — scheduler sees no breaks
    used_days = ALL_DAYS[:days_per_week]

    teachers_data = payload.get("teachers", [])
    rooms_data = payload.get("rooms", [])
    subjects_data = payload.get("subjects", [])
    classes_data = payload.get("classes", [])
    if not isinstance(teachers_data, list) or not teachers_data:
        raise ValueError("Add at least one teacher.")
    if not isinstance(rooms_data, list) or not rooms_data:
        raise ValueError("Add at least one room.")
    if not isinstance(subjects_data, list) or not subjects_data:
        raise ValueError("Add at least one subject.")
    if not isinstance(classes_data, list) or not classes_data:
        raise ValueError("Add at least one class.")

    scheduler = scheduler_class(days=used_days, periods_per_day=periods_per_day, breaks=break_periods)

    seen_room_codes = set()
    has_lab_room = False
    for idx, r in enumerate(rooms_data, start=1):
        name = clean_text(r.get("name")) or f"Room {idx}"
        code = normalize_code(r.get("code")) or f"R{idx:03d}"
        if code in seen_room_codes:
            raise ValueError(f"Duplicate room code: {code}")
        seen_room_codes.add(code)
        is_lab = parse_bool(r.get("is_lab"))
        has_lab_room = has_lab_room or is_lab
        scheduler.add_room(
            Room(
                name=name,
                code=code,
                is_lab=is_lab,
                capacity=parse_int(r.get("capacity"), default=0, minimum=0, maximum=5_000),
                department_code=normalize_code(r.get("department_code"))
            )
        )

    flexible_subject_map: dict[str, dict[str, Any]] = {}

    # BUG FIX 3: max_ppw must use the full periods_per_day — break_periods is always []
    max_ppw = max(1, len(used_days) * periods_per_day)
    for idx, s in enumerate(subjects_data, start=1):
        name = clean_text(s.get("name")) or f"Subject {idx}"
        code = normalize_code(s.get("code")) or f"S{idx:03d}"
        if code in flexible_subject_map:
            raise ValueError(f"Duplicate subject code: {code}")

        template = {
            "name": name,
            "code": code,
            "periods_per_week": parse_int(s.get("periods_per_week"), default=4, minimum=1, maximum=max_ppw),
            "consecutive_periods": parse_int(s.get("consecutive_periods"), default=1, minimum=1, maximum=periods_per_day),
            "requires_lab": parse_bool(s.get("requires_lab")),
            "requires_double_period": parse_bool(s.get("requires_double_period")),
            "is_batch_specific": parse_bool(s.get("is_batch_specific")),
            "department_code": normalize_code(s.get("department_code"))
        }
        flexible_subject_map[code] = template
        flexible_subject_map[normalize_code(name)] = template

    subject_to_teachers: dict[str, list[str]] = defaultdict(list)
    seen_teacher_codes = set()
    for idx, t in enumerate(teachers_data, start=1):
        name = clean_text(t.get("name")) or f"Teacher {idx}"
        code = normalize_code(t.get("code")) or f"T{idx:03d}"
        raw_subjects = parse_code_list(t.get("subjects", []))

        if code in seen_teacher_codes:
            raise ValueError(f"Duplicate teacher code: {code}")
        seen_teacher_codes.add(code)

        resolved_subject_codes = []
        for s_ref in raw_subjects:
            target = flexible_subject_map.get(s_ref)
            if not target:
                raise ValueError(f"Teacher '{name}' references unknown subject: '{s_ref}'")
            resolved_subject_codes.append(target["code"])
            subject_to_teachers[target["code"]].append(code)

        scheduler.add_teacher(
            Teacher(
                name=name,
                code=code,
                subjects=resolved_subject_codes,
                max_periods_per_day=parse_int(t.get("max_periods_per_day"), default=8, minimum=1, maximum=12),
                max_consecutive_periods=parse_int(t.get("max_consecutive_periods"), default=3, minimum=1, maximum=12),
                department_code=normalize_code(t.get("department_code")),
                division_code=normalize_code(t.get("division_code"))
            )
        )

    class_catalog: list[dict[str, str]] = []
    seen_class_codes = set()
    for idx, c in enumerate(classes_data, start=1):
        class_name = clean_text(c.get("name")) or f"Class {idx}"
        class_code = normalize_code(c.get("code")) or f"C{idx:03d}"
        if class_code in seen_class_codes:
            raise ValueError(f"Duplicate class code: {class_code}")
        seen_class_codes.add(class_code)

        raw_class_subject_refs = parse_code_list(c.get("subjects", []))
        if not raw_class_subject_refs:
            raise ValueError(f"Class '{class_name}' has no subjects.")

        class_subjects: list[Subject] = []
        for s_ref in raw_class_subject_refs:
            template = flexible_subject_map.get(s_ref)
            if template is None:
                raise ValueError(
                    f"Class '{class_name}' references unknown subject: '{s_ref}'."
                )

            subject_code = template["code"]
            teachers_for_subject = subject_to_teachers.get(subject_code, [])
            if not teachers_for_subject:
                raise ValueError(
                    f"No teacher mapped to subject '{template['name']}' used by class '{class_name}'."
                )
            if template["requires_lab"] and not has_lab_room:
                raise ValueError(
                    f"Subject '{template['name']}' requires a lab room, but no lab room exists."
                )
            class_subjects.append(
                Subject(
                    name=template["name"],
                    code=template["code"],
                    periods_per_week=template["periods_per_week"],
                    consecutive_periods=template.get("consecutive_periods", 1),
                    requires_lab=template["requires_lab"],
                    requires_double_period=template["requires_double_period"],
                    is_batch_specific=template["is_batch_specific"],
                    department_code=template["department_code"],
                    teachers=teachers_for_subject,
                )
            )

        scheduler.add_class(
            Class(
                name=class_name,
                code=class_code,
                subjects=class_subjects,
                strength=parse_int(c.get("strength"), default=30, minimum=1, maximum=10_000),
                division_code=normalize_code(c.get("division_code")),
                department_code=normalize_code(c.get("department_code")),
                num_batches=parse_int(c.get("num_batches"), default=1, minimum=1, maximum=4)
            )
        )
        class_catalog.append({"name": class_name, "code": class_code})

    return scheduler, class_catalog, used_days, periods_per_day, break_periods, institution_name


def compute_expected_periods(scheduler) -> int:
    """
    BUG FIX #2: The original code always multiplied batch-specific subjects by 2,
    regardless of the class's actual num_batches. This caused expected_periods to be
    inflated for classes with num_batches=1 (the default), making the equality check
    always fail. Now we correctly multiply by cls.num_batches.
    """
    total = 0
    for cls in scheduler.classes.values():
        for subject in cls.subjects:
            multiplier = cls.num_batches if subject.is_batch_specific else 1
            total += subject.periods_per_week * multiplier
    return total


def run_capacity_audit(scheduler, periods_per_day: int) -> list[str]:
    """
    Comprehensive pre-flight audit that catches all common impossibility causes.
    Runs BEFORE the GA to give users actionable error messages.
    """
    issues = []
    total_slots = len(scheduler.days) * periods_per_day
    available_slots = len(scheduler.all_slots)  # Excluding breaks

    # ── 1. Class Overload ────────────────────────────────────────────────────
    for c_code, cls in scheduler.classes.items():
        class_load = 0
        for s in cls.subjects:
            mult = cls.num_batches if s.is_batch_specific else 1
            class_load += s.periods_per_week * mult
        if class_load > available_slots:
            issues.append(
                f"OVERLOAD: Class '{cls.name}' needs {class_load} periods "
                f"but only {available_slots} slots exist per week. "
                f"Reduce subject periods or add more days."
            )

    # ── 2. Teacher Overload ───────────────────────────────────────────────────
    teacher_demand: dict = {}
    for cls in scheduler.classes.values():
        for s in cls.subjects:
            demand_per_teacher = s.periods_per_week
            for t_code in s.teachers:
                teacher_demand[t_code] = teacher_demand.get(t_code, 0) + demand_per_teacher

    for t_code, demand in teacher_demand.items():
        t = scheduler.teachers.get(t_code)
        if not t: continue
        teacher_max = len(scheduler.days) * getattr(t, 'max_periods_per_day', periods_per_day)
        if demand > teacher_max:
            issues.append(
                f"OVERLOAD: Teacher '{t.name}' is assigned {demand} periods "
                f"but can only teach {teacher_max} per week. "
                f"Add another teacher for their subjects."
            )

    # ── 3. Room Type Shortage ─────────────────────────────────────────────────
    lab_rooms = [r for r in scheduler.rooms.values() if r.is_lab]
    normal_rooms = [r for r in scheduler.rooms.values() if not r.is_lab]

    total_lab_demand = 0
    total_normal_demand = 0
    for cls in scheduler.classes.values():
        for s in cls.subjects:
            mult = 2 if s.is_batch_specific else 1
            if s.requires_lab:
                total_lab_demand += s.periods_per_week * mult
            else:
                total_normal_demand += s.periods_per_week

    lab_capacity = len(lab_rooms) * available_slots
    normal_capacity = len(normal_rooms) * available_slots

    if total_lab_demand > 0 and len(lab_rooms) == 0:
        issues.append(
            "CRITICAL: Lab subjects exist but NO lab rooms are defined. "
            "Add a lab room to your rooms CSV."
        )
    elif total_lab_demand > lab_capacity:
        issues.append(
            f"SHORTAGE: Lab demand ({total_lab_demand} periods) exceeds "
            f"Lab room capacity ({lab_capacity} slots across {len(lab_rooms)} labs). "
            f"Add more lab rooms or reduce lab subjects."
        )

    if total_normal_demand > normal_capacity:
        issues.append(
            f"SHORTAGE: Classroom demand ({total_normal_demand} periods) exceeds "
            f"room capacity ({normal_capacity} slots across {len(normal_rooms)} rooms). "
            f"Add more classroom rooms."
        )

    # ── 4. Teacher-Subject Mismatch ───────────────────────────────────────────
    for cls in scheduler.classes.values():
        for s in cls.subjects:
            if not s.teachers:
                issues.append(
                    f"MISSING: Subject '{s.name}' in class '{cls.name}' "
                    f"has no teacher assigned. "
                    f"Add a teacher for this subject in your CSV."
                )

    return issues


def rebuild_slots_with_overrides(scheduler, day_overrides: dict, periods_per_day: int):
    """
    BUG FIX #4 & #5: Rebuilt slot list uses periods_per_day (the parsed, validated
    integer) as the fallback instead of scheduler.max_periods_per_day which may not
    be set. Also converts breaks to a set for O(1) membership testing.
    """
    scheduler.day_limits.update(day_overrides)
    break_set = set(scheduler.breaks)
    scheduler.all_slots = []
    for day in scheduler.days:
        limit = scheduler.day_limits.get(day, periods_per_day)
        for period in range(1, limit + 1):
            if period not in break_set:
                scheduler.all_slots.append(TimeSlot(day, period))


@app.route("/favicon.ico")
def favicon():
    """Suppress browser favicon 404 with a No Content response."""
    return make_response('', 204)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        payload = request.get_json(silent=True) or {}
        # We use 'recesses' from the payload; the 'break_periods' will be derived from it.

        algorithm = payload.get("algorithm", "backtracking")
        scheduler_class = GeneticScheduler if algorithm == "genetic" else TimetableScheduler

        # Build once to extract metadata and validate inputs up-front.
        # This raises ValueError early if the payload is malformed.
        _, class_catalog, used_days, periods_per_day, break_periods, institution_name = (
            build_scheduler_from_payload(payload, scheduler_class=scheduler_class)
        )

        day_overrides = payload.get("day_overrides", {})

        # ----------------------------------------------------------------
        # BUG FIX #1: Rebuild the scheduler fresh on each retry attempt.
        # The original code reused the same scheduler object, so any partial
        # schedule from a failed attempt contaminated subsequent retries,
        # making it virtually impossible to succeed after the first failure.
        # ----------------------------------------------------------------
        success = False
        scheduler = None

        # ── Pre-Flight Constraint Audit ──────────────────────────────────────
        # Build once for the audit check. If it detects impossibilities,
        # return the specific error immediately without running the GA.
        pre_scheduler, _, _, _, _, _ = build_scheduler_from_payload(
            payload, scheduler_class=scheduler_class
        )
        if day_overrides:
            rebuild_slots_with_overrides(pre_scheduler, day_overrides, periods_per_day)

        pre_audit_issues = run_capacity_audit(pre_scheduler, periods_per_day)
        if pre_audit_issues:
            return (
                jsonify({
                    "success": False,
                    "error": "Cannot generate timetable. Fix these issues in your CSV first: " +
                             " | ".join(pre_audit_issues)
                }),
                400,
            )

        for attempt in range(MAX_GENERATION_RETRIES):
            # Fresh scheduler for every attempt — no dirty state.
            scheduler, _, _, _, _, _ = build_scheduler_from_payload(
                payload, scheduler_class=scheduler_class
            )

            if day_overrides:
                rebuild_slots_with_overrides(scheduler, day_overrides, periods_per_day)

            # BUG FIX #2: Use the corrected expected_periods computation.
            expected_periods = compute_expected_periods(scheduler)

            # Capture internal error messages from the scheduler's stdout
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = scheduler.generate_timetable()
            
            stdout_output = f.getvalue().strip()
            scheduled_count = len(scheduler.schedule)

            if result and scheduled_count >= expected_periods:
                success = True
                break
            # BUG FIX 2: Only store stdout as an error hint when the attempt actually
            # failed AND the output looks like an error (not a success message).
            elif stdout_output and not stdout_output.startswith("GA Success"):
                last_error_msg = stdout_output

        if not success:
            # Capacity Audit
            audit_issues = run_capacity_audit(scheduler, periods_per_day)
            
            msg = "Could not generate a conflict-free timetable with current constraints."
            if audit_issues:
                msg = "Critical Resource Overload Detected: " + " | ".join(audit_issues)
            elif 'last_error_msg' in locals() and last_error_msg:
                # Clean up logs to only show the most relevant parts
                relevant_error = last_error_msg.split('\n')[-2:]
                msg += f" Details: {' '.join(relevant_error)}"
            else:
                msg += " Try adding more teachers/rooms or reducing subject load."

            return (
                jsonify(
                    {
                        "success": False,
                        "error": msg,
                    }
                ),
                400,
            )

        timetables = {
            class_item["code"]: scheduler.get_class_timetable(class_item["code"])
            for class_item in class_catalog
        }

        # Serialize raw entries for batch-aware rendering
        raw_entries = [
            {
                "day": e.time_slot.day,
                "period": e.time_slot.period,
                "class_code": e.class_code,
                "subject": e.subject.name,
                "subject_code": e.subject.code,
                "teacher": e.teacher.name,
                "room": e.room.code,
                "batch_index": e.batch_index
            }
            for e in scheduler.schedule
        ]

        # Run Scientific Analytics
        analytics = TimetableAnalytics(scheduler.schedule, used_days, periods_per_day)
        diagnostic_report = analytics.get_diagnostic_report()

        return jsonify(
            {
                "success": True,
                "days": used_days,
                "periods_per_day": periods_per_day,
                "break_periods": break_periods,
                "day_limits": scheduler.day_limits,
                "classes": class_catalog,
                "timetables": timetables,
                "raw_entries": raw_entries,
                "analytics": diagnostic_report,
                "statistics": {
                    "total_classes": len(scheduler.classes),
                    "total_teachers": len(scheduler.teachers),
                    "total_rooms": len(scheduler.rooms),
                    "total_periods_scheduled": len(scheduler.schedule),
                },
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)