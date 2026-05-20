"""
AI Academic Timetable Scheduler
==============================
A robust scheduling engine for schools and colleges using:
- Constraint Satisfaction Problem (CSP) solving
- Heuristic optimization
- Intelligent backtracking
"""

import random
from typing import List, Dict, Set, Tuple, Optional, Any, DefaultDict
from dataclasses import dataclass, field
from collections import defaultdict
import json
import numpy as np
import pandas as pd


@dataclass
class TimeSlot:
    """Represents a single time slot in the timetable"""
    day: str
    period: int
    
    def __hash__(self):
        return hash((self.day, self.period))
    
    def __eq__(self, other):
        return self.day == other.day and self.period == other.period
    
    def __repr__(self):
        return f"{self.day}-P{self.period}"


@dataclass
class Subject:
    """Subject with teaching requirements"""
    name: str
    code: str
    periods_per_week: int
    requires_lab: bool = False
    requires_double_period: bool = False
    consecutive_periods: int = 1
    is_batch_specific: bool = False # If true, class is split into B1/B2
    department_code: Optional[str] = None
    teachers: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.code)


@dataclass
class Teacher:
    """Teacher with availability and department"""
    name: str
    code: str
    subjects: List[str] = field(default_factory=list)
    unavailable_slots: Set[TimeSlot] = field(default_factory=set)
    department_code: Optional[str] = None
    division_code: Optional[str] = None
    max_periods_per_day: int = 8
    max_consecutive_periods: int = 2
    
    def __hash__(self):
        return hash(self.code)


@dataclass
class Room:
    """Physical classroom or laboratory with department tag"""
    name: str
    code: str
    is_lab: bool = False
    capacity: int = 0
    department_code: Optional[str] = None
    unavailable_slots: Set[TimeSlot] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.code)


@dataclass
class Class:
    """Student group with division and department"""
    name: str
    code: str
    subjects: List[Subject] = field(default_factory=list)
    strength: int = 0
    division_code: Optional[str] = None
    department_code: Optional[str] = None
    num_batches: int = 1 # 1 means whole class, 2 means split into B1/B2
    
    def __hash__(self):
        return hash(self.code)


@dataclass
class TimetableEntry:
    """A scheduled lesson with batch info"""
    class_code: str
    subject: Subject
    teacher: Teacher
    room: Room
    time_slot: TimeSlot
    batch_index: Optional[int] = None # None=Whole Class, 1=Batch1, 2=Batch2...


class ConstraintValidator:
    """Validates basic timetable constraints"""
    
    @staticmethod
    def validate_room_type(subject: Subject, room: Room) -> bool:
        """Check if room type matches subject requirements"""
        return subject.requires_lab == room.is_lab
    

class HeuristicSelector:
    """Smart heuristics for subject selection"""
    
    @staticmethod
    def order_subjects_by_difficulty(subjects: List[Subject]) -> List[Subject]:
        """Order subjects by difficulty (most constrained first - MRV Heuristic)"""
        def difficulty_score(subject):
            score = 0
            # Higher weight for more periods (harder to fit)
            score += subject.periods_per_week * 20
            # Labs and double periods are significant constraints
            score += 100 if subject.requires_lab else 0
            score += 50 if subject.requires_double_period else 0
            # Fewer available teachers means more constrained
            if subject.teachers:
                score += (10 / len(subject.teachers)) * 30
            else:
                score += 500 # Critical error case but ranked highest
            return score
        
        return sorted(subjects, key=difficulty_score, reverse=True)


class TimetableScheduler:
    """Main AI-powered timetable scheduling engine"""
    
    def __init__(self, days: List[str], periods_per_day: int, 
                 breaks: Optional[List[int]] = None,
                 day_configs: Optional[Dict[str, int]] = None):
        self.days = days
        self.max_periods_per_day = periods_per_day
        self.breaks = breaks or []
        # day_limits: mapping of day name to its specific number of periods
        self.day_limits = {day: periods_per_day for day in days}
        if day_configs:
            self.day_limits.update(day_configs)
        
        self.classes: Dict[str, Class] = {}
        self.teachers: Dict[str, Teacher] = {}
        self.rooms: Dict[str, Room] = {}
        self.subjects: Dict[str, Subject] = {}
        
        self.validator = ConstraintValidator()
        self.heuristic = HeuristicSelector()
        
        self.schedule: List[TimetableEntry] = []
        self.generation_attempts = 0
        self.max_attempts = 10000 
        
        # Performance optimizations: tracking busy slots with sets
        # Format: {(day, period, resource_code), ...}
        self.teacher_busy: Set[Tuple[str, int, str]] = set()
        self.room_busy: Set[Tuple[str, int, str]] = set()
        self.class_busy: Set[Tuple[str, int, str]] = set()
        
        # O(1) Lookups for Variety and Performance
        # (Day, Period, ClassCode) -> SubjectCode
        self.class_subject_at: Dict[Tuple[str, int, str], str] = {}
        # (Day, TeacherCode) -> List of busy periods (sorted)
        self.teacher_periods_daily: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        
        # Tracking counts/history for constraints
        self.teacher_daily_load: DefaultDict[Any, int] = defaultdict(int)
        self.class_daily_labs: DefaultDict[Any, int] = defaultdict(int) 
        self.class_subject_daily: DefaultDict[Any, int] = defaultdict(int) 
        self.class_daily_load: DefaultDict[Any, int] = defaultdict(int) 
        
        self.all_slots = []
        for day in self.days:
            limit = self.day_limits.get(day, self.max_periods_per_day)
            for period in range(1, limit + 1):
                if period not in self.breaks:
                    self.all_slots.append(TimeSlot(day, period))
        
    def add_class(self, class_obj: Class):
        """Add a class to the system"""
        self.classes[class_obj.code] = class_obj
    
    def add_teacher(self, teacher: Teacher):
        """Add a teacher to the system"""
        self.teachers[teacher.code] = teacher
    
    def add_room(self, room: Room):
        """Add a room to the system"""
        self.rooms[room.code] = room
    
    def get_all_slots(self) -> List[TimeSlot]:
        """Return cached available time slots"""
        return self.all_slots
    
    def find_compatible_teachers(self, subject: Subject) -> List[Teacher]:
        """Find teachers who can teach this subject"""
        return [t for t in self.teachers.values() if subject.code in t.subjects]
    
    def find_compatible_rooms(self, subject: Subject) -> List[Room]:
        """Find rooms suitable for this subject"""
        return [r for r in self.rooms.values() if self.validator.validate_room_type(subject, r)]
    
    def is_valid_assignment(self, class_code: str, subject: Subject, teacher: Teacher, room: Room, slot: TimeSlot) -> bool:
        """Check if an assignment satisfies all constraints using sets for O(1) lookups"""
        # Teacher availability (inherent constraints)
        if slot in teacher.unavailable_slots:
            return False
            
        # Room availability (inherent constraints)
        if slot in room.unavailable_slots:
            return False
            
        # Global collision checks (already scheduled items)
        if (slot.day, slot.period, teacher.code) in self.teacher_busy:
            return False
        if (slot.day, slot.period, room.code) in self.room_busy:
            return False
        if (slot.day, slot.period, class_code) in self.class_busy:
            return False
            
        # 1 Lab Per Day Constraint
        if subject.requires_lab:
            if self.class_daily_labs[(class_code, slot.day)] > 0:
                return False
                
        # Teacher Workload Constraints
        # Max periods per day
        if self.teacher_daily_load[(teacher.code, slot.day)] >= teacher.max_periods_per_day:
            return False
            
        # Consecutive periods limit
        if not self.validate_teacher_consecutive(teacher, slot):
            return False

        # Room Capacity Check
        cls_obj = self.classes[class_code]
        req_capacity = cls_obj.strength
        if subject.is_batch_specific and cls_obj.num_batches > 1:
            req_capacity = (req_capacity + cls_obj.num_batches - 1) // cls_obj.num_batches
            
        if room.capacity < req_capacity:
            return False
        
        return True

    def validate_teacher_consecutive(self, teacher: Teacher, slot: TimeSlot) -> bool:
        """Fast O(1)-ish check using pre-sorted busy periods"""
        day_t_key = (slot.day, teacher.code)
        busies = self.teacher_periods_daily[day_t_key]
        
        if not busies:
            return True
        
        # If we add this slot, would it create too many consecutive?
        temp_busies = sorted(busies + [slot.period])
        
        max_consecutive = 1
        current_consecutive = 1
        for i in range(len(temp_busies) - 1):
            if temp_busies[i+1] == temp_busies[i] + 1:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        max_consecutive = max(max_consecutive, current_consecutive)
        
        return max_consecutive <= teacher.max_consecutive_periods

    def calculate_slot_score(self, class_code: str, subject: Subject, teacher: Teacher, slot: TimeSlot) -> float:
        """Score a slot based on compactness and variety (higher is better)"""
        score: float = 0.0
        
        # 1. Compactness Heuristic: Is this slot adjacent to ANY existing period for this class?
        # We want to avoid "holes" in the student's day.
        prev_p = slot.period - 1
        next_p = slot.period + 1
        
        has_prev = (slot.day, prev_p, class_code) in self.class_busy if prev_p >= 1 else False
        has_next = (slot.day, next_p, class_code) in self.class_busy if next_p <= self.max_periods_per_day else False
        
        if has_prev or has_next:
            score += 200.0 # Strongly prefer slots that connect to existing ones
            
        # 2. Variety & Cognitive Load Heuristic (Pedagogical Balance)
        # Avoid clumping same subject or too many heavy subjects
        is_heavy = subject.periods_per_week >= 4
        
        # Variety: Avoid same subject in adjacent slots
        for neighbor_p in [slot.period - 1, slot.period + 1]:
            neighbor_key = (slot.day, neighbor_p, class_code)
            if neighbor_key in self.class_busy:
                # We need to find what subject that was. We can check self.schedule
                for entry in self.schedule:
                    if entry.class_code == class_code and entry.time_slot.day == slot.day and entry.time_slot.period == neighbor_p:
                        if entry.subject.code == subject.code:
                            if not (subject.requires_double_period or subject.requires_lab):
                                score -= 150.0 # Heavy penalty for repeated subject
                        
                        # Cognitive Load: Prefer Heavy-Light pattern
                        neighbor_is_heavy = entry.subject.periods_per_week >= 4
                        if is_heavy and neighbor_is_heavy:
                            score -= 30.0 # Minor penalty for back-to-back heavy subjects
                        break
            
        # 3. Daily Diversity: Favor spreading subjects across the week
        # Penalty for each time this subject is already on this day for this class
        day_count = self.class_subject_daily[(class_code, slot.day, subject.code)]
        if day_count > 0:
            score -= (day_count * 150.0) # High penalty to force spreading
            
        # 4. Load Balancing: Favor days with fewer periods to ensure even distribution
        # This prevents the AI from packing everything into Monday-Tuesday
        class_day_load = self.class_daily_load[(class_code, slot.day)]
        score -= (class_day_load * 20.0) # Each existing period on this day makes it less attractive
            
        # 5. Morning Priority (User Requirement: No gaps in P1/P2)
        if slot.period <= 2:
            score += 200.0 # Extreme preference for filling first two slots
        elif slot.period <= 4:
            score += 50.0 
        
        # Heavy subject priority in morning
        if slot.period <= 4 and subject.periods_per_week >= 4:
            score += 50.0
            
        # 6. Saturday/Weekend specifics
        if slot.day in ["Saturday", "Sunday"]:
            # Prefer labs on full days (Monday-Friday) if possible
            if subject.requires_lab:
                score -= 100.0
            # If it's a short day, discourage very late slots
            day_limit = self.day_limits.get(slot.day, self.max_periods_per_day)
            if slot.period >= day_limit - 1:
                score -= 50.0 # Discourage academic heavy-hitters at the very end of half-days
            
        # 4. Teacher Compactness
        t_has_prev = (slot.day, prev_p, teacher.code) in self.teacher_busy if prev_p >= 1 else False
        t_has_next = (slot.day, next_p, teacher.code) in self.teacher_busy if next_p <= self.max_periods_per_day else False
        
        if t_has_prev or t_has_next:
            score += 20.0 # Prefer slots that keep teacher's day grouped
            
        # 5. Tie-breaking
        score += random.random() * 5.0
        
        return score

    def _mark_busy(self, class_code: str, teacher_code: str, room_code: str, slot: TimeSlot, busy: bool, is_lab: bool, subject_code: str = ""):
        """Mark or UNMARK resources as busy in O(1)"""
        key_teacher = (slot.day, slot.period, teacher_code)
        key_room = (slot.day, slot.period, room_code)
        key_class = (slot.day, slot.period, class_code)
        
        if busy:
            self.teacher_busy.add(key_teacher)
            self.room_busy.add(key_room)
            self.class_busy.add(key_class)
            self.class_subject_at[(slot.day, slot.period, class_code)] = subject_code
            
            day_t_key = (slot.day, teacher_code)
            self.teacher_periods_daily[day_t_key].append(slot.period)
            self.teacher_periods_daily[day_t_key].sort()
            
            self.teacher_daily_load[(teacher_code, slot.day)] += 1
            self.class_subject_daily[(class_code, slot.day, subject_code)] += 1
            self.class_daily_load[(class_code, slot.day)] += 1
            if is_lab:
                self.class_daily_labs[(class_code, slot.day)] += 1
        else:
            self.teacher_busy.discard(key_teacher)
            self.room_busy.discard(key_room)
            self.class_busy.discard(key_class)
            self.class_subject_at.pop((slot.day, slot.period, class_code), None)
            
            day_t_key = (slot.day, teacher_code)
            if slot.period in self.teacher_periods_daily[day_t_key]:
                self.teacher_periods_daily[day_t_key].remove(slot.period)
            
            self.teacher_daily_load[(teacher_code, slot.day)] -= 1
            self.class_subject_daily[(class_code, slot.day, subject_code)] -= 1
            self.class_daily_load[(class_code, slot.day)] -= 1
            if is_lab:
                self.class_daily_labs[(class_code, slot.day)] -= 1

    def generate_timetable(self) -> bool:
        """Generate complete timetable for all classes using unit interleaving"""
        self.schedule = []
        self.generation_attempts = 0
        self.teacher_busy.clear()
        self.room_busy.clear()
        self.class_busy.clear()
        self.teacher_daily_load.clear()
        self.class_daily_labs.clear()
        
        ordered_classes = sorted(self.classes.values(), key=lambda x: x.code)
        
        # 0. Pre-flight Capacity Check
        total_days = len(self.days)
        for class_obj in ordered_classes:
            limit_total = sum(self.day_limits.get(d, self.max_periods_per_day) for d in self.days)
            needed = sum(s.periods_per_week for s in class_obj.subjects)
            if needed > limit_total:
                print(f"ERROR: Class {class_obj.name} requires {needed} periods but only {limit_total} available.")
                return False
                
            # Lab Day Check: Max 1 lab per day means Labs Per Week <= Total Days
            # Note: Lab units are usually handled in blocks, but we check number of subject objects requiring lab
            lab_subject_count = sum(1 for s in class_obj.subjects if s.requires_lab)
            if lab_subject_count > total_days:
                print(f"ERROR: Class {class_obj.name} has {lab_subject_count} labs which exceeds total days ({total_days}).")
                return False

        # Restart loop for tough constraints
        max_restarts = 5
        for restart in range(max_restarts):
            if restart > 0:
                print(f"Restarting search (Attempt {restart+1}/{max_restarts})...")
                self.schedule = []
                self.teacher_busy.clear()
                self.room_busy.clear()
                self.class_busy.clear()
                self.class_subject_at.clear()
                self.teacher_periods_daily.clear()
                self.teacher_daily_load.clear()
                self.class_daily_labs.clear()
                self.class_subject_daily.clear()
                self.class_daily_load.clear()
                self.generation_attempts = 0

            success = True
            for class_obj in ordered_classes:
                subject_units: Dict[str, List[Tuple[Subject, int]]] = defaultdict(list)
                for subject in class_obj.subjects:
                    total: int = subject.periods_per_week
                    while total > 0:
                        unit_size = subject.consecutive_periods
                        # Special handling for labs/double periods if not explicitly set
                        if unit_size <= 1 and (subject.requires_lab or subject.requires_double_period):
                            unit_size = 2
                        
                        unit_size = min(unit_size, total)
                        subject_units[subject.code].append((subject, unit_size))
                        total -= unit_size
                
                # Interleave
                sorted_subject_codes = sorted(
                    subject_units.keys(), 
                    key=lambda code: self.heuristic.order_subjects_by_difficulty([subject_units[code][0][0]])[0].periods_per_week, 
                    reverse=True
                )
                
                units: List[Tuple[Subject, int]] = []
                max_units_per_subject = max(len(u) for u in subject_units.values()) if subject_units else 0
                
                for i in range(max_units_per_subject):
                    codes = list(sorted_subject_codes)
                    if restart > 0: random.shuffle(codes) # More randomness on restarts
                    for code in codes:
                        if i < len(subject_units[code]):
                            units.append(subject_units[code][i])
                
                if not self._schedule_units_pointer(class_obj, units, 0):
                    success = False
                    break
            
            if success:
                return True
        
        return False

    def _schedule_units_pointer(self, class_obj: Class, units: List[Tuple[Subject, int]], index: int) -> bool:
        """Optimized pointer-based backtracking"""
        if index >= len(units):
            return True
        
        if self.generation_attempts >= self.max_attempts:
            return False
            
        subject, unit_size = units[index]
        
        # Try teachers, rooms, and slots
        # To avoid massive indentation, we keep the helper single unit logic
        if self._schedule_single_unit(class_obj, subject, unit_size):
            if self._schedule_units_pointer(class_obj, units, index + 1):
                return True
            
            # Backtrack
            for _ in range(unit_size):
                entry = self.schedule.pop()
                self._mark_busy(class_obj.code, entry.teacher.code, entry.room.code, entry.time_slot, False, subject.requires_lab, subject.code)
        
        return False

    def _schedule_single_unit(self, class_obj: Class, subject: Subject, unit_size: int) -> bool:
        """Schedules exactly ONE block of unit_size"""
        self.generation_attempts += 1
        
        teachers = self.find_compatible_teachers(subject)
        rooms = self.find_compatible_rooms(subject)
        slots = self.get_all_slots()
        
        # Priority: teachers with less daily load remaining, with tie-breaking randomness
        random_teachers = sorted(
            list(teachers), 
            key=lambda t: (sum(self.teacher_daily_load[(t.code, d)] for d in self.days), random.random())
        )
        
        for teacher in random_teachers:
            suitable_slots = [s for s in slots if s not in teacher.unavailable_slots]
            
            scored_slots = []
            for s in suitable_slots:
                score = self.calculate_slot_score(class_obj.code, subject, teacher, s)
                scored_slots.append((score, s))
            scored_slots.sort(key=lambda x: x[0], reverse=True)
            
            random_rooms = list(rooms)
            random.shuffle(random_rooms)
            
            for _, slot in scored_slots:
                assigned_slots = [slot]
                if unit_size == 2:
                    next_p = slot.period + 1
                    if next_p > self.max_periods_per_day: continue
                    next_slot = TimeSlot(slot.day, next_p)
                    if next_slot in slots and next_slot not in teacher.unavailable_slots:
                        assigned_slots.append(next_slot)
                    else:
                        continue
                
                for room in random_rooms:
                    all_valid = True
                    for s in assigned_slots:
                        if not self.is_valid_assignment(class_obj.code, subject, teacher, room, s):
                            all_valid = False
                            break
                    
                    if all_valid:
                        for s in assigned_slots:
                            self._mark_busy(class_obj.code, teacher.code, room.code, s, True, subject.requires_lab, subject.code)
                            self.schedule.append(TimetableEntry(class_obj.code, subject, teacher, room, s))
                        return True
        return False
    
    def get_class_timetable(self, class_code: str) -> Dict:
        """Get timetable for a specific class"""
        class_schedule = [e for e in self.schedule if e.class_code == class_code]
        timetable = {day: {} for day in self.days}
        for entry in class_schedule:
            timetable[entry.time_slot.day][entry.time_slot.period] = {
                'subject': entry.subject.name,
                'subject_code': entry.subject.code,
                'teacher': entry.teacher.name,
                'room': entry.room.name,
                'is_lab': entry.room.is_lab
            }
        return timetable

    def get_teacher_timetable(self, teacher_code: str) -> Dict:
        """Get timetable for a specific teacher"""
        teacher_schedule = [e for e in self.schedule if e.teacher.code == teacher_code]
        timetable = {day: {} for day in self.days}
        for entry in teacher_schedule:
            timetable[entry.time_slot.day][entry.time_slot.period] = {
                'class': entry.class_code,
                'subject': entry.subject.name,
                'room': entry.room.name
            }
        return timetable
    


@dataclass
class GeneticBlock:
    """A contiguous block of periods for a subject (supports batches)"""
    class_code: str
    subject: Subject
    teacher: Teacher
    room: Room
    start_slot_idx: int
    size: int
    batch_index: Optional[int] = None # Support for split batches
    
    # Cached indices for NumPy vectorization
    class_idx: int = -1
    teacher_idx: int = -1
    room_idx: int = -1
    subject_idx: int = -1
    
    def get_entries(self, all_slots: List[TimeSlot]) -> List[TimetableEntry]:
        entries = []
        for i in range(self.size):
            entries.append(TimetableEntry(
                self.class_code, self.subject, self.teacher, self.room, 
                all_slots[self.start_slot_idx + i], self.batch_index
            ))
        return entries


class GeneticScheduler:
    """
    Improved Block-Based Genetic Algorithm Scheduler
    =============================================
    - Preserves Double Periods and Labs by using "GeneticBlock" units.
    - Eliminates random gaps using a Compactness Fitness Bonus.
    """

    def __init__(self, days: List[str], periods_per_day: int,
                 breaks: Optional[List[int]] = None,
                 day_configs: Optional[Dict[str, int]] = None):
        self.days = days
        self.max_periods_per_day = periods_per_day
        self.breaks = breaks or []
        self.day_limits = {day: periods_per_day for day in days}
        if day_configs: self.day_limits.update(day_configs)

        self.classes: Dict[str, Class] = {}
        self.teachers: Dict[str, Teacher] = {}
        self.rooms: Dict[str, Room] = {}
        self.subjects: Dict[str, Subject] = {} # Added for easy subject lookup in fitness

        self.population_size = 100
        self.max_generations = 1000
        self.mutation_rate = 0.20
        self.schedule: List[TimetableEntry] = []
        self._last_penalties: DefaultDict[int, int] = defaultdict(int)
        self._last_spread_violations: int = 0
        
        self.all_slots = []
        for day in self.days:
            limit = self.day_limits.get(day, self.max_periods_per_day)
            for p in range(1, limit + 1):
                if p not in self.breaks:
                    self.all_slots.append(TimeSlot(day, p))
        
        # Pre-compute valid start indices for different block sizes
        self._valid_starts_cache = {}
        slots_by_day = defaultdict(list)
        for i, s in enumerate(self.all_slots):
            slots_by_day[s.day].append(i)
            
        # Cover block sizes 1-6 (handles single, double, lab, and extended blocks)
        for size in range(1, 7):
            day_starts = {}
            for day, idxs in slots_by_day.items():
                good = []
                for i in idxs:
                    if i + size - 1 < len(self.all_slots) and self.all_slots[i + size - 1].day == day:
                        consec = True
                        for j in range(size - 1):
                            if self.all_slots[i+j+1].period != self.all_slots[i+j].period + 1:
                                consec = False; break
                        if consec:
                            start_p = self.all_slots[i].period
                            end_p = start_p + size - 1
                            if not any(start_p <= br and end_p >= br + 1 for br in self.breaks):
                                good.append(i)
                day_starts[day] = good
            self._valid_starts_cache[size] = day_starts
        
        # Resource Mapping for NumPy Optimizations
        self.day_map = {day: i for i, day in enumerate(self.days)}
        self.t_id_map = {}
        self.r_id_map = {}
        self.c_id_map = {}
        self.s_id_map = {}

    def _sync_resource_maps(self):
        """Build integer ID maps for faster matrix operations."""
        self.t_id_map = {code: i for i, code in enumerate(sorted(self.teachers.keys()))}
        self.r_id_map = {code: i for i, code in enumerate(sorted(self.rooms.keys()))}
        self.c_id_map = {code: i for i, code in enumerate(sorted(self.classes.keys()))}
        self.s_id_map = {code: i for i, code in enumerate(sorted(self.subjects.keys()))}
        self._precompute_resource_data()

    def _precompute_resource_data(self):
        """Pre-cache resource properties in NumPy arrays for vectorized checks."""
        # Rooms
        num_rooms = len(self.rooms)
        self.r_cap_arr = np.zeros(num_rooms, dtype=np.int32)
        self.r_lab_arr = np.zeros(num_rooms, dtype=np.int8)
        for code, idx in self.r_id_map.items():
            r = self.rooms[code]
            self.r_cap_arr[idx] = r.capacity
            self.r_lab_arr[idx] = 1 if r.is_lab else 0
            
        # Subjects
        num_subjs = len(self.subjects)
        self.s_lab_arr = np.zeros(num_subjs, dtype=np.int8)
        self.s_double_arr = np.zeros(num_subjs, dtype=np.int8)
        for code, idx in self.s_id_map.items():
            s = self.subjects[code]
            self.s_lab_arr[idx] = 1 if s.requires_lab else 0
            self.s_double_arr[idx] = 1 if (s.requires_lab or s.requires_double_period) else 0

        # Classes
        num_classes = len(self.classes)
        self.c_strength_arr = np.zeros(num_classes, dtype=np.int32)
        self.c_batch_strength_arr = np.zeros(num_classes, dtype=np.int32)
        for code, idx in self.c_id_map.items():
            cls_obj = self.classes[code]
            self.c_strength_arr[idx] = cls_obj.strength
            self.c_batch_strength_arr[idx] = (cls_obj.strength + cls_obj.num_batches - 1) // max(1, cls_obj.num_batches)
            
        # Unavailability Masks: (Days x Periods x ResourceIdx)
        num_days = len(self.days) + 2 # Safety buffer
        num_periods = self.max_periods_per_day + 10 # Safety buffer
        self.t_unavail_mask = np.zeros((num_days, num_periods, len(self.teachers)), dtype=np.int8)
        self.r_unavail_mask = np.zeros((num_days, num_periods, len(self.rooms)), dtype=np.int8)
        
        for t_code, t_idx in self.t_id_map.items():
            for s in self.teachers[t_code].unavailable_slots:
                d_idx = self.day_map.get(s.day)
                if d_idx is not None: self.t_unavail_mask[d_idx, s.period, t_idx] = 1

        for r_code, r_idx in self.r_id_map.items():
            for s in self.rooms[r_code].unavailable_slots:
                d_idx = self.day_map.get(s.day)
                if d_idx is not None: self.r_unavail_mask[d_idx, s.period, r_idx] = 1

    def add_class(self, class_obj: Class):
        self.classes[class_obj.code] = class_obj
        for subj in class_obj.subjects:
            self.subjects[subj.code] = subj
        self._sync_resource_maps()

    def add_teacher(self, teacher: Teacher): 
        self.teachers[teacher.code] = teacher
        self._sync_resource_maps()

    def add_room(self, room: Room): 
        self.rooms[room.code] = room
        self._sync_resource_maps()

    def _apply_indices(self, blocks: List[GeneticBlock]):
        """Fill cached indices for NumPy optimization."""
        for b in blocks:
            b.class_idx = self.c_id_map.get(b.class_code, -1)
            b.teacher_idx = self.t_id_map.get(b.teacher.code, -1)
            b.room_idx = self.r_id_map.get(b.room.code, -1)
            b.subject_idx = self.s_id_map.get(b.subject.code, -1)

    def _generate_random_individual(self) -> List[GeneticBlock]:
        """
        Collision-Aware Chromosome Generator.
        Tracks teacher and room occupancy as blocks are placed, ensuring the
        initial population starts nearly conflict-free. This is the single
        most important optimization for large institutional datasets.
        """
        blocks = []
        rooms_list = list(self.rooms.values())

        # ── Global Occupancy Trackers (across ALL classes in this chromosome) ──
        # Key: (day, period) -> set of teacher codes already in use
        teacher_busy: Dict[Tuple, Set[str]] = defaultdict(set)
        # Key: (day, period) -> set of room codes already in use
        room_busy: Dict[Tuple, Set[str]] = defaultdict(set)
        # Key: (day, period, class_code, batch) -> bool
        class_busy: Dict[Tuple, bool] = {}

        def is_slot_free(start_idx: int, size: int, t_code: str, r_code: str,
                         c_code: str, batch: int) -> bool:
            """Check if a slot range is free for the given teacher, room, and class."""
            for offset in range(size):
                si = start_idx + offset
                if si >= len(self.all_slots):
                    return False
                s = self.all_slots[si]
                dp = (s.day, s.period)
                if t_code in teacher_busy[dp]:
                    return False
                if r_code in room_busy[dp]:
                    return False
                # Class collision: check whole class AND batch
                if class_busy.get((s.day, s.period, c_code, 0)):
                    return False
                if batch != 0:
                    if class_busy.get((s.day, s.period, c_code, batch)):
                        return False
                else:
                    for b_idx in range(1, 5):
                        if class_busy.get((s.day, s.period, c_code, b_idx)):
                            return False
            return True

        def mark_slot_used(start_idx: int, size: int, t_code: str, r_code: str,
                           c_code: str, batch: int):
            for offset in range(size):
                si = start_idx + offset
                if si >= len(self.all_slots):
                    break
                s = self.all_slots[si]
                dp = (s.day, s.period)
                teacher_busy[dp].add(t_code)
                room_busy[dp].add(r_code)
                class_busy[(s.day, s.period, c_code, batch)] = True

        def subject_priority(s):
            """Sort subjects — hardest to fit goes first."""
            p = 0
            if s.requires_lab or s.requires_double_period:
                p -= 3
            if len(s.teachers) == 1:
                p -= 2
            if s.periods_per_week >= 4:
                p -= 1
            return p

        # ── Global teacher load counter (persists across all classes) ──────
        teacher_load_counter: Dict[str, int] = {}

        for cls in sorted(self.classes.values(), key=lambda x: x.code):
            class_day_load: Dict[str, int] = {day: 0 for day in self.days}
            subject_day_used: Dict[Tuple, Set[str]] = defaultdict(set)
            prioritized_subjects = sorted(cls.subjects, key=subject_priority)

            for subject in prioritized_subjects:
                num_batches = 2 if subject.is_batch_specific else 1

                for b_idx in range(1, num_batches + 1):
                    batch_id = b_idx if subject.is_batch_specific else None
                    batch_val = batch_id if batch_id is not None else 0
                    total_needed: int = subject.periods_per_week

                    block_size = subject.consecutive_periods
                    if (subject.requires_lab or subject.requires_double_period) and block_size < 2:
                        block_size = 2

                    # ── Teacher Selection: O(1) load-balanced choice ──────────
                    if not subject.teachers:
                        continue

                    t_code = min(subject.teachers, key=lambda tc: teacher_load_counter.get(tc, 0))
                    teacher = self.teachers[t_code]

                    eff_strength = cls.strength
                    if batch_id is not None and cls.num_batches > 1:
                        eff_strength = (eff_strength + cls.num_batches - 1) // cls.num_batches
                    comp_rooms = [r for r in rooms_list if r.is_lab == subject.requires_lab and r.capacity >= eff_strength]
                    dept_rooms = [r for r in comp_rooms if r.department_code == cls.department_code]
                    room_pool = dept_rooms if dept_rooms else comp_rooms

                    subj_batch_key = (subject.code, batch_id)

                    while total_needed > 0:
                        size = min(block_size, total_needed)

                        # Build candidate slots from the pre-computed cache
                        days_already_used = subject_day_used[subj_batch_key]
                        sorted_days = sorted(
                            self.days,
                            key=lambda d: (
                                1 if d in days_already_used else 0,
                                class_day_load[d],
                                random.random()
                            )
                        )

                        start_idx = None
                        chosen_day = None
                        chosen_room = None

                        # Try days in priority order, find a conflict-free slot
                        for day in sorted_days:
                            # Safe copy — never mutate the cached list
                            day_slots = list(self._valid_starts_cache.get(size, {}).get(day, []))
                            random.shuffle(day_slots)

                            for cand_idx in day_slots:
                                # Try each room in the pool
                                for r in random.sample(room_pool, len(room_pool)):
                                    if is_slot_free(cand_idx, size, t_code, r.code,
                                                   cls.code, batch_val):
                                        start_idx = cand_idx
                                        chosen_day = day
                                        chosen_room = r
                                        break
                                if start_idx is not None:
                                    break
                            if start_idx is not None:
                                break

                        # Fallback: if no clean slot found, pick any valid slot
                        # (GA will fix remaining conflicts via evolution)
                        if start_idx is None:
                            all_valid = [
                                i for lst in self._valid_starts_cache.get(size, {}).values()
                                for i in lst
                            ]
                            start_idx = random.choice(all_valid) if all_valid else 0
                            chosen_day = self.all_slots[start_idx].day
                            chosen_room = random.choice(room_pool) if room_pool else rooms_list[0]

                        mark_slot_used(start_idx, size, t_code,
                                       chosen_room.code, cls.code, batch_val)
                        class_day_load[chosen_day] += size
                        subject_day_used[subj_batch_key].add(chosen_day)
                        teacher_load_counter[t_code] = teacher_load_counter.get(t_code, 0) + 1
                        blocks.append(GeneticBlock(
                            cls.code, subject, teacher, chosen_room,
                            start_idx, size, batch_id
                        ))
                        total_needed -= size

        self._apply_indices(blocks)
        return blocks

    def calculate_fitness(self, blocks: List[GeneticBlock]) -> float:
        """High-Performance Vectorized Fitness Scoring."""
        score: float = 100000.0
        
        # Phase 1: Core Conflicts (Fully Vectorized)
        collisions, hard_violations, spread_violations, penalties = self._evaluate_chromosome(blocks)
        self._last_penalties = penalties
        
        score -= collisions * 5000
        score -= hard_violations * 2000
        score -= spread_violations * 25000 
        
        # Phase 2: Load Balancing (NumPy)
        num_days = len(self.days)
        num_classes = len(self.classes)
        num_teachers = len(self.teachers)
        num_periods = self.max_periods_per_day + 10 # Buffer
        
        class_loads = np.zeros((num_classes, num_days))
        for b in blocks:
            d_idx = self._slot_to_dp[b.start_slot_idx, 0]
            class_loads[b.class_idx, d_idx] += b.size
            
        variances = np.var(class_loads, axis=1)
        score -= np.sum(variances) * 1500 

        # Phase 3: Qualitative Matrix Ops (Compactness, Morning Priority)
        c_activity = np.zeros((num_classes, num_days, num_periods), dtype=np.int8)
        t_activity = np.zeros((num_teachers, num_days, num_periods), dtype=np.int8)
        
        for b in blocks:
            d_idx = self._slot_to_dp[b.start_slot_idx, 0]
            p_start = self._slot_to_dp[b.start_slot_idx, 1]
            for offset in range(b.size):
                p = p_start + offset
                if p < num_periods:
                    c_activity[b.class_idx, d_idx, p] = 1
                    t_activity[b.teacher_idx, d_idx, p] = 1
                    if p <= 3: score += 300 
                    elif p > 5: score -= 500

        # Blank day penalty
        total_needed = np.array([sum(s.periods_per_week for s in cls.subjects) for cls in self.classes.values()])
        blank_days = np.sum(class_loads == 0, axis=1)
        score -= np.sum((total_needed >= num_days) * blank_days * 5000)

        # Gap Detection (Vectorized-style scan)
        def count_gaps_in_activity(activity_matrix):
            total_gaps = 0
            for d in range(num_days):
                day_data = activity_matrix[:, d, :]
                for i in range(day_data.shape[0]):
                    active_ps = np.where(day_data[i] > 0)[0]
                    if len(active_ps) > 1:
                        gap = (active_ps[-1] - active_ps[0] + 1) - len(active_ps)
                        for br in self.breaks:
                            if active_ps[0] < br < active_ps[-1]:
                                gap -= 1
                        if gap > 0: total_gaps += gap
            return total_gaps

        c_gaps = count_gaps_in_activity(c_activity)
        t_gaps = count_gaps_in_activity(t_activity)
        
        score -= c_gaps * 3000
        score -= t_gaps * 5000

        self._last_spread_violations = spread_violations
        return float(score)

    def crossover(self, p1: List[GeneticBlock], p2: List[GeneticBlock]) -> List[GeneticBlock]:
        """Class-Day Exchange Crossover: preserves compact daily schedules by swapping entire class-days."""
        if not p1 or not p2: return p1 or p2
        
        # Identify all (Class, Day) pairs present
        # Since we use blocks, we need to map blocks to these pairs
        class_codes = list(self.classes.keys())
        days = self.days
        
        # We'll swap random (Class, Day) sets between parents
        # This preserves the internal correctness of a single class's day in one parent
        cut_class = random.choice(class_codes)
        
        child = []
        # Take blocks of the cut_class from p1, and others from p2?
        # A more robust version:
        for b in p1:
            if b.class_code == cut_class:
                child.append(GeneticBlock(b.class_code, b.subject, b.teacher, b.room, b.start_slot_idx, b.size, b.batch_index))
        
        for b in p2:
            if b.class_code != cut_class:
                child.append(GeneticBlock(b.class_code, b.subject, b.teacher, b.room, b.start_slot_idx, b.size, b.batch_index))
                
        self._apply_indices(child)
        return child

    def mutate(self, chromosome: List[GeneticBlock], urgent: bool = False):
        """Priority-Aware Repair Mutation: fix the most constrained blocks first."""
        rate = 0.5 if urgent else self.mutation_rate
        if random.random() < rate:
            bad_blocks = [idx for idx, p in self._last_penalties.items() if p > 0]
            if bad_blocks and random.random() < 0.7:
                # Among bad blocks, prefer fixing labs and single-teacher subjects first
                def block_urgency(idx):
                    b = chromosome[idx]
                    score = self._last_penalties.get(idx, 0) * 10
                    if b.subject.requires_lab or b.subject.requires_double_period:
                        score += 30  # Labs are hardest to relocate
                    if len(b.subject.teachers) == 1:
                        score += 20  # Single-teacher subjects have no swap option
                    if b.subject.periods_per_week >= 4:
                        score += 10  # High-frequency subjects need early resolution
                    return score

                # Weighted choice: pick the most urgent bad block with higher probability
                bad_blocks_sorted = sorted(bad_blocks, key=block_urgency, reverse=True)
                # 70% chance to pick from top-3 urgent, 30% fully random bad block
                if random.random() < 0.7 and bad_blocks_sorted:
                    idx = bad_blocks_sorted[0] if len(bad_blocks_sorted) <= 3 else random.choice(bad_blocks_sorted[:3])
                else:
                    idx = random.choice(bad_blocks)
            else:
                idx = random.randint(0, len(chromosome) - 1)
                
            b = chromosome[idx]

            # Build the set of days that already have a block of this subject
            # (from sibling genes in the same chromosome, excluding this gene).
            days_with_subject: Set[str] = set()
            for other_idx, other_b in enumerate(chromosome):
                if other_idx == idx:
                    continue
                if (other_b.class_code == b.class_code and
                        other_b.subject.code == b.subject.code and
                        other_b.batch_index == b.batch_index):
                    days_with_subject.add(self.all_slots[other_b.start_slot_idx].day)

            cur_day_starts = self._valid_starts_cache.get(b.size, {})
            preferred_valid = []
            fallback_valid = []
            
            for day, idxs in cur_day_starts.items():
                if day in days_with_subject:
                    fallback_valid.extend(idxs)
                else:
                    preferred_valid.extend(idxs)

            valid = preferred_valid if preferred_valid else fallback_valid
            if valid:
                b.start_slot_idx = random.choice(valid)
                if random.random() < 0.4:
                    b.teacher = self.teachers[random.choice(b.subject.teachers)]
                    cls_obj = self.classes[b.class_code]
                    eff_strength = cls_obj.strength
                    if b.batch_index is not None and cls_obj.num_batches > 1:
                        eff_strength = (eff_strength + cls_obj.num_batches - 1) // cls_obj.num_batches
                    comp_rooms = [r for r in list(self.rooms.values()) if r.is_lab == b.subject.requires_lab and r.capacity >= eff_strength]
                    if comp_rooms: b.room = random.choice(comp_rooms)
            
            self._apply_indices(chromosome)

    def hill_climb(self, chromosome: List[GeneticBlock], iterations: int = 5):
        """Local search: specifically find and fix collisions/violations in a chromosome"""
        for _ in range(iterations):
            bad_indices = [idx for idx, p in self._last_penalties.items() if p > 0]
            if not bad_indices: break
            
            # Target one random bad block
            idx = random.choice(bad_indices)
            b = chromosome[idx]
            original_idx = b.start_slot_idx
            
            # Search for a better slot
            best_local_idx = original_idx
            min_violations = self._last_penalties[idx]
            
            valid_starts = self._valid_starts_cache.get(b.size, {})
            all_valid = [i for lst in valid_starts.values() for i in lst]
            
            # Use cached fitness structures to check for clashing
            # This is much faster than the old O(N) loop
            sample_size = min(len(all_valid), 15)
            for test_idx in random.sample(all_valid, sample_size):
                b.start_slot_idx = test_idx
                # Quick-check vs others
                # In hill_climb, we don't have the current matrices, so we do a quick O(N) but restricted
                col = 0
                for j in range(b.size):
                    s = self.all_slots[test_idx + j]
                    dp = self._slot_to_dp[test_idx + j]
                    d, p = dp[0], dp[1]
                    # Check specifically for current block's resources
                    # We'll use a local search shortcut
                    for other_idx, other_b in enumerate(chromosome):
                        if other_idx == idx: continue
                        # Does other_b cover (d, p)?
                        # This is still O(N), but we skip if day doesn't match
                        other_start_dp = self._slot_to_dp[other_b.start_slot_idx]
                        if other_start_dp[0] != d: continue
                        
                        if p >= other_start_dp[1] and p < other_start_dp[1] + other_b.size:
                            is_class_col = False
                            if b.class_idx == other_b.class_idx:
                                if b.batch_index is None or other_b.batch_index is None:
                                    is_class_col = True
                                elif b.batch_index == other_b.batch_index:
                                    is_class_col = True
                            
                            if b.teacher_idx == other_b.teacher_idx or \
                               b.room_idx == other_b.room_idx or \
                               is_class_col:
                                col += 1
                                break
                    if col >= min_violations: break
                
                if col < min_violations:
                    min_violations = col
                    best_local_idx = test_idx
                if min_violations == 0: break
            
            b.start_slot_idx = best_local_idx
            
            # --- Advanced: Scientific Subject Spread ---
            # Try to push same-subject blocks to different days if they are clumped
            sub_map = defaultdict(set)
            for idx, blk in enumerate(chromosome):
                sub_map[(blk.class_code, blk.subject.code)].add(idx)
            
            for (c_code, s_code), idxs in sub_map.items():
                if len(idxs) > 1:
                    days_used = [self.all_slots[chromosome[i].start_slot_idx].day for i in idxs]
                    if len(set(days_used)) < len(days_used):
                        # Clumped! Target a redundant block and move it to a fresh day
                        bad_idx = next(i for i in idxs if days_used.count(self.all_slots[chromosome[i].start_slot_idx].day) > 1)
                        orig_start = chromosome[bad_idx].start_slot_idx
                        target_days = set(self.days) - set(days_used)
                        if target_days:
                            new_day = random.choice(list(target_days))
                            new_slots = [i for i, s in enumerate(self.all_slots) if s.day == new_day and i + chromosome[bad_idx].size <= len(self.all_slots)]
                            if new_slots:
                                chromosome[bad_idx].start_slot_idx = random.choice(new_slots)
                                if not self._is_clash_free(chromosome):
                                    chromosome[bad_idx].start_slot_idx = orig_start

    def generate_timetable(self) -> bool:
        """Main GA Loop — Time-Aware & Adaptive for 15-30s target."""
        import time
        TIME_LIMIT = 28  # Hard cutoff at 28s, leaving 2s buffer for extraction
        start_time = time.perf_counter()

        # ── Adaptive Population Scaling ──────────────────────────────────────
        # Scale population to class count so small problems finish in <5s
        # and large problems still converge within the time window.
        num_classes = len(self.classes)
        if num_classes <= 3:
            self.population_size = 50
            self.max_generations = 500
        elif num_classes <= 6:
            self.population_size = 80
            self.max_generations = 800
        elif num_classes <= 10:
            self.population_size = 120
            self.max_generations = 1500
        else:  # Large institutional dataset (10+ classes)
            self.population_size = 150
            self.max_generations = 2000

        population = []

        # ── Seeding Phase ─────────────────────────────────────────────────────
        # Fast seeding: 1 seed for small, up to 3 for large (each takes ~1-2s)
        seeds_count = 1 if num_classes <= 5 else min(3, num_classes // 4)
        print(f"GA Seeding: Generating {seeds_count} initial seeds...")
        seed_gen = SeedGenerator(self)

        for _ in range(seeds_count):
            # Stop seeding if we're already using too much time
            if time.perf_counter() - start_time > 5:
                print("Seeding time limit reached, filling with random...")
                break
            seed = seed_gen.generate_seed()
            if seed and len(seed) > 0:
                population.append((self.calculate_fitness(seed), seed))

        # Fill remaining with random individuals
        while len(population) < self.population_size:
            ind = self._generate_random_individual()
            score = self.calculate_fitness(ind)
            population.append((score, ind))

        best_overall_score = float('-inf')
        stagnation = 0

        for gen in range(self.max_generations):
            # ── Hard Time Limit Check ─────────────────────────────────────────
            elapsed = time.perf_counter() - start_time
            if elapsed >= TIME_LIMIT:
                print(f"Time limit reached at Gen {gen} ({elapsed:.1f}s). Extracting best result...")
                break

            population.sort(key=lambda x: x[0], reverse=True)
            best_score, best = population[0]

            # ── Violation Check ───────────────────────────────────────────────
            final_collisions, final_hard_v, final_spread_v, _ = self._evaluate_chromosome(best)

            # ── Aggressive Early-Exit: Accept if conflict-free ────────────────
            # We no longer wait for perfect spread — a conflict-free schedule is
            # a valid, usable timetable. Spread is a soft quality bonus.
            if final_collisions == 0 and final_hard_v == 0:
                print(f"GA Success at Gen {gen}! ({elapsed:.1f}s) Spread violations: {final_spread_v}. Compacting...")
                self._compact_all(best)
                self.schedule = []
                for b in best:
                    self.schedule.extend(b.get_entries(self.all_slots))
                return True

            # ── Memetic Step: Hill climbing ───────────────────────────────────
            if gen % 10 == 0 or stagnation > 50:
                for i in range(min(len(population), 3)):
                    child = population[i][1]
                    self.calculate_fitness(child)
                    self.hill_climb(child, iterations=10 if final_collisions > 5 else 5)
                    population[i] = (self.calculate_fitness(child), child)

            # ── Stagnation Handling ───────────────────────────────────────────
            if best_score > best_overall_score:
                best_overall_score = best_score
                stagnation = 0
            else:
                stagnation += 1
                # On stagnation, re-seed FASTER by using fewer seeds
                if stagnation > 80:
                    print(f"Stagnation at Gen {gen} ({elapsed:.1f}s). Re-seeding...")
                    new_pop = population[:5]
                    # Only inject 2 seeds to conserve time
                    for _ in range(2):
                        if time.perf_counter() - start_time < TIME_LIMIT - 5:
                            s = seed_gen.generate_seed()
                            if s:
                                new_pop.append((self.calculate_fitness(s), s))
                    while len(new_pop) < self.population_size:
                        ind = self._generate_random_individual()
                        new_pop.append((self.calculate_fitness(ind), ind))
                    population = new_pop
                    stagnation = 0

            # ── Next Generation ───────────────────────────────────────────────
            next_gen = list(population[:20])
            while len(next_gen) < self.population_size:
                t1, t2 = random.choice(population[:50]), random.choice(population[:50])
                p1 = t1[1] if t1[0] > t2[0] else t2[1]

                child = self.crossover(p1, random.choice(population[:40])[1])
                self.calculate_fitness(child)
                self.mutate(child, urgent=(final_collisions > 0 or final_hard_v > 0))

                if random.random() < 0.08:  # Reduced hill-climb rate for speed
                    self.hill_climb(child, iterations=2)

                child_score = self.calculate_fitness(child)
                next_gen.append((child_score, child))

            population = next_gen
            if gen % 25 == 0:
                print(f"Gen {gen} ({elapsed:.1f}s) | Col: {final_collisions} | Hard: {final_hard_v} | Spread: {final_spread_v} | Score: {best_score:.0f}")

        # ── Fallback: Time limit or generation limit hit ───────────────────────
        # Accept the best result found so far if it has zero hard conflicts.
        population.sort(key=lambda x: x[0], reverse=True)
        best_final = population[0][1]
        final_col, final_hard, final_spread, _ = self._evaluate_chromosome(best_final)
        self._compact_all(best_final)
        self.schedule = []
        for b in best_final:
            self.schedule.extend(b.get_entries(self.all_slots))
        return final_col == 0 and final_hard == 0

    def _evaluate_chromosome(self, chromosome: List[GeneticBlock]) -> Tuple[int, int, int, Dict[int, int]]:
        """Fully Vectorized evaluation of a chromosome using NumPy."""
        # Use buffers to prevent IndexError during messy intermediate generations
        num_days = len(self.days) + 2
        num_periods = self.max_periods_per_day + 10
        
        # 1. Collect Block Data into Vectors
        # We pre-cached these IDs in _apply_indices
        c_idxs = np.array([b.class_idx for b in chromosome], dtype=np.int32)
        t_idxs = np.array([b.teacher_idx for b in chromosome], dtype=np.int32)
        r_idxs = np.array([b.room_idx for b in chromosome], dtype=np.int32)
        s_idxs = np.array([b.subject_idx for b in chromosome], dtype=np.int32)
        start_idxs = np.array([b.start_slot_idx for b in chromosome], dtype=np.int32)
        sizes = np.array([b.size for b in chromosome], dtype=np.int32)
        batch_idxs = np.array([b.batch_index if b.batch_index is not None else 0 for b in chromosome], dtype=np.int8)

        # 2. Map start_slot_idx to (DayIdx, PeriodIdx)
        if not hasattr(self, '_slot_to_dp'):
            self._slot_to_dp = np.array([(self.day_map[s.day], s.period) for s in self.all_slots], dtype=np.int32)
        
        d_idxs = self._slot_to_dp[start_idxs, 0]
        p_idxs = self._slot_to_dp[start_idxs, 1]

        # 3. Hard Constraints (Vectorized)
        p_vec = np.zeros(len(chromosome), dtype=np.int32)

        # Room Capacity & Lab Affinity
        eff_strength = np.where(batch_idxs > 0, self.c_batch_strength_arr[c_idxs], self.c_strength_arr[c_idxs])
        cap_fail = self.r_cap_arr[r_idxs] < eff_strength
        lab_fail = self.r_lab_arr[r_idxs] != self.s_lab_arr[s_idxs]
        block_fail = (self.s_double_arr[s_idxs] == 1) & (sizes < 2)
        
        p_vec += cap_fail.astype(int) + lab_fail.astype(int) + (block_fail.astype(int) * 2)
        hard_violations = np.sum(p_vec)

        # 4. Occupancy Tracking (Vectorized)
        tb_arr = np.zeros((num_days, num_periods, len(self.teachers)), dtype=np.int8)
        rb_arr = np.zeros((num_days, num_periods, len(self.rooms)), dtype=np.int8)
        cb_arr = np.zeros((num_days, num_periods, len(self.classes), 5), dtype=np.int8)
        
        collisions = 0
        
        max_b_size = np.max(sizes) if len(sizes) > 0 else 0
        for offset in range(max_b_size):
            mask = offset < sizes
            if not np.any(mask): break
            
            curr_p = p_idxs[mask] + offset
            curr_d = d_idxs[mask]
            curr_t = t_idxs[mask]
            curr_r = r_idxs[mask]
            curr_c = c_idxs[mask]
            curr_batch = batch_idxs[mask]
            
            # Unavailability check
            t_un = self.t_unavail_mask[curr_d, curr_p, curr_t]
            r_un = self.r_unavail_mask[curr_d, curr_p, curr_r]
            p_vec[mask] += (t_un + r_un)
            hard_violations += np.sum(t_un + r_un)
            
            # Recess check
            if offset == 0:
                for br in self.breaks:
                    br_fail = (p_idxs <= br) & (p_idxs + sizes - 1 >= br + 1) & (sizes > 1)
                    p_vec += br_fail.astype(int) * 2
                    hard_violations += np.sum(br_fail) * 2

            # Collision Logic
            for i, idx in enumerate(np.where(mask)[0]):
                d, p, t, r, c, b_val = curr_d[i], curr_p[i], curr_t[i], curr_r[i], curr_c[i], curr_batch[i]
                
                clash = False
                if tb_arr[d, p, t] > 0: clash = True
                if rb_arr[d, p, r] > 0: clash = True
                if b_val == 0:
                    if np.any(cb_arr[d, p, c, :] > 0): clash = True
                elif cb_arr[d, p, c, 0] > 0 or cb_arr[d, p, c, b_val] > 0:
                    clash = True
                    
                if clash:
                    collisions += 1
                    p_vec[idx] += 1
                
                tb_arr[d, p, t] = 1
                rb_arr[d, p, r] = 1
                cb_arr[d, p, c, b_val] = 1

        # 5. Spread Violations
        spread_violations = 0
        subj_day_count = defaultdict(int) 
        for i, b in enumerate(chromosome):
            subj_day_count[(b.class_idx, d_idxs[i], b.subject_idx)] += 1
            
        for key, count in subj_day_count.items():
            if count > 1:
                spread_violations += (count - 1)

        penalties = {idx: p for idx, p in enumerate(p_vec)}
        return collisions, int(hard_violations), spread_violations, penalties

    def _compact_all(self, chromosome: List[GeneticBlock]):
        """Post-processing: slide blocks to earlier slots if no clash."""
        class_codes = sorted(list(self.classes.keys()))
        for c_code in class_codes:
            c_blocks = sorted([b for b in chromosome if b.class_code == c_code], 
                             key=lambda x: x.start_slot_idx)
            for b in c_blocks:
                original_idx = b.start_slot_idx
                original_day = self.all_slots[original_idx].day
                
                sibling_days: Set[str] = set()
                for other in chromosome:
                    if other is b: continue
                    if (other.class_code == c_code and other.subject.code == b.subject.code and other.batch_index == b.batch_index):
                        sibling_days.add(self.all_slots[other.start_slot_idx].day)
                
                for target_idx in range(original_idx):
                    target_day = self.all_slots[target_idx].day
                    # STRICT: Only allow compacting within the SAME day to preserve load balance
                    if target_day != original_day: continue
                    
                    if target_idx + b.size > len(self.all_slots): continue
                    if self.all_slots[target_idx].day != self.all_slots[target_idx + b.size - 1].day: continue
                    if target_day in sibling_days: continue
                    
                    target_start_p = self.all_slots[target_idx].period
                    target_end_p = target_start_p + b.size - 1
                    if any(target_start_p <= br and target_end_p >= br + 1 for br in self.breaks):
                        continue
                    
                    b.start_slot_idx = target_idx
                    if self._is_clash_free(chromosome): break
                    else: b.start_slot_idx = original_idx

    def _is_clash_free(self, chromosome: List[GeneticBlock]) -> bool:
        """Fast clash check using vectorized evaluation."""
        c, h, _, _ = self._evaluate_chromosome(chromosome)
        return c == 0 and h == 0

    def get_class_timetable(self, class_code: str) -> Dict[str, Any]:
        timetable: Dict[str, Any] = {day: {} for day in self.days}
        for entry in self.schedule:
            if entry.class_code == class_code:
                timetable[entry.time_slot.day][entry.time_slot.period] = {
                    'subject': entry.subject.name, 'subject_code': entry.subject.code,
                    'teacher': entry.teacher.name, 'room': entry.room.name, 'is_lab': entry.room.is_lab
                }
        return timetable

    def get_teacher_timetable(self, teacher_code: str) -> Dict[str, Any]:
        timetable: Dict[str, Any] = {day: {} for day in self.days}
        for entry in self.schedule:
            if entry.teacher.code == teacher_code:
                timetable[entry.time_slot.day][entry.time_slot.period] = {
                    'class': entry.class_code, 'subject': entry.subject.name, 'room': entry.room.name
                }
        return timetable

class TimetableAnalytics:
    """Advanced Diagnostic Analytics Engine (Pandas-Based)"""
    def __init__(self, schedule: List[TimetableEntry], days: List[str], periods_per_day: int):
        self.schedule = schedule
        self.days = days
        self.periods_per_day = periods_per_day
        self.df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        data = []
        for e in self.schedule:
            data.append({
                'day': e.time_slot.day, 'period': e.time_slot.period, 'class': e.class_code,
                'subject': e.subject.name, 'subject_code': e.subject.code, 'teacher': e.teacher.name,
                'room': e.room.name, 'is_lab': e.room.is_lab
            })
        return pd.DataFrame(data)

    def analyze_teacher_load(self) -> pd.DataFrame:
        if self.df.empty: return pd.DataFrame()
        load = self.df.groupby('teacher').size().reset_index(name='total_weekly_periods')
        daily = self.df.groupby(['teacher', 'day']).size().reset_index(name='daily_count')
        max_daily = daily.groupby('teacher')['daily_count'].max().reset_index(name='max_periods_per_day')
        return pd.merge(load, max_daily, on='teacher')

    def analyze_class_gaps(self) -> Dict[str, float]:
        if self.df.empty: return {}
        gap_scores = {}
        for c_code in self.df['class'].unique():
            c_df = self.df[self.df['class'] == c_code]
            total_gaps = 0
            for day in self.days:
                periods = sorted(c_df[c_df['day'] == day]['period'].tolist())
                if len(periods) > 1:
                    span = periods[-1] - periods[0] + 1
                    total_gaps += (span - len(periods))
            gap_scores[c_code] = total_gaps
        return gap_scores

    def analyze_room_utilization(self) -> pd.DataFrame:
        if self.df.empty: return pd.DataFrame()
        total_slots = len(self.days) * self.periods_per_day
        usage = self.df.groupby(['room', 'is_lab']).size().reset_index(name='occupied_slots')
        usage['utilization_pct'] = (usage['occupied_slots'] / total_slots) * 100
        return usage.sort_values('utilization_pct', ascending=False)

    def get_diagnostic_report(self) -> Dict[str, Any]:
        if self.df.empty: return {"error": "No data"}
        teacher_analysis = self.analyze_teacher_load()
        class_gaps = self.analyze_class_gaps()
        room_use = self.analyze_room_utilization()
        
        # Load Variance (Measure of how evenly the week is distributed)
        # Higher variance means "clumpy" schedules.
        daily_counts = self.df.groupby(['class', 'day']).size().unstack(fill_value=0)
        variance = daily_counts.var(axis=1).mean() if not daily_counts.empty else 0

        return {
            "overall_stats": {
                "total_periods": len(self.df),
                "avg_teacher_load": float(teacher_analysis['total_weekly_periods'].mean()) if not teacher_analysis.empty else 0,
                "total_gaps": sum(class_gaps.values()),
                "max_room_utilization": float(room_use['utilization_pct'].max()) if not room_use.empty else 0,
                "load_variance": float(variance)
            },
            "teacher_efficiency": teacher_analysis.to_dict(orient='records'),
            "class_compactness": class_gaps,
            "room_usage": room_use.to_dict(orient='records')
        }

class SeedGenerator:
    """
    Bridges Backtracking and Genetic algorithms.
    Uses TimetableScheduler to generate 'Skeleton' individuals for GA seeding.
    """
    def __init__(self, scheduler: "GeneticScheduler"):
        self.days = scheduler.days
        self.periods_per_day = scheduler.max_periods_per_day
        self.breaks = scheduler.breaks
        self.day_configs = scheduler.day_limits
        self.ga_scheduler = scheduler

    def generate_seed(self) -> Optional[List[GeneticBlock]]:
        """Generates one valid/partially-valid chromosome using backtracking."""
        # No need for redundant import if in same file
        
        # 1. Create a backtracking instance with SAME constraints
        bt = TimetableScheduler(self.days, self.periods_per_day, self.breaks, self.day_configs)
        
        # 2. Sync resources
        bt.classes = self.ga_scheduler.classes
        bt.teachers = self.ga_scheduler.teachers
        bt.rooms = self.ga_scheduler.rooms
        bt.subjects = self.ga_scheduler.subjects
        
        # 3. Generate a quick schedule
        # We limit attempts to keep it FAST. We don't need a PERFECT schedule, just a good SEED.
        bt.max_attempts = 500 
        success = bt.generate_timetable()
        
        if not bt.schedule:
            return None
            
        # 4. Convert TimetableEntry -> GeneticBlock
        blocks: List[GeneticBlock] = []
        all_slots_list = self.ga_scheduler.all_slots
        
        # Group by contiguous units
        # Sort by class, then day, then period to find blocks
        entries = sorted(bt.schedule, key=lambda e: (e.class_code, e.time_slot.day, e.time_slot.period))
        
        i = 0
        while i < len(entries):
            e = entries[i]
            # Find how many contiguous entries of SAME subject/teacher/room there are
            size = 1
            while (i + size < len(entries) and 
                   entries[i+size].class_code == e.class_code and
                   entries[i+size].subject.code == e.subject.code and
                   entries[i+size].teacher.code == e.teacher.code and
                   entries[i+size].room.code == e.room.code and
                   entries[i+size].time_slot.day == e.time_slot.day and
                   entries[i+size].time_slot.period == e.time_slot.period + size):
                size += 1
            
            # Find start_slot_idx in GA's all_slots
            try:
                # Find index of TimeSlot(e.time_slot.day, e.time_slot.period) in ga_scheduler.all_slots
                start_slot = next(idx for idx, s in enumerate(all_slots_list) 
                                if s.day == e.time_slot.day and s.period == e.time_slot.period)
                
                blocks.append(GeneticBlock(
                    e.class_code, e.subject, e.teacher, e.room,
                    start_slot, size, e.batch_index
                ))
            except StopIteration:
                pass # Should not happen if slots match
                
            i += size
            
        return blocks