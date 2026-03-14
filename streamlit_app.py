
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, time as dtime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import re

import requests
import streamlit as st


# ============================================================
# Torn Stat Tracker v2 - Merged Build (fixed)
# ------------------------------------------------------------
# Includes:
# - Torn API v2 sync
# - Real train-by-train gym gain engine
# - Auto-detected and manual training modifiers
# - Jump scheduler merged into the same gain model
# - War / non-training day handling
# - Battlestats parser fix for Torn v2 nested value objects
# - Stable unlocked gym multiselect with quick-fill helper
# ============================================================

STAT_KEYS = ["strength", "speed", "defense", "dexterity"]
PROPERTY_POOL_NAMES = {"Small Pool", "Medium Pool", "Large Pool"}

# Gym gain formula constants from Torn wiki.
FORMULA_A = 3.480061091e-7
FORMULA_B = 250.0
FORMULA_C = 3.091619094e-6
FORMULA_D = 6.82775184551527e-5
FORMULA_E = -0.0301431777

# Happy-loss model: Torn wiki says gym training removes 40-60% of the
# energy used per train from happy. We use the midpoint for planning.
DEFAULT_EXPECTED_HAPPY_LOSS_RATIO = 0.50

# API
TORN_V2_BASE_URL = "https://api.torn.com/v2"
TORN_API_TIMEOUT_SECONDS = 20
TORN_API_COMMENT = "torn-stat-tracker-v2"


@dataclass
class PlayerStats:
    strength: float = 0.0
    speed: float = 0.0
    defense: float = 0.0
    dexterity: float = 0.0

    def total(self) -> float:
        return self.strength + self.speed + self.defense + self.dexterity

    def as_dict(self) -> Dict[str, float]:
        return {
            "strength": self.strength,
            "speed": self.speed,
            "defense": self.defense,
            "dexterity": self.dexterity,
        }

    def get(self, stat_key: str) -> float:
        return self.as_dict().get(stat_key, 0.0)

    def with_gain(self, stat_key: str, gain: float) -> "PlayerStats":
        values = self.as_dict()
        values[stat_key] = values.get(stat_key, 0.0) + gain
        return PlayerStats(**values)


@dataclass
class RatioProfile:
    strength: float = 30.86
    speed: float = 24.69
    defense: float = 22.22
    dexterity: float = 22.22

    def as_percent_map(self) -> Dict[str, float]:
        return {
            "strength": self.strength,
            "speed": self.speed,
            "defense": self.defense,
            "dexterity": self.dexterity,
        }


@dataclass
class RecoveryState:
    current_energy: int = 0
    max_energy: int = 150
    regen_per_tick: int = 5
    regen_minutes_per_tick: int = 10
    xanax_per_day: int = 3
    xanax_energy: int = 250
    daily_refill_enabled: bool = True
    refill_energy: int = 150
    drug_cd_minutes: int = 0
    booster_cd_minutes: int = 0
    current_happy: int = 0
    max_happy: int = 0
    donor: bool = True

    def natural_energy_per_day(self, energy_regen_bonus_pct: float = 0.0) -> int:
        base = (24 * 60 / self.regen_minutes_per_tick) * self.regen_per_tick
        return int(base * (1 + energy_regen_bonus_pct / 100.0))

    def baseline_energy_per_day(self, energy_regen_bonus_pct: float = 0.0) -> int:
        total = self.natural_energy_per_day(energy_regen_bonus_pct=energy_regen_bonus_pct)
        total += self.xanax_per_day * self.xanax_energy
        if self.daily_refill_enabled:
            total += self.refill_energy
        return total


@dataclass
class Gym:
    name: str
    tier: str
    energy_cost: int
    unlock_cost: int
    e_for_next_gym: Optional[int]
    gains: Dict[str, float]

    def gain_for(self, stat_key: str) -> float:
        return self.gains.get(stat_key, 0.0)


@dataclass
class TrainingModifiers:
    all_gym_gains_pct: float = 0.0
    strength_gym_gains_pct: float = 0.0
    speed_gym_gains_pct: float = 0.0
    defense_gym_gains_pct: float = 0.0
    dexterity_gym_gains_pct: float = 0.0
    happy_loss_reduction_pct: float = 0.0
    energy_regen_bonus_pct: float = 0.0
    detected_sources: List[str] = field(default_factory=list)

    def stat_specific_pct(self, stat_key: str) -> float:
        return {
            "strength": self.strength_gym_gains_pct,
            "speed": self.speed_gym_gains_pct,
            "defense": self.defense_gym_gains_pct,
            "dexterity": self.dexterity_gym_gains_pct,
        }.get(stat_key, 0.0)

    def gym_multiplier_for_stat(self, stat_key: str) -> float:
        return 1.0 + (self.all_gym_gains_pct + self.stat_specific_pct(stat_key)) / 100.0

    def merge(self, other: "TrainingModifiers") -> "TrainingModifiers":
        return TrainingModifiers(
            all_gym_gains_pct=self.all_gym_gains_pct + other.all_gym_gains_pct,
            strength_gym_gains_pct=self.strength_gym_gains_pct + other.strength_gym_gains_pct,
            speed_gym_gains_pct=self.speed_gym_gains_pct + other.speed_gym_gains_pct,
            defense_gym_gains_pct=self.defense_gym_gains_pct + other.defense_gym_gains_pct,
            dexterity_gym_gains_pct=self.dexterity_gym_gains_pct + other.dexterity_gym_gains_pct,
            happy_loss_reduction_pct=self.happy_loss_reduction_pct + other.happy_loss_reduction_pct,
            energy_regen_bonus_pct=self.energy_regen_bonus_pct + other.energy_regen_bonus_pct,
            detected_sources=self.detected_sources + other.detected_sources,
        )


@dataclass
class PlayerState:
    stats: PlayerStats = field(default_factory=PlayerStats)
    recovery: RecoveryState = field(default_factory=RecoveryState)
    unlocked_gyms: List[str] = field(default_factory=list)
    faction_war_days: List[date] = field(default_factory=list)
    torn_name: str = ""
    torn_id: Optional[int] = None
    faction_id: Optional[int] = None
    faction_name: str = ""
    training_modifiers: TrainingModifiers = field(default_factory=TrainingModifiers)
    api_notes: List[str] = field(default_factory=list)
    last_sync: Optional[datetime] = None


@dataclass
class GoalSettings:
    target_total_stats: float = 250_000_000
    target_date: date = field(default_factory=lambda: date.today() + timedelta(days=210))
    fhc_allowed: bool = True
    cans_allowed: bool = True
    auto_schedule_happy_jumps: bool = True
    schedule_99k_jump: bool = False
    scheduled_99k_jump_date: date = field(default_factory=lambda: date.today() + timedelta(days=7))
    scheduled_99k_jump_time: dtime = dtime(hour=0, minute=15)
    current_gym_energy_progress: int = 0
    skip_war_days: bool = True
    normal_day_start_happy: int = 5_000
    happy_jump_start_happy: int = 34_000
    super_happy_jump_start_happy: int = 99_999
    jump_stack_energy_target: int = 1_000
    jump_stack_xanax_uses: int = 4
    allow_jump_on_war_days: bool = False
    jump_min_extra_gain_pct: float = 12.0
    jump_prep_hours: float = 30.0
    assumed_xanax_cooldown_hours: float = 8.0


@dataclass
class JumpPlan:
    jump_type: str
    target_stat: str
    gym_name: str
    prep_start: datetime
    execute_at: datetime
    projected_normal_gain: float
    projected_jump_gain: float
    projected_gain_delta: float
    notes: List[str] = field(default_factory=list)


@dataclass
class JumpStep:
    when: datetime
    action: str
    details: str


@dataclass
class DailyInstruction:
    plan_date: date
    day_type: str
    target_stat: str
    gym_name: str
    estimated_energy: int
    recommended_trains: int
    estimated_gain: float
    start_happy: int
    end_happy: int
    notes: List[str] = field(default_factory=list)


@dataclass
class GymUnlockProjection:
    current_gym: str
    next_gym: Optional[str]
    current_progress: int
    required_progress: Optional[int]
    remaining_energy: Optional[int]
    estimated_unlock_at: Optional[datetime]


def build_gym_db() -> List[Gym]:
    return [
        Gym("Premier Fitness", "light", 5, 10, 200, {"strength": 2.0, "speed": 2.0, "defense": 2.0, "dexterity": 2.0}),
        Gym("Average Joes", "light", 5, 100, 500, {"strength": 2.4, "speed": 2.4, "defense": 2.7, "dexterity": 2.4}),
        Gym("Woody's Workout", "light", 5, 250, 1_000, {"strength": 2.7, "speed": 3.2, "defense": 3.0, "dexterity": 2.7}),
        Gym("Beach Bods", "light", 5, 500, 2_000, {"strength": 3.2, "speed": 3.2, "defense": 3.2, "dexterity": 0.0}),
        Gym("Silver Gym", "light", 5, 1_000, 2_750, {"strength": 3.4, "speed": 3.6, "defense": 3.4, "dexterity": 3.2}),
        Gym("Pour Femme", "light", 5, 2_500, 3_000, {"strength": 3.4, "speed": 3.6, "defense": 3.6, "dexterity": 3.8}),
        Gym("Davies Den", "light", 5, 5_000, 3_500, {"strength": 3.7, "speed": 0.0, "defense": 3.7, "dexterity": 3.7}),
        Gym("Global Gym", "light", 5, 10_000, 4_000, {"strength": 4.0, "speed": 4.0, "defense": 4.0, "dexterity": 4.0}),
        Gym("Knuckle Heads", "middle", 10, 50_000, 6_000, {"strength": 4.8, "speed": 4.4, "defense": 4.0, "dexterity": 4.2}),
        Gym("Pioneer Fitness", "middle", 10, 100_000, 7_000, {"strength": 4.4, "speed": 4.6, "defense": 4.8, "dexterity": 4.4}),
        Gym("Anabolic Anomalies", "middle", 10, 250_000, 8_000, {"strength": 5.0, "speed": 4.6, "defense": 5.2, "dexterity": 4.6}),
        Gym("Core", "middle", 10, 500_000, 11_000, {"strength": 5.0, "speed": 5.2, "defense": 5.0, "dexterity": 5.0}),
        Gym("Racing Fitness", "middle", 10, 1_000_000, 12_420, {"strength": 5.0, "speed": 5.4, "defense": 4.8, "dexterity": 5.2}),
        Gym("Complete Cardio", "middle", 10, 2_000_000, 18_000, {"strength": 5.5, "speed": 5.7, "defense": 5.5, "dexterity": 5.2}),
        Gym("Legs, Bums and Tums", "middle", 10, 3_000_000, 18_100, {"strength": 0.0, "speed": 5.5, "defense": 5.5, "dexterity": 5.7}),
        Gym("Deep Burn", "middle", 10, 5_000_000, 24_140, {"strength": 6.0, "speed": 6.0, "defense": 6.0, "dexterity": 6.0}),
        Gym("Apollo Gym", "heavy", 10, 7_500_000, 31_260, {"strength": 6.0, "speed": 6.2, "defense": 6.4, "dexterity": 6.2}),
        Gym("Gun Shop", "heavy", 10, 10_000_000, 36_610, {"strength": 6.5, "speed": 6.4, "defense": 6.2, "dexterity": 6.2}),
        Gym("Force Training", "heavy", 10, 15_000_000, 46_640, {"strength": 6.4, "speed": 6.5, "defense": 6.4, "dexterity": 6.8}),
        Gym("Cha Cha's", "heavy", 10, 20_000_000, 56_520, {"strength": 6.4, "speed": 6.4, "defense": 6.8, "dexterity": 7.0}),
        Gym("Atlas", "heavy", 10, 30_000_000, 67_775, {"strength": 7.0, "speed": 6.4, "defense": 6.4, "dexterity": 6.5}),
        Gym("Last Round", "heavy", 10, 50_000_000, 84_535, {"strength": 6.8, "speed": 6.5, "defense": 7.0, "dexterity": 6.5}),
        Gym("The Edge", "heavy", 10, 75_000_000, 106_305, {"strength": 6.8, "speed": 7.0, "defense": 7.0, "dexterity": 6.8}),
        Gym("George's", "heavy", 10, 100_000_000, None, {"strength": 7.3, "speed": 7.3, "defense": 7.3, "dexterity": 7.3}),
    ]


GYM_DB = build_gym_db()
GYM_INDEX = {gym.name: gym for gym in GYM_DB}


def ordered_gym_names() -> List[str]:
    return [gym.name for gym in GYM_DB]


class TornAPIError(Exception):
    pass


class TornAPISelectionError(TornAPIError):
    pass


def _clean_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        raise TornAPIError("API key is required.")
    return key


def _api_headers() -> Dict[str, str]:
    return {"Accept": "application/json", "User-Agent": "TornStatTrackerV2/fixed"}


def _api_get(path: str, api_key: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    key = _clean_api_key(api_key)
    url = f"{TORN_V2_BASE_URL}{path}"
    query: Dict[str, Any] = {"key": key, "comment": TORN_API_COMMENT}
    if params:
        query.update(params)
    response = requests.get(url, params=query, headers=_api_headers(), timeout=TORN_API_TIMEOUT_SECONDS)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise TornAPIError("Torn API returned a non-JSON response.") from exc
    if isinstance(payload, dict) and "error" in payload:
        error_blob = payload.get("error") or {}
        code = error_blob.get("code", "?")
        message = error_blob.get("error") or error_blob.get("message") or "Unknown API error"
        if int(code) == 16:
            raise TornAPISelectionError(f"Access level is too low for {path}: {message}")
        raise TornAPIError(f"Torn API error {code} on {path}: {message}")
    if not isinstance(payload, dict):
        raise TornAPIError(f"Unexpected response shape for {path}.")
    return payload


def _nested_get(payload: Any, path: Tuple[str, ...], default: Any = None) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _first_present(payload: Dict[str, Any], candidates: List[Tuple[str, ...]], default: Any = None) -> Any:
    for path in candidates:
        value = _nested_get(payload, path, default=None)
        if value is not None:
            return value
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _coerce_cooldown_minutes(value: Any) -> int:
    if isinstance(value, (int, float, str)):
        seconds = _safe_int(value, 0)
        return max(0, math.ceil(seconds / 60))
    if isinstance(value, dict):
        seconds = _safe_int(_first_present(value, [("remaining",), ("time",), ("seconds",), ("cooldown",), ("fulltime",)], default=0), 0)
        return max(0, math.ceil(seconds / 60))
    return 0


def _walk_objects(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _extract_strings(value: Any) -> List[str]:
    strings: List[str] = []
    if isinstance(value, dict):
        for child in value.values():
            strings.extend(_extract_strings(child))
    elif isinstance(value, list):
        for child in value:
            strings.extend(_extract_strings(child))
    elif isinstance(value, str):
        strings.append(value)
    return strings


def _find_first_numeric_for_key(value: Any, target_key: str) -> Optional[float]:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() == target_key.lower():
                if isinstance(child, (int, float, str)):
                    parsed = _safe_float(child, default=float("nan"))
                    if not math.isnan(parsed):
                        return parsed
                if isinstance(child, dict):
                    for nested_key in ("value", "current", "amount", "total"):
                        if nested_key in child:
                            parsed = _safe_float(child.get(nested_key), default=float("nan"))
                            if not math.isnan(parsed):
                                return parsed
            found = _find_first_numeric_for_key(child, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_first_numeric_for_key(child, target_key)
            if found is not None:
                return found
    return None


def _extract_gym_names_from_payloads(payloads: Iterable[Dict[str, Any]]) -> List[str]:
    found: set[str] = set()
    valid_names = set(ordered_gym_names())
    for payload in payloads:
        for obj in _walk_objects(payload):
            if not isinstance(obj, dict):
                continue
            for key in ("name", "gym", "gym_name"):
                maybe_name = obj.get(key)
                if isinstance(maybe_name, str) and maybe_name in valid_names:
                    found.add(maybe_name)
    return [name for name in ordered_gym_names() if name in found]


def _timestamp_to_date(value: Any) -> Optional[date]:
    ts = _safe_int(value, 0)
    if ts <= 0:
        return None
    try:
        return datetime.utcfromtimestamp(ts).date()
    except (OverflowError, OSError, ValueError):
        return None


def _expand_date_range(start_ts: Any, end_ts: Any) -> List[date]:
    start_date = _timestamp_to_date(start_ts)
    end_date = _timestamp_to_date(end_ts)
    if start_date is None and end_date is None:
        return []
    if start_date is None:
        start_date = end_date
    if end_date is None:
        end_date = start_date
    if start_date is None or end_date is None:
        return []
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    span_days = min((end_date - start_date).days, 31)
    return [start_date + timedelta(days=offset) for offset in range(span_days + 1)]


def _extract_war_days(*payloads: Dict[str, Any]) -> List[date]:
    found: set[date] = set()
    start_keys = ["start", "started", "start_timestamp", "war_start", "begin"]
    end_keys = ["end", "ended", "end_timestamp", "war_end", "finish"]
    today_utc = datetime.utcnow().date()
    future_limit = today_utc + timedelta(days=60)
    for payload in payloads:
        for obj in _walk_objects(payload):
            if not isinstance(obj, dict):
                continue
            start_value = next((obj.get(key) for key in start_keys if obj.get(key) is not None), None)
            end_value = next((obj.get(key) for key in end_keys if obj.get(key) is not None), None)
            for day in _expand_date_range(start_value, end_value):
                if today_utc <= day <= future_limit:
                    found.add(day)
    return sorted(found)


def _parse_profile(profile: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[int], str]:
    profile_root = profile
    if isinstance(profile.get("profile"), dict):
        profile_root = profile["profile"]
    elif isinstance(profile.get("player"), dict):
        profile_root = profile["player"]
    torn_name = _safe_str(_first_present(profile_root, [("name",), ("player_name",)], default=""))
    torn_id = _safe_int(_first_present(profile_root, [("player_id",), ("id",), ("player", "id")], default=0), 0) or None
    faction_blob = _first_present(profile_root, [("faction",)], default={})
    if not isinstance(faction_blob, dict):
        faction_blob = {}
    faction_id = _safe_int(_first_present(faction_blob, [("id",), ("faction_id",)], default=0), 0) or None
    faction_name = _safe_str(_first_present(faction_blob, [("name",), ("faction_name",)], default=""))
    return torn_name, torn_id, faction_id, faction_name


def _parse_battlestats(payload: Dict[str, Any]) -> PlayerStats:
    stats_root = payload
    for key in ("battlestats", "battle_stats", "stats"):
        if isinstance(payload.get(key), dict):
            stats_root = payload[key]
            break

    def get_stat(stat_name: str) -> float:
        blob = stats_root.get(stat_name)

        # Torn v2 shape:
        # "strength": {"value": 400108, "modifier": -26, ...}
        if isinstance(blob, dict):
            value = _safe_float(blob.get("value"), 0.0)
            if value > 0:
                return value

        direct = _first_present(
            stats_root,
            [
                (stat_name,),
                (stat_name, "value"),
                (f"{stat_name}_info", "value"),
            ],
            default=None,
        )
        if direct is not None:
            return _safe_float(direct, 0.0)

        recursive = _find_first_numeric_for_key(payload, stat_name)
        if recursive is not None:
            return float(recursive)

        return 0.0

    return PlayerStats(
        strength=get_stat("strength"),
        speed=get_stat("speed"),
        defense=get_stat("defense"),
        dexterity=get_stat("dexterity"),
    )

def _parse_recovery_state(bars: Dict[str, Any], cooldowns: Dict[str, Any]) -> RecoveryState:
    bars_root = bars.get("bars") if isinstance(bars.get("bars"), dict) else bars
    cooldowns_root = cooldowns.get("cooldowns") if isinstance(cooldowns.get("cooldowns"), dict) else cooldowns
    energy_blob = _first_present(bars_root, [("energy",)], default={})
    happy_blob = _first_present(bars_root, [("happy",)], default={})
    if not isinstance(energy_blob, dict):
        energy_blob = {}
    if not isinstance(happy_blob, dict):
        happy_blob = {}
    return RecoveryState(
        current_energy=_safe_int(_first_present(energy_blob, [("current",), ("amount",), ("value",)], default=0), 0),
        max_energy=_safe_int(_first_present(energy_blob, [("maximum",), ("max",)], default=150), 150),
        regen_per_tick=5,
        regen_minutes_per_tick=10,
        xanax_per_day=3,
        xanax_energy=250,
        daily_refill_enabled=True,
        refill_energy=150,
        drug_cd_minutes=_coerce_cooldown_minutes(_first_present(cooldowns_root, [("drug",)], default=0)),
        booster_cd_minutes=_coerce_cooldown_minutes(_first_present(cooldowns_root, [("booster",)], default=0)),
        current_happy=_safe_int(_first_present(happy_blob, [("current",), ("amount",), ("value",)], default=0), 0),
        max_happy=_safe_int(_first_present(happy_blob, [("maximum",), ("max",)], default=0), 0),
        donor=True,
    )


def _parse_education_modifiers(payload: Dict[str, Any]) -> TrainingModifiers:
    strings = " ".join(_extract_strings(payload)).upper()
    mods = TrainingModifiers()
    if "SPT3510" in strings:
        mods.all_gym_gains_pct += 1.0
        mods.detected_sources.append("Education SPT3510 (+1% all gym gains)")
    if "SPT2440" in strings:
        mods.strength_gym_gains_pct += 1.0
        mods.detected_sources.append("Education SPT2440 (+1% strength gym gains)")
    if "SPT2450" in strings:
        mods.speed_gym_gains_pct += 1.0
        mods.detected_sources.append("Education SPT2450 (+1% speed gym gains)")
    if "SPT2460" in strings:
        mods.defense_gym_gains_pct += 1.0
        mods.detected_sources.append("Education SPT2460 (+1% defense gym gains)")
    if "SPT2470" in strings:
        mods.dexterity_gym_gains_pct += 1.0
        mods.detected_sources.append("Education SPT2470 (+1% dexterity gym gains)")
    return mods


def _parse_job_modifiers(payload: Dict[str, Any]) -> TrainingModifiers:
    text = " ".join(_extract_strings(payload)).lower()
    mods = TrainingModifiers()
    if "fitness center" in text:
        mods.all_gym_gains_pct += 3.0
        mods.happy_loss_reduction_pct += 50.0
        mods.detected_sources.append("Fitness Center (+3% gym gains, -50% gym happy loss)")
    if "gents strip club" in text:
        mods.dexterity_gym_gains_pct += 10.0
        mods.detected_sources.append("Gents Strip Club (+10% dexterity gym gains)")
    if "ladies strip club" in text:
        mods.defense_gym_gains_pct += 10.0
        mods.detected_sources.append("Ladies Strip Club (+10% defense gym gains)")
    if "higher daddy, higher!" in text:
        mods.energy_regen_bonus_pct += 20.0
        mods.detected_sources.append("Higher Daddy, Higher! (+20% energy regeneration)")
    return mods


def _parse_property_modifiers(payload: Dict[str, Any]) -> TrainingModifiers:
    text = " ".join(_extract_strings(payload))
    mods = TrainingModifiers()
    if any(pool_name in text for pool_name in PROPERTY_POOL_NAMES):
        mods.all_gym_gains_pct += 2.0
        mods.detected_sources.append("Property pool (+2% gym gains)")
    return mods


def _extract_numeric_from_context(serialized: str, needle: str) -> Optional[float]:
    pattern = re.compile(rf"{re.escape(needle)}[^0-9]{{0,40}}(\d+(?:\.\d+)?)", re.IGNORECASE)
    match = pattern.search(serialized)
    if not match:
        return None
    return _safe_float(match.group(1), default=0.0)


def _parse_faction_upgrade_modifiers(payload: Dict[str, Any]) -> TrainingModifiers:
    serialized = str(payload)
    mods = TrainingModifiers()
    for label, attr in [
        ("Strength training", "strength_gym_gains_pct"),
        ("Speed training", "speed_gym_gains_pct"),
        ("Defense training", "defense_gym_gains_pct"),
        ("Dexterity training", "dexterity_gym_gains_pct"),
    ]:
        value = _extract_numeric_from_context(serialized, label)
        if value and value > 0:
            setattr(mods, attr, getattr(mods, attr) + value)
            mods.detected_sources.append(f"Faction {label} (+{value:.0f}%)")
    return mods


def build_demo_player_state() -> PlayerState:
    return PlayerState(
        torn_name="Demo Player",
        torn_id=123456,
        faction_id=654321,
        faction_name="Demo Faction",
        stats=PlayerStats(strength=12_000_000, speed=8_900_000, defense=7_200_000, dexterity=7_150_000),
        recovery=RecoveryState(
            current_energy=87,
            max_energy=150,
            regen_per_tick=5,
            regen_minutes_per_tick=10,
            xanax_per_day=3,
            xanax_energy=250,
            daily_refill_enabled=True,
            refill_energy=150,
            drug_cd_minutes=97,
            booster_cd_minutes=850,
            current_happy=4500,
            max_happy=5000,
            donor=True,
        ),
        unlocked_gyms=[
            "Premier Fitness", "Average Joes", "Woody's Workout", "Beach Bods", "Silver Gym", "Pour Femme",
            "Davies Den", "Global Gym", "Knuckle Heads", "Pioneer Fitness", "Anabolic Anomalies", "Core",
            "Racing Fitness", "Complete Cardio", "Legs, Bums and Tums", "Deep Burn", "Apollo Gym",
            "Gun Shop", "Force Training", "Cha Cha's", "Atlas", "Last Round", "The Edge",
        ],
        faction_war_days=[date.today() + timedelta(days=2)],
        training_modifiers=TrainingModifiers(
            all_gym_gains_pct=3.0,
            happy_loss_reduction_pct=50.0,
            detected_sources=["Demo Fitness Center (+3% gym gains, -50% gym happy loss)"],
        ),
        api_notes=["Demo data loaded.", "Unlocked gyms are demo values."],
        last_sync=datetime.now(),
    )


def fetch_player_state_from_api(api_key: str, manual_unlocked_gyms: Optional[List[str]] = None) -> PlayerState:
    profile = _api_get("/user/profile", api_key)
    bars = _api_get("/user/bars", api_key)
    cooldowns = _api_get("/user/cooldowns", api_key)
    battlestats = _api_get("/user/battlestats", api_key)

    api_notes: List[str] = []
    torn_name, torn_id, faction_id, faction_name = _parse_profile(profile)
    stats = _parse_battlestats(battlestats)
    recovery = _parse_recovery_state(bars, cooldowns)

    education_payload: Dict[str, Any] = {}
    job_payload: Dict[str, Any] = {}
    property_payload: Dict[str, Any] = {}
    faction_basic: Dict[str, Any] = {}
    faction_wars: Dict[str, Any] = {}
    faction_ranked_wars: Dict[str, Any] = {}
    faction_upgrades: Dict[str, Any] = {}

    for path_name, target in [
        ("/user/education", "education_payload"),
        ("/user/job", "job_payload"),
        ("/user/property", "property_payload"),
        ("/faction/basic", "faction_basic"),
        ("/faction/wars", "faction_wars"),
        ("/faction/rankedwars", "faction_ranked_wars"),
        ("/faction/upgrades", "faction_upgrades"),
    ]:
        try:
            payload = _api_get(path_name, api_key, params={"limit": 10} if path_name.endswith("rankedwars") else None)
            if target == "education_payload":
                education_payload = payload
            elif target == "job_payload":
                job_payload = payload
            elif target == "property_payload":
                property_payload = payload
            elif target == "faction_basic":
                faction_basic = payload
            elif target == "faction_wars":
                faction_wars = payload
            elif target == "faction_ranked_wars":
                faction_ranked_wars = payload
            elif target == "faction_upgrades":
                faction_upgrades = payload
        except TornAPIError as exc:
            api_notes.append(f"{path_name} skipped: {exc}")

    if not faction_name and isinstance(faction_basic, dict):
        faction_name = _safe_str(_first_present(faction_basic, [("name",), ("faction", "name")], default=""))
    if faction_id is None and isinstance(faction_basic, dict):
        faction_id = _safe_int(_first_present(faction_basic, [("id",), ("faction", "id")], default=0), 0) or None

    faction_war_days = _extract_war_days(faction_wars, faction_ranked_wars)
    if faction_war_days:
        api_notes.append(f"Loaded {len(faction_war_days)} upcoming war/non-training dates from faction endpoints.")
    else:
        api_notes.append("No upcoming war dates were detected from faction endpoints.")

    unlocked_gyms = _extract_gym_names_from_payloads([profile, bars, cooldowns, battlestats, faction_basic, education_payload, job_payload, property_payload])
    if not unlocked_gyms and manual_unlocked_gyms:
        unlocked_gyms = [name for name in ordered_gym_names() if name in set(manual_unlocked_gyms)]
        api_notes.append("Using manual unlocked gyms because API payloads did not expose them.")
    elif not unlocked_gyms:
        api_notes.append("Unlocked gyms were not exposed by the synced API payloads. Select them manually in the UI.")

    training_modifiers = TrainingModifiers()
    for parsed in [
        _parse_education_modifiers(education_payload),
        _parse_job_modifiers(job_payload),
        _parse_property_modifiers(property_payload),
        _parse_faction_upgrade_modifiers(faction_upgrades),
    ]:
        training_modifiers = training_modifiers.merge(parsed)

    if training_modifiers.detected_sources:
        api_notes.append("Detected training modifiers from API: " + "; ".join(training_modifiers.detected_sources))
    else:
        api_notes.append("No training modifiers were auto-detected. Use manual overrides if needed.")

    if stats.total() <= 0:
        api_notes.append("Battle stats came back as zero. This usually means the API key does not include limited access to /user/battlestats, or Torn returned a battlestats shape this parser still does not recognize.")

    return PlayerState(
        torn_name=torn_name,
        torn_id=torn_id,
        faction_id=faction_id,
        faction_name=faction_name,
        stats=stats,
        recovery=recovery,
        unlocked_gyms=unlocked_gyms,
        faction_war_days=faction_war_days,
        training_modifiers=training_modifiers,
        api_notes=api_notes,
        last_sync=datetime.now(),
    )


def highest_unlocked_gym_index(state: PlayerState) -> int:
    indices = [ordered_gym_names().index(name) for name in state.unlocked_gyms if name in GYM_INDEX]
    return max(indices) if indices else 0


def unlocked_names_through_index(highest_idx: int) -> List[str]:
    names = ordered_gym_names()
    highest_idx = max(0, min(highest_idx, len(names) - 1))
    return names[: highest_idx + 1]


def next_gym_name_for_index(highest_idx: int) -> Optional[str]:
    names = ordered_gym_names()
    if highest_idx + 1 < len(names):
        return names[highest_idx + 1]
    return None


def next_gym_threshold_for_index(highest_idx: int) -> Optional[int]:
    if 0 <= highest_idx < len(GYM_DB):
        return GYM_DB[highest_idx].e_for_next_gym
    return None


def best_gym_for_stat_from_names(unlocked_names: List[str], stat_key: str) -> Optional[Gym]:
    candidates = [GYM_INDEX[name] for name in unlocked_names if name in GYM_INDEX and GYM_INDEX[name].gain_for(stat_key) > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda gym: (gym.gain_for(stat_key), GYM_DB.index(gym)), reverse=True)
    return candidates[0]


def estimate_segment_end_time(start_dt: datetime, segment_energy: int, total_day_energy: int) -> datetime:
    if total_day_energy <= 0:
        return start_dt
    fraction = segment_energy / total_day_energy
    minutes = max(1, int(round(fraction * 16 * 60)))
    return start_dt + timedelta(minutes=minutes)


def end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def natural_energy_between(state: PlayerState, mods: TrainingModifiers, start_dt: datetime, end_dt: datetime) -> int:
    if end_dt <= start_dt:
        return 0
    minutes = (end_dt - start_dt).total_seconds() / 60.0
    if minutes <= 0:
        return 0
    daily_natural = state.recovery.natural_energy_per_day(energy_regen_bonus_pct=mods.energy_regen_bonus_pct)
    return max(0, int(minutes / 1440.0 * daily_natural))


def build_today_energy_blocks(state: PlayerState, goal: GoalSettings, mods: TrainingModifiers, now_dt: Optional[datetime] = None) -> List[Tuple[datetime, int, str]]:
    now_dt = now_dt or datetime.now()
    blocks: List[Tuple[datetime, int, str]] = []

    if state.recovery.current_energy > 0:
        blocks.append((now_dt, int(state.recovery.current_energy), 'current energy'))

    if state.recovery.daily_refill_enabled:
        refill_time = now_dt + timedelta(minutes=10)
        blocks.append((refill_time, int(state.recovery.refill_energy), 'daily refill'))

    eod = end_of_day(now_dt)
    natural_e = natural_energy_between(state, mods, now_dt, eod)
    if natural_e > 0:
        blocks.append((eod, natural_e, 'natural regen through end of day'))

    drug_clear_dt = now_dt + timedelta(minutes=max(0, state.recovery.drug_cd_minutes))
    if drug_clear_dt.date() == now_dt.date() and state.recovery.drug_cd_minutes > 0:
        blocks.append((drug_clear_dt, int(state.recovery.xanax_energy), 'next xanax after cooldown'))

    return blocks


def estimate_today_unlock_from_blocks(state: PlayerState, goal: GoalSettings, mods: TrainingModifiers, now_dt: Optional[datetime] = None) -> Optional[Tuple[datetime, str]]:
    now_dt = now_dt or datetime.now()
    highest_idx = highest_unlocked_gym_index(state)
    next_gym = next_gym_name_for_index(highest_idx)
    threshold = next_gym_threshold_for_index(highest_idx)
    progress = max(0, int(goal.current_gym_energy_progress))
    if next_gym is None or threshold is None:
        return None

    remaining = threshold - progress
    if remaining <= 0:
        return (now_dt, next_gym)

    blocks = sorted(build_today_energy_blocks(state, goal, mods, now_dt), key=lambda x: x[0])
    accumulated = 0
    for when, energy, _source in blocks:
        accumulated += energy
        if accumulated >= remaining:
            return (when, next_gym)
    return None


def simulate_day_with_unlocks(
    state: PlayerState,
    ratio: RatioProfile,
    goal: GoalSettings,
    plan_day: date,
    manual_mods: TrainingModifiers,
    highest_idx: int,
    progress_e: int,
) -> Tuple[DailyInstruction, PlayerStats, int, int, Optional[datetime]]:
    combined_mods = state.training_modifiers.merge(manual_mods)
    projected_unlocked = unlocked_names_through_index(highest_idx)
    projected_state = PlayerState(
        stats=state.stats,
        recovery=state.recovery,
        unlocked_gyms=projected_unlocked,
        faction_war_days=list(state.faction_war_days),
        torn_name=state.torn_name,
        torn_id=state.torn_id,
        faction_id=state.faction_id,
        faction_name=state.faction_name,
        training_modifiers=state.training_modifiers,
        api_notes=list(state.api_notes),
        last_sync=state.last_sync,
    )
    day_type, jump_plan = day_type_for_date(projected_state, ratio, goal, plan_day, combined_mods)
    target_stat = choose_target_stat(projected_state.stats, ratio)

    if day_type == 'war':
        instruction = DailyInstruction(plan_day, 'war', 'none', 'none', 0, 0, 0.0, 0, 0, ['Faction war day. Training skipped in baseline planner.'])
        return instruction, state.stats, highest_idx, progress_e, None

    day_energy = energy_budget_for_day(projected_state, goal, plan_day, combined_mods, day_type)
    if day_energy <= 0:
        instruction = DailyInstruction(plan_day, day_type, target_stat, 'none', 0, 0, 0.0, 0, 0, ['No training energy budget for this day.'])
        return instruction, state.stats, highest_idx, progress_e, None

    start_happy = projected_start_happy_for_day(projected_state, goal, plan_day, day_type)
    current_stats = state.stats
    current_happy = start_happy
    total_gain = 0.0
    total_trains = 0
    energy_left = day_energy
    used_gyms: List[str] = []
    unlock_notes: List[str] = []
    unlock_time: Optional[datetime] = None
    time_cursor = datetime.combine(plan_day, dtime(hour=8, minute=0))
    if jump_plan is not None and plan_day == jump_plan.execute_at.date():
        time_cursor = jump_plan.execute_at

    while energy_left > 0:
        unlocked_names = unlocked_names_through_index(highest_idx)
        gym = best_gym_for_stat_from_names(unlocked_names, target_stat)
        if gym is None:
            break
        if gym.name not in used_gyms:
            used_gyms.append(gym.name)

        threshold = next_gym_threshold_for_index(highest_idx)
        segment_energy = energy_left
        if threshold is not None:
            remaining_to_unlock = max(0, threshold - progress_e)
            if remaining_to_unlock == 0 and next_gym_name_for_index(highest_idx) is not None:
                unlock_notes.append(f'Projected unlock at start of day: {next_gym_name_for_index(highest_idx)}.')
                highest_idx += 1
                progress_e = 0
                continue
            if 0 < remaining_to_unlock < energy_left:
                segment_energy = remaining_to_unlock

        sim = simulate_training_block(current_stats, target_stat, gym, segment_energy, current_happy, combined_mods)
        total_gain += float(sim['total_gain'])
        total_trains += int(sim['trains'])
        current_stats = current_stats.with_gain(target_stat, float(sim['total_gain']))
        current_happy = int(sim['ending_happy'])
        energy_left -= segment_energy

        if threshold is not None:
            progress_e += segment_energy
            if progress_e >= threshold and next_gym_name_for_index(highest_idx) is not None:
                overflow = progress_e - threshold
                unlock_name = next_gym_name_for_index(highest_idx)
                unlock_time = estimate_segment_end_time(time_cursor, segment_energy, day_energy)
                unlock_notes.append(f'Projected unlock during day: {unlock_name} at about {unlock_time.strftime("%H:%M")}. Remaining energy shifts immediately to the better available gym.')
                highest_idx += 1
                progress_e = overflow
                time_cursor = unlock_time
                continue
        time_cursor = estimate_segment_end_time(time_cursor, segment_energy, day_energy)

    gym_name = ' → '.join(used_gyms) if used_gyms else 'unknown'
    notes = [
        f'Train {target_stat.title()} in {gym_name}.',
        f'Phase: {milestone_phase(current_stats)}.',
        f'Expected happy loss per train: {expected_happy_loss_per_train((best_gym_for_stat_from_names(unlocked_names_through_index(highest_idx), target_stat) or GYM_DB[0]).energy_cost, combined_mods)}.',
    ]
    if jump_plan is not None:
        notes.append(f'Next jump window: {jump_plan.execute_at.strftime("%Y-%m-%d %H:%M")}.')
    if day_type == 'prep':
        notes.append('Prep day: do not assume more drug uses are possible until cooldown and stack timing allow them.')
    elif day_type == 'happy_jump':
        notes.append('Happy jump day: follow the timed action planner below.')
    elif day_type == 'super_happy_jump':
        notes.append('99k jump day: this uses your manually scheduled jump time.')
    else:
        notes.append('Normal training day.')
    notes.extend(unlock_notes)
    if combined_mods.detected_sources:
        notes.append('Detected modifiers: ' + '; '.join(combined_mods.detected_sources[:4]))

    instruction = DailyInstruction(plan_day, day_type, target_stat, gym_name, day_energy, total_trains, total_gain, start_happy, current_happy, notes)
    return instruction, current_stats, highest_idx, progress_e, unlock_time


def calculate_current_ratio(stats: PlayerStats) -> Dict[str, float]:
    total = max(stats.total(), 1.0)
    return {k: v / total * 100 for k, v in stats.as_dict().items()}


def milestone_phase(stats: PlayerStats) -> str:
    lowest_stat = min(stats.as_dict().values())
    if lowest_stat < 400_000:
        return "all_to_400k"
    if lowest_stat < 600_000:
        return "all_to_600k"
    if lowest_stat < 800_000:
        return "all_to_800k"
    return "baldr_ratio"



def current_milestone_cap(stats: PlayerStats) -> Optional[int]:
    phase = milestone_phase(stats)
    if phase == "all_to_400k":
        return 400_000
    if phase == "all_to_600k":
        return 600_000
    if phase == "all_to_800k":
        return 800_000
    return None



def choose_target_stat(stats: PlayerStats, ratio: RatioProfile) -> str:
    phase = milestone_phase(stats)
    stat_map = stats.as_dict()
    if phase != "baldr_ratio":
        cap = current_milestone_cap(stats)
        eligible = {k: v for k, v in stat_map.items() if cap is not None and v < cap}
        if eligible:
            # During milestone phases, finish the stat closest to the cap first.
            return max(eligible, key=eligible.get)
        return min(stat_map, key=stat_map.get)

    current_ratio = calculate_current_ratio(stats)
    target_ratio = ratio.as_percent_map()
    deficits = {stat_key: target_ratio[stat_key] - current_ratio[stat_key] for stat_key in STAT_KEYS}
    return max(deficits, key=deficits.get)



def get_unlocked_gyms(state: PlayerState) -> List[Gym]:
    return [GYM_INDEX[name] for name in state.unlocked_gyms if name in GYM_INDEX]


def best_gym_for_stat(state: PlayerState, stat_key: str) -> Optional[Gym]:
    candidates = [gym for gym in get_unlocked_gyms(state) if gym.gain_for(stat_key) > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda gym: (gym.gain_for(stat_key), GYM_DB.index(gym)), reverse=True)
    return candidates[0]


def expected_happy_loss_per_train(energy_per_train: int, mods: TrainingModifiers) -> int:
    raw_loss = energy_per_train * DEFAULT_EXPECTED_HAPPY_LOSS_RATIO
    reduced_loss = raw_loss * max(0.0, 1.0 - mods.happy_loss_reduction_pct / 100.0)
    return max(0, int(round(reduced_loss)))


def single_train_gain(stat_total: float, happy: float, gym_dots: float, energy_per_train: int, stat_modifier_multiplier: float) -> float:
    safe_happy = max(0.0, happy)
    inside = ((FORMULA_A * math.log(safe_happy + FORMULA_B) + FORMULA_C) * stat_total) + (FORMULA_D * (safe_happy + FORMULA_B)) + FORMULA_E
    base_gain = gym_dots * energy_per_train * inside
    return max(0.0, base_gain * stat_modifier_multiplier)


def simulate_training_block(
    base_stats: PlayerStats,
    stat_key: str,
    gym: Gym,
    total_energy: int,
    starting_happy: int,
    mods: TrainingModifiers,
) -> Dict[str, Any]:
    if total_energy <= 0:
        return {"total_gain": 0.0, "ending_happy": starting_happy, "trains": 0, "ending_stat": base_stats.get(stat_key), "per_train_preview": []}
    trains = total_energy // gym.energy_cost
    current_stat = float(base_stats.get(stat_key))
    current_happy = int(starting_happy)
    stat_multiplier = mods.gym_multiplier_for_stat(stat_key)
    happy_loss = expected_happy_loss_per_train(gym.energy_cost, mods)
    total_gain = 0.0
    preview_rows: List[Dict[str, float]] = []
    for train_number in range(1, int(trains) + 1):
        gain = single_train_gain(
            stat_total=current_stat,
            happy=current_happy,
            gym_dots=gym.gain_for(stat_key),
            energy_per_train=gym.energy_cost,
            stat_modifier_multiplier=stat_multiplier,
        )
        total_gain += gain
        current_stat += gain
        if train_number <= 5:
            preview_rows.append({"train": train_number, "happy_before": float(current_happy), "gain": gain})
        current_happy = max(0, current_happy - happy_loss)
    return {"total_gain": total_gain, "ending_happy": current_happy, "trains": int(trains), "ending_stat": current_stat, "per_train_preview": preview_rows}


def next_quarter_hour(dt: datetime) -> datetime:
    minute_block = ((dt.minute // 15) + 1) * 15
    if minute_block >= 60:
        base = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        base = dt.replace(minute=minute_block, second=0, microsecond=0)
    return base + timedelta(seconds=5)


def projected_start_happy_for_day(state: PlayerState, goal: GoalSettings, plan_day: date, day_type: str) -> int:
    if day_type == "happy_jump":
        return goal.happy_jump_start_happy
    if day_type == "super_happy_jump":
        return goal.super_happy_jump_start_happy
    if plan_day == date.today() and state.recovery.current_happy > 0:
        return state.recovery.current_happy
    if state.recovery.max_happy > 0:
        return state.recovery.max_happy
    return goal.normal_day_start_happy


def selected_99k_execute_at(goal: GoalSettings) -> Optional[datetime]:
    if not goal.schedule_99k_jump:
        return None
    return datetime.combine(goal.scheduled_99k_jump_date, goal.scheduled_99k_jump_time)


def next_viable_happy_jump_window(state: PlayerState, goal: GoalSettings) -> datetime:
    now_dt = datetime.now()
    prep_ready = now_dt + timedelta(minutes=max(state.recovery.drug_cd_minutes, state.recovery.booster_cd_minutes))
    candidate = next_quarter_hour(prep_ready + timedelta(hours=goal.jump_prep_hours))
    while (not goal.allow_jump_on_war_days) and (candidate.date() in state.faction_war_days):
        candidate = next_quarter_hour(candidate + timedelta(days=1))
    return candidate


def planned_xanax_stack_times(state: PlayerState, goal: GoalSettings, execute_at: datetime) -> List[datetime]:
    first_candidate = execute_at - timedelta(hours=goal.jump_prep_hours)
    earliest_allowed = datetime.now() + timedelta(minutes=max(0, state.recovery.drug_cd_minutes))
    first = max(first_candidate, earliest_allowed)
    times = [first]
    for _ in range(1, int(goal.jump_stack_xanax_uses)):
        times.append(times[-1] + timedelta(hours=float(goal.assumed_xanax_cooldown_hours)))
    return times


def build_jump_plan(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, mods: TrainingModifiers) -> Optional[JumpPlan]:
    target_stat = choose_target_stat(state.stats, ratio)
    gym = best_gym_for_stat(state, target_stat)
    if gym is None:
        return None

    normal_energy = state.recovery.baseline_energy_per_day(energy_regen_bonus_pct=mods.energy_regen_bonus_pct)
    normal_start_happy = max(state.recovery.max_happy, goal.normal_day_start_happy)
    normal_sim = simulate_training_block(state.stats, target_stat, gym, normal_energy, normal_start_happy, mods)

    jump_energy = goal.jump_stack_energy_target + (state.recovery.refill_energy if state.recovery.daily_refill_enabled else 0)
    happy_sim = simulate_training_block(state.stats, target_stat, gym, jump_energy, goal.happy_jump_start_happy, mods)
    super_sim = simulate_training_block(state.stats, target_stat, gym, jump_energy, goal.super_happy_jump_start_happy, mods)
    threshold = 1 + goal.jump_min_extra_gain_pct / 100.0

    execute_at: Optional[datetime] = None
    chosen_type: Optional[str] = None
    chosen_gain = 0.0
    notes: List[str] = []

    manual_99k_dt = selected_99k_execute_at(goal)
    if manual_99k_dt is not None:
        chosen_type = "super_happy_jump"
        chosen_gain = float(super_sim["total_gain"])
        execute_at = manual_99k_dt
        notes.append("99k jump date was manually selected by you.")
    elif goal.auto_schedule_happy_jumps and happy_sim["total_gain"] > normal_sim["total_gain"] * threshold:
        chosen_type = "happy_jump"
        chosen_gain = float(happy_sim["total_gain"])
        execute_at = next_viable_happy_jump_window(state, goal)

    if chosen_type is None or execute_at is None:
        return None

    prep_start = execute_at - timedelta(hours=goal.jump_prep_hours)
    xanax_times = planned_xanax_stack_times(state, goal, execute_at)
    final_cd_clear = xanax_times[-1] + timedelta(hours=float(goal.assumed_xanax_cooldown_hours))
    if final_cd_clear > execute_at:
        notes.append("Warning: current drug cooldown and stack timing push the final Xanax cooldown past the planned jump window.")
    if state.recovery.booster_cd_minutes > 0 and datetime.now() + timedelta(minutes=state.recovery.booster_cd_minutes) > execute_at:
        notes.append("Warning: booster cooldown is still active beyond the planned jump window.")
    if execute_at.date() in state.faction_war_days and not goal.allow_jump_on_war_days:
        notes.append("Warning: this planned jump falls on a war day.")

    notes.extend([
        f"Planner reserves about {goal.jump_prep_hours:.0f} hours of prep time for the jump.",
        f"Planner assumes {goal.jump_stack_xanax_uses} Xanax with roughly {goal.assumed_xanax_cooldown_hours:.1f} hours between doses.",
        "Use the jump just after a 15-minute happy reset mark.",
        "Train immediately after applying happy items / ecstasy, then use daily refill.",
    ])

    return JumpPlan(
        jump_type=chosen_type,
        target_stat=target_stat,
        gym_name=gym.name,
        prep_start=prep_start,
        execute_at=execute_at,
        projected_normal_gain=float(normal_sim["total_gain"]),
        projected_jump_gain=chosen_gain,
        projected_gain_delta=float(chosen_gain - normal_sim["total_gain"]),
        notes=notes,
    )


def build_jump_sequence(state: PlayerState, goal: GoalSettings, jump_plan: JumpPlan) -> List[JumpStep]:
    steps: List[JumpStep] = []
    xanax_times = planned_xanax_stack_times(state, goal, jump_plan.execute_at)

    if state.recovery.current_energy > 0 and datetime.now().date() <= jump_plan.execute_at.date():
        steps.append(JumpStep(datetime.now(), "Spend current energy", f"Use your current {state.recovery.current_energy} energy now unless you are already in the final save-for-jump window."))

    for idx, use_time in enumerate(xanax_times, start=1):
        steps.append(JumpStep(use_time, f"Take Xanax #{idx}", "Take the dose as soon as drug cooldown clears, then let the next cooldown run."))

    ready_time = jump_plan.execute_at - timedelta(minutes=2)
    steps.append(JumpStep(ready_time, "Be ready in gym", f"Open {jump_plan.gym_name} and have all items ready before the quarter-hour mark."))

    if jump_plan.jump_type == "happy_jump":
        steps.append(JumpStep(jump_plan.execute_at, "Use happy items", "Use your standard happy jump item set right after the quarter-hour reset."))
        steps.append(JumpStep(jump_plan.execute_at + timedelta(seconds=10), "Use Ecstasy", "Use Ecstasy immediately after happy items."))
    else:
        steps.append(JumpStep(jump_plan.execute_at, "Execute 99k setup", "Use your planned 99k method/service on the selected day, then start training immediately."))

    steps.append(JumpStep(jump_plan.execute_at + timedelta(seconds=20), "Train stacked energy", f"Train all stacked energy into {jump_plan.target_stat.title()} at {jump_plan.gym_name}."))

    if state.recovery.daily_refill_enabled:
        steps.append(JumpStep(jump_plan.execute_at + timedelta(minutes=1), "Use daily refill", "Use daily refill as soon as the stacked energy is spent."))
        steps.append(JumpStep(jump_plan.execute_at + timedelta(minutes=1, seconds=10), "Train refill energy", f"Train refill energy into {jump_plan.target_stat.title()} at {jump_plan.gym_name}."))

    if jump_plan.jump_type == "super_happy_jump" and (goal.fhc_allowed or goal.cans_allowed):
        fill = []
        if goal.fhc_allowed:
            fill.append("FHC")
        if goal.cans_allowed:
            fill.append("cans")
        steps.append(JumpStep(jump_plan.execute_at + timedelta(minutes=2), "Optional filler energy", f"Use approved {' and '.join(fill)} only if you want to push extra energy after the refill block."))

    return sorted(steps, key=lambda x: x.when)


def day_type_for_date(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, on_day: date, mods: TrainingModifiers) -> Tuple[str, Optional[JumpPlan]]:
    jump_plan = build_jump_plan(state, ratio, goal, mods)
    if jump_plan is not None:
        if on_day == jump_plan.execute_at.date():
            return jump_plan.jump_type, jump_plan
        if jump_plan.prep_start.date() <= on_day < jump_plan.execute_at.date():
            return "prep", jump_plan

    if on_day in state.faction_war_days and not goal.allow_jump_on_war_days:
        return "war", jump_plan

    return "normal", jump_plan


def energy_budget_for_day(state: PlayerState, goal: GoalSettings, plan_day: date, mods: TrainingModifiers, day_type: str) -> int:
    if goal.skip_war_days and plan_day in state.faction_war_days and day_type not in {"happy_jump", "super_happy_jump"}:
        return 0

    # For today, use only energy that is actually reachable from the current
    # moment forward. This prevents the preview from assuming all 3 Xanax are
    # available before their cooldowns clear.
    if plan_day == date.today() and day_type not in {"happy_jump", "super_happy_jump"}:
        blocks = build_today_energy_blocks(state, goal, mods, datetime.now())
        if day_type == "prep":
            # Prep day should remain conservative: only current energy plus any
            # natural regen/refill that is reachable today, but no extra jump stack.
            return max(0, sum(energy for _when, energy, _source in blocks))
        return max(0, sum(energy for _when, energy, _source in blocks))

    if day_type == "prep":
        return min(state.recovery.max_energy, max(state.recovery.current_energy, 0))
    if day_type in {"happy_jump", "super_happy_jump"}:
        return goal.jump_stack_energy_target + (state.recovery.refill_energy if state.recovery.daily_refill_enabled else 0)
    return state.recovery.baseline_energy_per_day(energy_regen_bonus_pct=mods.energy_regen_bonus_pct)


def build_daily_instruction(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, plan_day: date, manual_mods: TrainingModifiers) -> DailyInstruction:
    combined_mods = state.training_modifiers.merge(manual_mods)
    day_type, jump_plan = day_type_for_date(state, ratio, goal, plan_day, combined_mods)
    target_stat = choose_target_stat(state.stats, ratio)
    gym = best_gym_for_stat(state, target_stat)

    if day_type == "war":
        return DailyInstruction(plan_day, "war", "none", "none", 0, 0, 0.0, 0, 0, ["Faction war day. Training skipped in baseline planner."])
    if gym is None:
        return DailyInstruction(plan_day, "blocked", target_stat, "unknown", 0, 0, 0.0, 0, 0, ["No unlocked gym found for target stat. Set your unlocked gyms in the UI."])

    day_energy = energy_budget_for_day(state, goal, plan_day, combined_mods, day_type)
    start_happy = projected_start_happy_for_day(state, goal, plan_day, day_type)
    sim = simulate_training_block(state.stats, target_stat, gym, day_energy, start_happy, combined_mods)

    notes = [
        f"Train {target_stat.title()} in {gym.name}.",
        f"Chosen because {gym.name} has the highest unlocked {target_stat} multiplier ({gym.gain_for(target_stat):.1f}).",
        f"Phase: {milestone_phase(state.stats)}.",
        f"Expected happy loss per train: {expected_happy_loss_per_train(gym.energy_cost, combined_mods)}.",
        f"Effective gym gain multiplier for {target_stat}: {combined_mods.gym_multiplier_for_stat(target_stat):.3f}x.",
    ]
    if jump_plan is not None:
        notes.append(f"Next jump window: {jump_plan.execute_at.strftime('%Y-%m-%d %H:%M')}.")
    if day_type == "prep":
        notes.append("Prep day: do not assume more drug uses are possible until cooldown and stack timing allow them.")
    elif day_type == "happy_jump":
        notes.append("Happy jump day: follow the timed action planner below.")
    elif day_type == "super_happy_jump":
        notes.append("99k jump day: this uses your manually scheduled date/time.")
    else:
        notes.append("Normal training day.")
    if combined_mods.detected_sources:
        notes.append("Detected modifiers: " + "; ".join(combined_mods.detected_sources[:4]))

    return DailyInstruction(plan_day, day_type, target_stat, gym.name, day_energy, sim["trains"], sim["total_gain"], start_happy, int(sim["ending_happy"]), notes)


def build_plan_preview(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers, days: int = 14) -> List[DailyInstruction]:
    plan: List[DailyInstruction] = []
    projected_stats = state.stats
    highest_idx = highest_unlocked_gym_index(state)
    progress_e = max(0, int(goal.current_gym_energy_progress))

    for offset in range(days):
        plan_day = date.today() + timedelta(days=offset)
        projected_state = PlayerState(
            stats=projected_stats,
            recovery=state.recovery,
            unlocked_gyms=unlocked_names_through_index(highest_idx),
            faction_war_days=list(state.faction_war_days),
            torn_name=state.torn_name,
            torn_id=state.torn_id,
            faction_id=state.faction_id,
            faction_name=state.faction_name,
            training_modifiers=state.training_modifiers,
            api_notes=list(state.api_notes),
            last_sync=state.last_sync,
        )
        instruction, projected_stats, highest_idx, progress_e, _unlock_time = simulate_day_with_unlocks(
            projected_state, ratio, goal, plan_day, manual_mods, highest_idx, progress_e
        )
        plan.append(instruction)

    return plan


def days_until_goal_estimate(state: PlayerState, goal: GoalSettings, manual_mods: TrainingModifiers) -> Optional[int]:
    remaining = goal.target_total_stats - state.stats.total()
    if remaining <= 0:
        return 0

    target_stat = choose_target_stat(state.stats, RatioProfile())
    gym = best_gym_for_stat(state, target_stat)
    if gym is None:
        return None

    combined_mods = state.training_modifiers.merge(manual_mods)
    start_happy = max(state.recovery.max_happy, goal.normal_day_start_happy)
    baseline_energy = state.recovery.baseline_energy_per_day(energy_regen_bonus_pct=combined_mods.energy_regen_bonus_pct)
    sim = simulate_training_block(state.stats, target_stat, gym, baseline_energy, start_happy, combined_mods)
    est_daily_gain = float(sim["total_gain"])

    jump_plan = build_jump_plan(state, RatioProfile(), goal, combined_mods)
    if jump_plan is not None:
        est_daily_gain += max(0.0, jump_plan.projected_gain_delta) / 7.0

    if est_daily_gain <= 0:
        return None

    return math.ceil(remaining / est_daily_gain)


def init_state() -> None:
    if "player_state" not in st.session_state:
        st.session_state.player_state = None
    if "goal_settings" not in st.session_state:
        st.session_state.goal_settings = GoalSettings()
    if "ratio_profile" not in st.session_state:
        st.session_state.ratio_profile = RatioProfile()
    if "manual_unlocked_gyms" not in st.session_state:
        st.session_state.manual_unlocked_gyms = []
    if "gym_multiselect" not in st.session_state:
        st.session_state.gym_multiselect = []
    if "highest_unlocked_gym_selector" not in st.session_state:
        st.session_state.highest_unlocked_gym_selector = "-- none --"
    if "manual_99k_jump_date" not in st.session_state:
        st.session_state.manual_99k_jump_date = date.today() + timedelta(days=7)


def render_sidebar() -> Tuple[str, int]:
    st.sidebar.header("Connection")
    api_key = st.sidebar.text_input("Torn API key", type="password")
    preview_days = st.sidebar.slider("Preview days", min_value=7, max_value=90, value=30, step=1)

    st.sidebar.header("Planner assumptions")
    st.sidebar.caption("These match the locked v2 rules.")
    st.sidebar.write("- 3 Xanax/day")
    st.sidebar.write("- Daily refill used")
    st.sidebar.write("- Booster cooldown tracked")
    st.sidebar.write("- Auto jump scheduling enabled")
    st.sidebar.write("- War days can block training")
    st.sidebar.write("- Happy loss modeled with expected value")

    return api_key, preview_days


def estimate_next_gym_unlock(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers, days: int = 90) -> GymUnlockProjection:
    highest_idx = highest_unlocked_gym_index(state)
    current_gym = ordered_gym_names()[highest_idx]
    next_gym = next_gym_name_for_index(highest_idx)
    threshold = next_gym_threshold_for_index(highest_idx)
    progress_e = max(0, int(goal.current_gym_energy_progress))
    if next_gym is None or threshold is None:
        return GymUnlockProjection(current_gym=current_gym, next_gym=None, current_progress=progress_e, required_progress=None, remaining_energy=None, estimated_unlock_at=None)

    projected_stats = state.stats
    projected_highest_idx = highest_idx
    projected_progress = progress_e
    for offset in range(days):
        plan_day = date.today() + timedelta(days=offset)
        projected_state = PlayerState(
            stats=projected_stats,
            recovery=state.recovery,
            unlocked_gyms=unlocked_names_through_index(projected_highest_idx),
            faction_war_days=list(state.faction_war_days),
            torn_name=state.torn_name,
            torn_id=state.torn_id,
            faction_id=state.faction_id,
            faction_name=state.faction_name,
            training_modifiers=state.training_modifiers,
            api_notes=list(state.api_notes),
            last_sync=state.last_sync,
        )
        _instruction, projected_stats, new_highest_idx, projected_progress, unlock_time = simulate_day_with_unlocks(
            projected_state, ratio, goal, plan_day, manual_mods, projected_highest_idx, projected_progress
        )
        if new_highest_idx > projected_highest_idx:
            return GymUnlockProjection(
                current_gym=current_gym,
                next_gym=next_gym,
                current_progress=progress_e,
                required_progress=threshold,
                remaining_energy=max(0, threshold - progress_e),
                estimated_unlock_at=unlock_time,
            )
        projected_highest_idx = new_highest_idx

    return GymUnlockProjection(
        current_gym=current_gym,
        next_gym=next_gym,
        current_progress=progress_e,
        required_progress=threshold,
        remaining_energy=max(0, threshold - progress_e),
        estimated_unlock_at=None,
    )


def render_goal_controls(goal: GoalSettings) -> GoalSettings:
    st.subheader("Goal setup")
    c1, c2 = st.columns(2)
    with c1:
        target_total = st.number_input("Target total battle stats", min_value=1_000_000, value=int(goal.target_total_stats), step=1_000_000)
    with c2:
        target_date = st.date_input("Target date", value=goal.target_date)

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        fhc_allowed = st.checkbox("FHC allowed", value=goal.fhc_allowed)
    with c4:
        cans_allowed = st.checkbox("Cans allowed", value=goal.cans_allowed)
    with c5:
        auto_schedule_happy_jumps = st.checkbox("Auto-schedule happy jumps", value=goal.auto_schedule_happy_jumps)
    with c6:
        schedule_99k_jump = st.checkbox("Schedule 99k jump", value=goal.schedule_99k_jump)

    skip_war_days = st.checkbox("Skip war days", value=goal.skip_war_days)

    c7, c8, c9 = st.columns(3)
    with c7:
        normal_day_start_happy = st.number_input("Normal day happy", min_value=0, value=int(goal.normal_day_start_happy), step=100)
    with c8:
        happy_jump_start_happy = st.number_input("Happy jump happy", min_value=0, value=int(goal.happy_jump_start_happy), step=1000)
    with c9:
        super_happy_jump_start_happy = st.number_input("99k jump happy", min_value=0, value=int(goal.super_happy_jump_start_happy), step=1000)

    c10, c11, c12 = st.columns(3)
    with c10:
        jump_stack_energy_target = st.number_input("Jump stack energy", min_value=150, value=int(goal.jump_stack_energy_target), step=50)
    with c11:
        jump_stack_xanax_uses = st.number_input("Jump stack Xanax uses", min_value=1, value=int(goal.jump_stack_xanax_uses), step=1)
    with c12:
        allow_jump_on_war_days = st.checkbox("Allow jumps on war days", value=goal.allow_jump_on_war_days)

    c13, c14, c15 = st.columns(3)
    with c13:
        jump_min_extra_gain_pct = st.number_input("Minimum extra gain % to prefer a jump", min_value=0.0, value=float(goal.jump_min_extra_gain_pct), step=1.0)
    with c14:
        jump_prep_hours = st.number_input("Jump prep hours", min_value=1.0, value=float(goal.jump_prep_hours), step=1.0)
    with c15:
        assumed_xanax_cooldown_hours = st.number_input("Assumed Xanax cooldown hours", min_value=1.0, value=float(goal.assumed_xanax_cooldown_hours), step=0.5)

    scheduled_99k_jump_date = goal.scheduled_99k_jump_date
    scheduled_99k_jump_time = goal.scheduled_99k_jump_time
    if schedule_99k_jump:
        c16, c17 = st.columns(2)
        with c16:
            scheduled_99k_jump_date = st.date_input("Planned 99k jump date", value=goal.scheduled_99k_jump_date, key="manual_99k_jump_date")
        with c17:
            scheduled_99k_jump_time = st.time_input("Planned 99k jump time", value=goal.scheduled_99k_jump_time, step=timedelta(minutes=15))

    current_gym_energy_progress = st.number_input(
        "Current estimated gym energy progress toward next unlock",
        min_value=0,
        value=int(goal.current_gym_energy_progress),
        step=10,
        help="This progress is not exposed by the Torn API. Enter your estimated current progress toward the next gym unlock, like the Torntools browser extension shows.",
    )

    return GoalSettings(
        target_total_stats=float(target_total),
        target_date=target_date,
        fhc_allowed=fhc_allowed,
        cans_allowed=cans_allowed,
        auto_schedule_happy_jumps=auto_schedule_happy_jumps,
        schedule_99k_jump=schedule_99k_jump,
        scheduled_99k_jump_date=scheduled_99k_jump_date,
        scheduled_99k_jump_time=scheduled_99k_jump_time,
        current_gym_energy_progress=int(current_gym_energy_progress),
        skip_war_days=skip_war_days,
        normal_day_start_happy=int(normal_day_start_happy),
        happy_jump_start_happy=int(happy_jump_start_happy),
        super_happy_jump_start_happy=int(super_happy_jump_start_happy),
        jump_stack_energy_target=int(jump_stack_energy_target),
        jump_stack_xanax_uses=int(jump_stack_xanax_uses),
        allow_jump_on_war_days=allow_jump_on_war_days,
        jump_min_extra_gain_pct=float(jump_min_extra_gain_pct),
        jump_prep_hours=float(jump_prep_hours),
        assumed_xanax_cooldown_hours=float(assumed_xanax_cooldown_hours),
    )


def render_ratio_controls(ratio: RatioProfile) -> RatioProfile:
    st.subheader("Baldr ratio")
    c1, c2, c3, c4 = st.columns(4)
    strength = c1.number_input("Strength %", min_value=0.0, max_value=100.0, value=float(ratio.strength), step=0.01)
    speed = c2.number_input("Speed %", min_value=0.0, max_value=100.0, value=float(ratio.speed), step=0.01)
    defense = c3.number_input("Defense %", min_value=0.0, max_value=100.0, value=float(ratio.defense), step=0.01)
    dexterity = c4.number_input("Dexterity %", min_value=0.0, max_value=100.0, value=float(ratio.dexterity), step=0.01)
    total = strength + speed + defense + dexterity
    if abs(total - 100.0) > 0.05:
        st.warning(f"Ratio totals {total:.2f}%. Ideally this should equal 100%.")
    return RatioProfile(strength=strength, speed=speed, defense=defense, dexterity=dexterity)


def render_manual_modifier_controls() -> TrainingModifiers:
    st.subheader("Manual training modifier overrides")
    st.caption("Use these when the API does not expose a training bonus cleanly, or to test scenarios.")

    c1, c2, c3 = st.columns(3)
    with c1:
        all_gym = st.number_input("All gym gains %", value=0.0, step=0.5)
        happy_loss_reduction = st.number_input("Gym happy loss reduction %", value=0.0, step=5.0)
    with c2:
        strength = st.number_input("Strength gym gains %", value=0.0, step=0.5)
        speed = st.number_input("Speed gym gains %", value=0.0, step=0.5)
    with c3:
        defense = st.number_input("Defense gym gains %", value=0.0, step=0.5)
        dexterity = st.number_input("Dexterity gym gains %", value=0.0, step=0.5)

    energy_regen_bonus = st.number_input("Energy regeneration bonus %", value=0.0, step=5.0)

    sources = []
    if any(x != 0.0 for x in [all_gym, strength, speed, defense, dexterity, happy_loss_reduction, energy_regen_bonus]):
        sources.append("Manual override")

    return TrainingModifiers(
        all_gym_gains_pct=float(all_gym),
        strength_gym_gains_pct=float(strength),
        speed_gym_gains_pct=float(speed),
        defense_gym_gains_pct=float(defense),
        dexterity_gym_gains_pct=float(dexterity),
        happy_loss_reduction_pct=float(happy_loss_reduction),
        energy_regen_bonus_pct=float(energy_regen_bonus),
        detected_sources=sources,
    )


def render_progress_section(state: PlayerState, goal: GoalSettings, ratio: RatioProfile) -> None:
    st.subheader("Progress")
    total_progress = min(1.0, state.stats.total() / max(goal.target_total_stats, 1.0))
    st.write(f"Overall goal: {state.stats.total():,.0f} / {goal.target_total_stats:,.0f}")
    st.progress(total_progress)

    phase = milestone_phase(state.stats)
    if phase != "baldr_ratio":
        cap = current_milestone_cap(state.stats) or 400_000
        st.write(f"Current milestone phase: {phase} (target {cap:,} each stat)")
        cols = st.columns(4)
        for idx, stat_key in enumerate(STAT_KEYS):
            value = state.stats.get(stat_key)
            with cols[idx]:
                st.caption(f"{stat_key.title()}: {value:,.0f} / {cap:,.0f}")
                st.progress(min(1.0, value / max(cap, 1.0)))
    else:
        st.write("Baldr ratio maintenance")
        current_ratio = calculate_current_ratio(state.stats)
        target_ratio = ratio.as_percent_map()
        cols = st.columns(4)
        for idx, stat_key in enumerate(STAT_KEYS):
            with cols[idx]:
                diff = abs(current_ratio[stat_key] - target_ratio[stat_key])
                closeness = max(0.0, 1.0 - diff / max(target_ratio[stat_key], 1.0))
                st.caption(f"{stat_key.title()}: {current_ratio[stat_key]:.2f}% / {target_ratio[stat_key]:.2f}%")
                st.progress(closeness)


def render_next_gym_progress(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    st.subheader("Gym unlock progress")
    projection = estimate_next_gym_unlock(state, ratio, goal, manual_mods)
    if projection.next_gym is None or projection.required_progress is None:
        st.write("You are already at the highest gym in the current planner database.")
        return

    st.write(f"Current highest unlocked gym: **{projection.current_gym}**")
    st.write(f"Next gym: **{projection.next_gym}**")
    st.write(f"Estimated progress: **{projection.current_progress:,} / {projection.required_progress:,} E**")
    st.progress(min(1.0, projection.current_progress / max(projection.required_progress, 1)))
    st.caption(f"Remaining energy to unlock: {projection.remaining_energy:,} E")
    if projection.estimated_unlock_at is not None:
        st.success(f"Projected unlock: {projection.estimated_unlock_at.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("Projected unlock is beyond the current preview window.")


def render_player_snapshot(state: PlayerState, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    combined_mods = state.training_modifiers.merge(manual_mods)

    st.subheader("Current player snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strength", f"{state.stats.strength:,.0f}")
    c2.metric("Speed", f"{state.stats.speed:,.0f}")
    c3.metric("Defense", f"{state.stats.defense:,.0f}")
    c4.metric("Dexterity", f"{state.stats.dexterity:,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total stats", f"{state.stats.total():,.0f}")
    c6.metric("Current energy", f"{state.recovery.current_energy}/{state.recovery.max_energy}")
    c7.metric("Drug cooldown", f"{state.recovery.drug_cd_minutes} min")
    c8.metric("Booster cooldown", f"{state.recovery.booster_cd_minutes} min")

    header_bits = []
    if state.torn_name:
        header_bits.append(state.torn_name)
    if state.torn_id:
        header_bits.append(f"ID {state.torn_id}")
    if state.faction_name:
        header_bits.append(f"Faction: {state.faction_name}" if not state.faction_id else f"Faction: {state.faction_name} [{state.faction_id}]")
    if header_bits:
        st.caption(" | ".join(header_bits))

    current_ratio = calculate_current_ratio(state.stats)
    st.caption("Current ratio: " + " | ".join(f"{k.title()} {v:.2f}%" for k, v in current_ratio.items()))
    st.caption(
        f"Baseline daily energy: {state.recovery.baseline_energy_per_day(energy_regen_bonus_pct=combined_mods.energy_regen_bonus_pct):,} "
        f"(natural {state.recovery.natural_energy_per_day(energy_regen_bonus_pct=combined_mods.energy_regen_bonus_pct):,} + xanax + refill)"
    )

    if state.last_sync is not None:
        st.caption(f"Last sync: {state.last_sync.strftime('%Y-%m-%d %H:%M:%S')}")

    if combined_mods.detected_sources:
        st.caption("Training modifiers in effect: " + "; ".join(combined_mods.detected_sources))

    if state.api_notes:
        for note in state.api_notes:
            st.info(note)


def render_unlocked_gym_editor(state: PlayerState) -> None:
    st.subheader("Unlocked gyms")
    st.caption("Live API sync may not expose unlocked gyms directly. Use this as the source of truth whenever needed.")

    gym_names = ordered_gym_names()

    if "gym_multiselect" not in st.session_state:
        st.session_state.gym_multiselect = list(state.unlocked_gyms or st.session_state.manual_unlocked_gyms)

    c1, c2, c3 = st.columns([3, 1, 1])

    with c1:
        selected = st.multiselect(
            "Select the gyms you currently have unlocked",
            options=gym_names,
            default=st.session_state.gym_multiselect,
            key="gym_multiselect_widget",
        )

    with c2:
        highest_gym = st.selectbox(
            "Quick fill to highest unlocked gym",
            options=["-- none --"] + gym_names,
            index=0,
            key="highest_unlocked_gym_selector",
        )

    with c3:
        apply_fill = st.button("Apply quick fill", use_container_width=True)

    if apply_fill and highest_gym != "-- none --":
        highest_idx = gym_names.index(highest_gym)
        selected = gym_names[: highest_idx + 1]
        st.session_state.gym_multiselect = selected
        st.session_state.manual_unlocked_gyms = selected
        state.unlocked_gyms = selected
        st.rerun()
    else:
        st.session_state.gym_multiselect = selected

    ordered_selection = [name for name in gym_names if name in st.session_state.gym_multiselect]
    st.session_state.manual_unlocked_gyms = ordered_selection
    state.unlocked_gyms = ordered_selection

    if ordered_selection:
        st.caption(f"Highest unlocked gym in planner: {ordered_selection[-1]}")
    else:
        st.warning("No unlocked gyms selected yet. The planner cannot choose a gym until this is set.")

def render_today_panel(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    today_instruction = build_daily_instruction(state, ratio, goal, date.today(), manual_mods)
    st.subheader("Today")
    st.write(f"**Day type:** {today_instruction.day_type.replace('_', ' ').title()}")
    st.write(f"**Train:** {today_instruction.target_stat.title() if today_instruction.target_stat != 'none' else 'None'}")
    st.write(f"**Gym:** {today_instruction.gym_name}")
    st.write(f"**Planned energy:** {today_instruction.estimated_energy:,}")
    st.write(f"**Estimated trains:** {today_instruction.recommended_trains:,}")
    st.write(f"**Estimated gain:** {today_instruction.estimated_gain:,.2f}")
    st.write(f"**Happy:** {today_instruction.start_happy:,} → {today_instruction.end_happy:,}")

    for note in today_instruction.notes:
        st.write(f"- {note}")


def render_jump_panel(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    st.subheader("Next jump window")
    combined_mods = state.training_modifiers.merge(manual_mods)
    jump_plan = build_jump_plan(state, ratio, goal, combined_mods)
    if jump_plan is None:
        st.write("No jump is currently projected to beat normal training by enough to schedule one.")
        return

    st.write(f"**Recommended jump:** {jump_plan.jump_type.replace('_', ' ').title()}")
    st.write(f"**Target stat:** {jump_plan.target_stat.title()}")
    st.write(f"**Gym:** {jump_plan.gym_name}")
    st.write(f"**Prep starts:** {jump_plan.prep_start.strftime('%Y-%m-%d %H:%M')}")
    st.write(f"**Execute at:** {jump_plan.execute_at.strftime('%Y-%m-%d %H:%M')}")
    st.write(f"**Projected normal gain:** {jump_plan.projected_normal_gain:,.2f}")
    st.write(f"**Projected jump gain:** {jump_plan.projected_jump_gain:,.2f}")
    st.write(f"**Projected extra gain:** {jump_plan.projected_gain_delta:,.2f}")

    for note in jump_plan.notes:
        st.write(f"- {note}")


def build_today_action_plan(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> List[JumpStep]:
    combined_mods = state.training_modifiers.merge(manual_mods)
    today_type, jump_plan = day_type_for_date(state, ratio, goal, date.today(), combined_mods)
    target_stat = choose_target_stat(state.stats, ratio)
    gym = best_gym_for_stat(state, target_stat)
    now_dt = datetime.now()
    actions: List[JumpStep] = []

    if gym is None:
        return actions

    if jump_plan is not None and today_type in {"prep", "happy_jump", "super_happy_jump"}:
        return [step for step in build_jump_sequence(state, goal, jump_plan) if step.when.date() == date.today()]

    if state.recovery.current_energy > 0:
        actions.append(JumpStep(now_dt, "Train current energy", f"Train your current {state.recovery.current_energy} energy into {target_stat.title()} at {gym.name}."))

    if state.recovery.daily_refill_enabled:
        refill_dt = now_dt + timedelta(minutes=10)
        actions.append(JumpStep(refill_dt, "Use daily refill", "Use your daily refill after your current energy block if you are training today."))
        actions.append(JumpStep(refill_dt + timedelta(minutes=1), "Train refill energy", f"Train the refill energy into {target_stat.title()} at {gym.name}."))

    natural_e = natural_energy_between(state, combined_mods, now_dt, end_of_day(now_dt))
    if natural_e > 0:
        actions.append(JumpStep(end_of_day(now_dt), "Natural regen through end of day", f"About {natural_e} natural energy will regenerate by the end of today at your current regen rate."))

    if state.recovery.drug_cd_minutes > 0:
        drug_clear_dt = now_dt + timedelta(minutes=state.recovery.drug_cd_minutes)
        actions.append(JumpStep(drug_clear_dt, "Drug cooldown clears", "You cannot take another Xanax until this time."))
        if drug_clear_dt.date() == now_dt.date():
            actions.append(JumpStep(drug_clear_dt + timedelta(minutes=1), "Take next Xanax", f"If you are following the baseline plan, take Xanax at cooldown clear and train the extra {state.recovery.xanax_energy} energy after it lands."))
    else:
        actions.append(JumpStep(now_dt, "Drug available now", "You can take your next Xanax whenever you decide to use it."))

    unlock_today = estimate_today_unlock_from_blocks(state, goal, combined_mods, now_dt)
    if unlock_today is not None:
        unlock_dt, next_gym = unlock_today
        actions.append(JumpStep(unlock_dt, "Next gym unlocks", f"Projected unlock: {next_gym}. This estimate only uses today's current energy, refill, natural regen, and a same-day Xanax if cooldown clears today."))

    return actions


def render_daily_planner_panel(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    st.subheader("Daily planner")
    steps = build_today_action_plan(state, ratio, goal, manual_mods)
    if not steps:
        st.write("No timed actions available yet.")
        return
    rows = []
    for idx, step in enumerate(sorted(steps, key=lambda x: x.when), start=1):
        rows.append({
            "Step": idx,
            "When": step.when.strftime("%Y-%m-%d %H:%M"),
            "Action": step.action,
            "Details": step.details,
        })
    st.dataframe(rows, use_container_width=True)


def render_jump_sequence_panel(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    st.subheader("Jump sequence")
    combined_mods = state.training_modifiers.merge(manual_mods)
    jump_plan = build_jump_plan(state, ratio, goal, combined_mods)
    if jump_plan is None:
        st.write("No jump is currently scheduled.")
        return
    rows = []
    for idx, step in enumerate(build_jump_sequence(state, goal, jump_plan), start=1):
        rows.append({
            "Step": idx,
            "When": step.when.strftime("%Y-%m-%d %H:%M"),
            "Action": step.action,
            "Details": step.details,
        })
    st.dataframe(rows, use_container_width=True)


def render_plan_table(plan: List[DailyInstruction]) -> None:
    st.subheader("Plan preview")
    rows = []
    for item in plan:
        rows.append(
            {
                "Date": item.plan_date.isoformat(),
                "Type": item.day_type,
                "Train": item.target_stat,
                "Gym": item.gym_name,
                "Energy": item.estimated_energy,
                "Trains": item.recommended_trains,
                "Est gain": round(item.estimated_gain, 2),
                "Happy start": item.start_happy,
                "Happy end": item.end_happy,
                "Notes": " | ".join(item.notes),
            }
        )
    st.dataframe(rows, use_container_width=True)


def render_forecast(state: PlayerState, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    st.subheader("Forecast")
    est_days = days_until_goal_estimate(state, goal, manual_mods)
    target_days = (goal.target_date - date.today()).days
    if est_days is None:
        st.error("Forecast unavailable until a valid gym path and gain model are set.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Estimated days to goal", est_days)
    c2.metric("Days until target date", target_days)
    if est_days <= target_days:
        st.success("Current baseline projection is on pace for the selected goal.")
    else:
        st.warning("Current baseline projection is behind the selected goal. Jump/booster optimization will matter.")


def render_war_calendar_editor(state: PlayerState) -> None:
    st.subheader("War / non-training days")
    st.caption("API-loaded war days can be edited manually here. Manual edits become the planner source of truth for this session.")
    default_text = ", ".join(d.isoformat() for d in state.faction_war_days)
    raw = st.text_input("War days (comma-separated YYYY-MM-DD)", value=default_text)
    parsed_days: List[date] = []
    if raw.strip():
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed_days.append(date.fromisoformat(token))
            except ValueError:
                st.warning(f"Could not parse war date: {token}")
    state.faction_war_days = sorted(set(parsed_days))


def render_gain_debug_panel(state: PlayerState, goal: GoalSettings, ratio: RatioProfile, manual_mods: TrainingModifiers) -> None:
    st.subheader("Gain engine debug")
    target_stat = choose_target_stat(state.stats, ratio)
    gym = best_gym_for_stat(state, target_stat)
    if gym is None:
        st.warning("No gym selected yet.")
        return

    combined_mods = state.training_modifiers.merge(manual_mods)
    start_happy = max(state.recovery.max_happy, goal.normal_day_start_happy)
    sim = simulate_training_block(
        base_stats=state.stats,
        stat_key=target_stat,
        gym=gym,
        total_energy=min(150, state.recovery.baseline_energy_per_day(energy_regen_bonus_pct=combined_mods.energy_regen_bonus_pct)),
        starting_happy=start_happy,
        mods=combined_mods,
    )

    st.caption(f"Previewing first {min(150, state.recovery.baseline_energy_per_day(energy_regen_bonus_pct=combined_mods.energy_regen_bonus_pct))} energy into {target_stat.title()} at {gym.name}.")
    rows = []
    for item in sim["per_train_preview"]:
        rows.append({"Train #": item["train"], "Happy before": round(item["happy_before"], 2), "Gain": round(item["gain"], 4)})
    st.dataframe(rows, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Torn Stat Tracker v2", layout="wide")
    init_state()

    st.title("Torn Stat Tracker v2")
    st.caption("Day-by-day battle stat planner with Baldr ratio, gym optimization, merged jump scheduling, real gain simulation, and war-day handling.")

    api_key, preview_days = render_sidebar()

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if st.button("Sync from Torn API", use_container_width=True):
            try:
                st.session_state.player_state = fetch_player_state_from_api(api_key=api_key, manual_unlocked_gyms=st.session_state.manual_unlocked_gyms)
                synced = st.session_state.player_state
                if synced.unlocked_gyms:
                    st.session_state.manual_unlocked_gyms = list(synced.unlocked_gyms)
                    st.session_state.gym_multiselect = list(synced.unlocked_gyms)
                base_happy = synced.recovery.max_happy or synced.recovery.current_happy
                if base_happy > 0:
                    st.session_state.goal_settings.normal_day_start_happy = int(base_happy)
                st.success("Profile synced from Torn API.")
            except Exception as exc:
                st.error(f"Sync failed: {exc}")
    with col_b:
        if st.button("Load demo data", use_container_width=True):
            demo_state = build_demo_player_state()
            st.session_state.player_state = demo_state
            st.session_state.manual_unlocked_gyms = list(demo_state.unlocked_gyms)
            st.session_state.gym_multiselect = list(demo_state.unlocked_gyms)
            st.session_state.goal_settings.normal_day_start_happy = int(demo_state.recovery.max_happy or demo_state.recovery.current_happy or st.session_state.goal_settings.normal_day_start_happy)
            st.success("Demo profile loaded.")

    if st.session_state.player_state is None:
        st.info("Load demo data or sync with your API key to begin.")
        return

    player_state: PlayerState = st.session_state.player_state
    st.session_state.goal_settings = render_goal_controls(st.session_state.goal_settings)
    st.session_state.ratio_profile = render_ratio_controls(st.session_state.ratio_profile)
    manual_mods = render_manual_modifier_controls()

    render_progress_section(player_state, st.session_state.goal_settings, st.session_state.ratio_profile)
    render_next_gym_progress(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
    render_player_snapshot(player_state, st.session_state.goal_settings, manual_mods)
    render_unlocked_gym_editor(player_state)
    render_war_calendar_editor(player_state)
    render_today_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
    render_jump_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
    render_daily_planner_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
    render_jump_sequence_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)

    plan = build_plan_preview(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods, days=preview_days)
    render_plan_table(plan)
    render_forecast(player_state, st.session_state.goal_settings, manual_mods)
    render_gain_debug_panel(player_state, st.session_state.goal_settings, st.session_state.ratio_profile, manual_mods)

    st.subheader("Next implementation targets")
    st.write("1. Add alternate 99k recipes and exact item counts for your preferred method.")
    st.write("2. Add richer faction upgrade parsing from the live upgrades payload.")
    st.write("3. Add export / save functionality for the training plan.")
    st.write("4. Add calendar import / reminder export for jump schedules.")


if __name__ == "__main__":
    main()
