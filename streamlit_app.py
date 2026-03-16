
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict, is_dataclass
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo, available_timezones

import json
import math
import re
import hashlib

import requests
import streamlit as st
import streamlit.components.v1 as components


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
DEFAULT_APP_TIMEZONE_NAME = "America/Chicago"
DEFAULT_APP_TIMEZONE_LABEL = "America/Chicago (Central Time)"
TORN_TIMEZONE = ZoneInfo("UTC")
TORN_TIMEZONE_LABEL = "TST (UTC)"
APP_TIMEZONE = ZoneInfo(DEFAULT_APP_TIMEZONE_NAME)
ALL_TIMEZONE_OPTIONS = sorted(available_timezones())
PERSISTENCE_PATH = Path(".streamlit/torn_planner_persistence.json")




def set_app_timezone(timezone_name: Optional[str]) -> None:
    global APP_TIMEZONE
    try:
        APP_TIMEZONE = ZoneInfo(timezone_name or DEFAULT_APP_TIMEZONE_NAME)
    except Exception:
        APP_TIMEZONE = ZoneInfo(DEFAULT_APP_TIMEZONE_NAME)

def _api_namespace(api_key: str) -> Optional[str]:
    key = (api_key or "").strip()
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _jsonify(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__kind__": "datetime", "value": value.isoformat()}
    if isinstance(value, date) and not isinstance(value, datetime):
        return {"__kind__": "date", "value": value.isoformat()}
    if isinstance(value, dtime):
        return {"__kind__": "time", "value": value.isoformat()}
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value


def _dejsonify(value: Any) -> Any:
    if isinstance(value, dict):
        kind = value.get("__kind__")
        if kind == "datetime":
            return datetime.fromisoformat(value["value"])
        if kind == "date":
            return date.fromisoformat(value["value"])
        if kind == "time":
            return dtime.fromisoformat(value["value"])
        return {k: _dejsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_dejsonify(v) for v in value]
    return value


def _player_state_from_dict(data: Dict[str, Any]) -> PlayerState:
    return PlayerState(
        stats=PlayerStats(**data.get("stats", {})),
        recovery=RecoveryState(**data.get("recovery", {})),
        unlocked_gyms=list(data.get("unlocked_gyms", [])),
        faction_war_days=list(data.get("faction_war_days", [])),
        torn_name=data.get("torn_name", ""),
        torn_id=data.get("torn_id"),
        faction_id=data.get("faction_id"),
        faction_name=data.get("faction_name", ""),
        training_modifiers=TrainingModifiers(**data.get("training_modifiers", {})),
        api_notes=list(data.get("api_notes", [])),
        last_sync=data.get("last_sync"),
    )


def _current_persistence_payload() -> Dict[str, Any]:
    return {
        "goal_settings": st.session_state.get("goal_settings"),
        "ratio_profile": st.session_state.get("ratio_profile"),
        "manual_mods": st.session_state.get("manual_mods"),
        "manual_unlocked_gyms": st.session_state.get("manual_unlocked_gyms", []),
        "gym_multiselect": st.session_state.get("gym_multiselect", []),
        "highest_unlocked_gym_selector": st.session_state.get("highest_unlocked_gym_selector", "-- none --"),
        "manual_99k_jump_entries": st.session_state.get("manual_99k_jump_entries", []),
        "player_state": st.session_state.get("player_state"),
        "preview_days": st.session_state.get("preview_days", 30),
        "display_timezone_name": st.session_state.get("display_timezone_name", DEFAULT_APP_TIMEZONE_NAME),
        "use_tct_times": st.session_state.get("use_tct_times", False),
        "sleep_schedule_enabled": st.session_state.get("sleep_schedule_enabled", False),
        "sleep_start_time": st.session_state.get("sleep_start_time", dtime(hour=23, minute=0)),
        "sleep_end_time": st.session_state.get("sleep_end_time", dtime(hour=7, minute=0)),
        "notifications_enabled": st.session_state.get("notifications_enabled", True),
        "notification_toasts_enabled": st.session_state.get("notification_toasts_enabled", True),
        "notification_browser_enabled": st.session_state.get("notification_browser_enabled", False),
        "notification_lead_minutes": st.session_state.get("notification_lead_minutes", 10),
        "notify_refill_ready": st.session_state.get("notify_refill_ready", True),
        "notify_drug_clear": st.session_state.get("notify_drug_clear", True),
        "notify_booster_clear": st.session_state.get("notify_booster_clear", True),
        "notify_jump_prep": st.session_state.get("notify_jump_prep", True),
        "notify_jump_execute": st.session_state.get("notify_jump_execute", True),
        "notify_gym_unlock": st.session_state.get("notify_gym_unlock", True),
    }


def _read_persistence_store() -> Dict[str, Any]:
    if not PERSISTENCE_PATH.exists():
        return {"profiles": {}}
    try:
        payload = _dejsonify(json.loads(PERSISTENCE_PATH.read_text(encoding="utf-8")))
        if isinstance(payload, dict) and isinstance(payload.get("profiles"), dict):
            return payload
    except Exception:
        pass
    return {"profiles": {}}


def _write_persistence_store(store: Dict[str, Any]) -> None:
    PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERSISTENCE_PATH.write_text(json.dumps(_jsonify(store), indent=2), encoding="utf-8")


def _safe_dataclass_load(cls, payload: Any, aliases: Optional[Dict[str, str]] = None):
    if not isinstance(payload, dict):
        return cls()
    aliases = aliases or {}
    allowed = {f.name for f in fields(cls)}
    filtered = {}
    for key, value in payload.items():
        mapped = aliases.get(key, key)
        if mapped in allowed:
            filtered[mapped] = value
    return cls(**filtered)


def reset_runtime_state(keep_api_fields: bool = True) -> None:
    loaded_namespace = st.session_state.get("_loaded_persistence_namespace") if keep_api_fields else None
    st.session_state.player_state = None
    st.session_state.goal_settings = GoalSettings()
    st.session_state.ratio_profile = RatioProfile()
    st.session_state.manual_mods = TrainingModifiers()
    st.session_state.manual_unlocked_gyms = []
    st.session_state.gym_multiselect = []
    st.session_state.selected_calendar_date = None
    st.session_state.highest_unlocked_gym_selector = "-- none --"
    st.session_state.manual_99k_jump_date = local_today() + timedelta(days=7)
    st.session_state.manual_99k_jump_entries = manual_99k_schedule_datetimes(st.session_state.goal_settings)
    st.session_state.preview_days = 30
    st.session_state.display_timezone_name = DEFAULT_APP_TIMEZONE_NAME
    st.session_state.use_tct_times = False
    st.session_state.sleep_schedule_enabled = False
    st.session_state.sleep_start_time = dtime(hour=23, minute=0)
    st.session_state.sleep_end_time = dtime(hour=7, minute=0)
    st.session_state.notifications_enabled = True
    st.session_state.notification_toasts_enabled = True
    st.session_state.notification_browser_enabled = False
    st.session_state.notification_lead_minutes = 10
    st.session_state.notify_refill_ready = True
    st.session_state.notify_drug_clear = True
    st.session_state.notify_booster_clear = True
    st.session_state.notify_jump_prep = True
    st.session_state.notify_jump_execute = True
    st.session_state.notify_gym_unlock = True
    st.session_state._notified_events = []
    st.session_state._persistence_error = None
    if keep_api_fields:
        st.session_state._loaded_persistence_namespace = loaded_namespace


def _apply_persistent_payload(payload: Dict[str, Any]) -> None:
    if isinstance(payload.get("goal_settings"), dict):
        st.session_state.goal_settings = _safe_dataclass_load(
            GoalSettings,
            payload["goal_settings"],
            aliases={
                "auto_schedule_jumps": "auto_schedule_happy_jumps",
            },
        )
    if isinstance(payload.get("ratio_profile"), dict):
        st.session_state.ratio_profile = _safe_dataclass_load(RatioProfile, payload["ratio_profile"])
    if isinstance(payload.get("manual_mods"), dict):
        st.session_state.manual_mods = _safe_dataclass_load(TrainingModifiers, payload["manual_mods"])
    if isinstance(payload.get("player_state"), dict):
        st.session_state.player_state = _player_state_from_dict(payload["player_state"])
    st.session_state.manual_unlocked_gyms = list(payload.get("manual_unlocked_gyms", []))
    st.session_state.gym_multiselect = list(payload.get("gym_multiselect", []))
    st.session_state.highest_unlocked_gym_selector = payload.get("highest_unlocked_gym_selector", "-- none --")
    st.session_state.manual_99k_jump_entries = list(payload.get("manual_99k_jump_entries", st.session_state.get("manual_99k_jump_entries", [])))
    st.session_state.preview_days = int(payload.get("preview_days", 30))
    st.session_state.display_timezone_name = payload.get("display_timezone_name", DEFAULT_APP_TIMEZONE_NAME)
    st.session_state.use_tct_times = bool(payload.get("use_tct_times", False))
    st.session_state.sleep_schedule_enabled = bool(payload.get("sleep_schedule_enabled", False))
    st.session_state.sleep_start_time = payload.get("sleep_start_time", dtime(hour=23, minute=0))
    st.session_state.sleep_end_time = payload.get("sleep_end_time", dtime(hour=7, minute=0))
    st.session_state.notifications_enabled = bool(payload.get("notifications_enabled", True))
    st.session_state.notification_toasts_enabled = bool(payload.get("notification_toasts_enabled", True))
    st.session_state.notification_browser_enabled = bool(payload.get("notification_browser_enabled", False))
    st.session_state.notification_lead_minutes = int(payload.get("notification_lead_minutes", 10))
    st.session_state.notify_refill_ready = bool(payload.get("notify_refill_ready", True))
    st.session_state.notify_drug_clear = bool(payload.get("notify_drug_clear", True))
    st.session_state.notify_booster_clear = bool(payload.get("notify_booster_clear", True))
    st.session_state.notify_jump_prep = bool(payload.get("notify_jump_prep", True))
    st.session_state.notify_jump_execute = bool(payload.get("notify_jump_execute", True))
    st.session_state.notify_gym_unlock = bool(payload.get("notify_gym_unlock", True))


def load_persistent_state_for_api(api_key: str) -> None:
    namespace = _api_namespace(api_key)
    current_namespace = st.session_state.get("_loaded_persistence_namespace")
    if namespace == current_namespace:
        return

    reset_runtime_state(keep_api_fields=True)
    st.session_state._loaded_persistence_namespace = namespace

    if not namespace:
        return

    try:
        store = _read_persistence_store()
        payload = store.get("profiles", {}).get(namespace)
        if isinstance(payload, dict):
            _apply_persistent_payload(payload)
    except Exception as exc:
        st.session_state._persistence_error = str(exc)


def save_persistent_state(api_key: str = "") -> None:
    namespace = _api_namespace(api_key)
    if not namespace:
        return
    store = _read_persistence_store()
    profiles = store.setdefault("profiles", {})
    payload = _current_persistence_payload()
    metadata = {
        "torn_name": getattr(st.session_state.get("player_state"), "torn_name", "") if st.session_state.get("player_state") else "",
        "torn_id": getattr(st.session_state.get("player_state"), "torn_id", None) if st.session_state.get("player_state") else None,
        "saved_at": local_now(),
    }
    payload["_meta"] = metadata
    profiles[namespace] = payload
    _write_persistence_store(store)


def clear_persistent_state(api_key: str = "") -> None:
    namespace = _api_namespace(api_key)
    if not namespace or not PERSISTENCE_PATH.exists():
        return
    store = _read_persistence_store()
    profiles = store.get("profiles", {})
    if namespace in profiles:
        del profiles[namespace]
    _write_persistence_store(store)


def local_now() -> datetime:
    return datetime.now(APP_TIMEZONE)


def local_today() -> date:
    return local_now().date()


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=APP_TIMEZONE)
    return dt.astimezone(APP_TIMEZONE)


def to_tst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=APP_TIMEZONE)
    return dt.astimezone(TORN_TIMEZONE)


def fmt_local(dt: datetime) -> str:
    return to_local(dt).strftime("%Y-%m-%d %H:%M %Z")


def fmt_tst(dt: datetime) -> str:
    return to_tst(dt).strftime("%Y-%m-%d %H:%M TST")




def get_app_timezone_label() -> str:
    if getattr(st, "session_state", None) is not None and bool(st.session_state.get("use_tct_times", False)):
        return TORN_TIMEZONE_LABEL
    return str(st.session_state.get("display_timezone_name", DEFAULT_APP_TIMEZONE_NAME))

def app_vs_tst_text(now_dt: Optional[datetime] = None) -> str:
    now_dt = now_dt or local_now()
    if bool(st.session_state.get("use_tct_times", False)):
        return "Displayed times are currently using TST"

    local_dt = to_local(now_dt)
    tst_dt = to_tst(now_dt)
    offset = tst_dt.utcoffset() - local_dt.utcoffset()
    if offset == timedelta(0):
        return f"{get_app_timezone_label()} matches TST right now"

    direction = "behind" if offset > timedelta(0) else "ahead of"
    delta = abs(offset)
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    if minutes:
        return f"{get_app_timezone_label()} is {hours}h {minutes}m {direction} TST right now"
    return f"{get_app_timezone_label()} is {hours}h {direction} TST right now"


def ct_vs_tst_text(now_dt: Optional[datetime] = None) -> str:
    now_dt = to_local(now_dt or local_now())
    offset = now_dt.utcoffset() or timedelta(0)
    if offset < timedelta(0):
        delta = -offset
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        if minutes:
            return f"CT is {hours}h {minutes}m behind TST right now"
        return f"CT is {hours}h behind TST right now"
    if offset > timedelta(0):
        delta = offset
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        if minutes:
            return f"CT is {hours}h {minutes}m ahead of TST right now"
        return f"CT is {hours}h ahead of TST right now"
    return "CT matches TST right now"


def browser_notify(title: str, body: str) -> None:
    safe_title = json.dumps(title)
    safe_body = json.dumps(body)
    components.html(
        f"""
        <script>
        const title = {safe_title};
        const body = {safe_body};
        const send = () => {{ try {{ new Notification(title, {{ body }}); }} catch (e) {{}} }};
        if (window.Notification) {{
            if (Notification.permission === 'granted') {{ send(); }}
            else if (Notification.permission !== 'denied') {{ Notification.requestPermission().then(p => {{ if (p === 'granted') send(); }}); }}
        }}
        </script>
        """,
        height=0,
    )


def _notification_seen(event_id: str) -> bool:
    return event_id in set(st.session_state.get('_notified_events', []))


def _mark_notification_seen(event_id: str) -> None:
    seen = set(st.session_state.get('_notified_events', []))
    seen.add(event_id)
    st.session_state._notified_events = list(seen)


def emit_notification(event_id: str, title: str, body: str) -> None:
    if _notification_seen(event_id):
        return
    if bool(st.session_state.get('notification_toasts_enabled', True)):
        st.toast(f"{title}: {body}")
    if bool(st.session_state.get('notification_browser_enabled', False)):
        browser_notify(title, body)
    _mark_notification_seen(event_id)


def maybe_notify_at(event_id: str, target_dt: Optional[datetime], title: str, body: str) -> None:
    if target_dt is None or not bool(st.session_state.get('notifications_enabled', True)):
        return
    now_dt = local_now()
    target_dt = to_local(target_dt)
    lead = timedelta(minutes=int(st.session_state.get('notification_lead_minutes', 10)))
    if target_dt - lead <= now_dt <= target_dt + timedelta(minutes=10):
        emit_notification(event_id, title, body)


def run_notification_checks(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    if not bool(st.session_state.get('notifications_enabled', True)):
        return

    now_dt = local_now()
    if bool(st.session_state.get('notify_refill_ready', True)):
        refill_dt = next_daily_refill_ready_local(goal, now_dt=now_dt)
        maybe_notify_at(f"refill:{refill_dt.isoformat()}", refill_dt, 'Daily refill ready', f'Refill is available at {fmt_local(refill_dt)}')

    if bool(st.session_state.get('notify_drug_clear', True)) and state.recovery.drug_cd_minutes > 0:
        drug_dt = now_dt + timedelta(minutes=state.recovery.drug_cd_minutes)
        maybe_notify_at(f"drug:{drug_dt.isoformat()}", drug_dt, 'Drug cooldown clear', f'Next Xanax window opens at {fmt_local(drug_dt)}')

    if bool(st.session_state.get('notify_booster_clear', True)) and state.recovery.booster_cd_minutes > 0:
        booster_dt = now_dt + timedelta(minutes=state.recovery.booster_cd_minutes)
        maybe_notify_at(f"booster:{booster_dt.isoformat()}", booster_dt, 'Booster cooldown clear', f'Booster items are available at {fmt_local(booster_dt)}')

    projection = estimate_next_gym_unlock(state, ratio, goal, manual_mods, days=90)
    if bool(st.session_state.get('notify_gym_unlock', True)) and getattr(projection, 'estimated_unlock_at', None) is not None and projection.next_gym:
        maybe_notify_at(
            f"gymunlock:{projection.next_gym}:{projection.estimated_unlock_at.isoformat()}",
            projection.estimated_unlock_at,
            'Gym unlock reached',
            f"{projection.next_gym} unlocks at {fmt_local(projection.estimated_unlock_at)}",
        )

    jump_plan = build_jump_plan(state, ratio, goal, state.training_modifiers.merge(manual_mods))
    if jump_plan is not None:
        if bool(st.session_state.get('notify_jump_prep', True)):
            maybe_notify_at(
                f"jumpprep:{jump_plan.execute_at.isoformat()}",
                jump_plan.prep_start,
                'Jump prep start',
                f"Start preparing for your {jump_plan.jump_type.replace('_', ' ')} targeting {jump_plan.target_stat.title()} in {jump_plan.gym_name}",
            )
        if bool(st.session_state.get('notify_jump_execute', True)):
            maybe_notify_at(
                f"jumpexec:{jump_plan.execute_at.isoformat()}",
                jump_plan.execute_at,
                'Jump execute now',
                f"Run your {jump_plan.jump_type.replace('_', ' ')} in {jump_plan.gym_name} and train {jump_plan.target_stat.title()} now",
            )


def next_tst_midnight_local(now_dt: Optional[datetime] = None) -> datetime:
    now_tst = to_tst(now_dt or local_now())
    next_midnight_tst = datetime.combine(now_tst.date() + timedelta(days=1), dtime.min, tzinfo=TORN_TIMEZONE)
    return next_midnight_tst.astimezone(APP_TIMEZONE)


def next_daily_refill_ready_local(goal: "GoalSettings", now_dt: Optional[datetime] = None, after_dt: Optional[datetime] = None) -> datetime:
    now_dt = to_local(now_dt or local_now())
    after_dt = to_local(after_dt or now_dt)
    if not getattr(goal, "daily_refill_used_today", True):
        return max(after_dt, now_dt + timedelta(minutes=10))
    return max(after_dt, next_tst_midnight_local(now_dt))


def _minutes_between_times(start: dtime, end: dtime) -> int:
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    if end_minutes >= start_minutes:
        return end_minutes - start_minutes
    return (24 * 60 - start_minutes) + end_minutes


def sleep_minutes_per_day(goal: "GoalSettings") -> int:
    if not getattr(goal, "sleep_schedule_enabled", False):
        return 0
    start = getattr(goal, "sleep_start_time", dtime(hour=23, minute=0))
    end = getattr(goal, "sleep_end_time", dtime(hour=7, minute=0))
    if start == end:
        return 0
    return _minutes_between_times(start, end)


def awake_minutes_per_day(goal: "GoalSettings") -> int:
    return max(0, 24 * 60 - sleep_minutes_per_day(goal))


def is_sleep_time(goal: "GoalSettings", dt: datetime) -> bool:
    if not getattr(goal, "sleep_schedule_enabled", False):
        return False
    local_dt = to_local(dt)
    start = getattr(goal, "sleep_start_time", dtime(hour=23, minute=0))
    end = getattr(goal, "sleep_end_time", dtime(hour=7, minute=0))
    current = local_dt.time().replace(second=0, microsecond=0)
    if start == end:
        return False
    if start < end:
        return start <= current < end
    return current >= start or current < end


def next_awake_time(goal: "GoalSettings", dt: datetime) -> datetime:
    candidate = to_local(dt)
    if not is_sleep_time(goal, candidate):
        return candidate
    end = getattr(goal, "sleep_end_time", dtime(hour=7, minute=0))
    if candidate.time() < end:
        wake_date = candidate.date()
    else:
        wake_date = candidate.date() + timedelta(days=1)
    return datetime.combine(wake_date, end, tzinfo=APP_TIMEZONE)


def next_awake_quarter_hour(goal: "GoalSettings", dt: datetime) -> datetime:
    candidate = to_local(dt)
    if is_sleep_time(goal, candidate):
        candidate = next_awake_time(goal, candidate)
        if candidate.minute % 15 == 0 and candidate.second == 0 and candidate.microsecond == 0:
            return candidate.replace(second=5)
    return next_quarter_hour(candidate)


def schedule_action_time(goal: "GoalSettings", dt: datetime) -> datetime:
    return next_awake_time(goal, to_local(dt))


def estimated_daily_xanax_capacity(state: "PlayerState", goal: "GoalSettings") -> int:
    awake_minutes = awake_minutes_per_day(goal) if getattr(goal, "sleep_schedule_enabled", False) else 24 * 60
    cooldown_minutes = max(1, int(round(float(getattr(goal, "assumed_xanax_cooldown_hours", 8.0)) * 60)))
    if awake_minutes <= 0:
        return 0
    capacity = 1 + max(0, (awake_minutes - 1) // cooldown_minutes)
    return max(0, min(int(state.recovery.xanax_per_day), int(capacity)))


def planner_baseline_energy_per_day(state: "PlayerState", goal: "GoalSettings", mods: "TrainingModifiers") -> int:
    total = state.recovery.natural_energy_per_day(energy_regen_bonus_pct=mods.energy_regen_bonus_pct)
    total += estimated_daily_xanax_capacity(state, goal) * state.recovery.xanax_energy
    if state.recovery.daily_refill_enabled:
        total += state.recovery.refill_energy
    return total


def sleep_schedule_summary(goal: "GoalSettings", assumed_xanax_cooldown_hours: Optional[float] = None) -> str:
    if not getattr(goal, "sleep_schedule_enabled", False):
        return "Sleep schedule is off."
    awake = awake_minutes_per_day(goal)
    hours = awake // 60
    minutes = awake % 60
    cooldown_hours = float(assumed_xanax_cooldown_hours if assumed_xanax_cooldown_hours is not None else getattr(goal, "assumed_xanax_cooldown_hours", 8.0))
    cooldown_minutes = max(1, int(round(cooldown_hours * 60)))
    xanax_capacity = 1 + max(0, (awake - 1) // cooldown_minutes) if awake > 0 else 0
    return f"Wake window is about {hours}h {minutes}m per day. Baseline Xanax capacity while awake is about {xanax_capacity} dose(s)."


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


BUILD_FAMILY_OPTIONS = ["Custom", "Balanced", "Baldr", "Hank", "Goober Min", "Goober Max"]
PAIR_STAT_MAP = {"strength": "speed", "speed": "strength", "defense": "dexterity", "dexterity": "defense"}
PAIR_GROUPS = {
    "strength": ("strength", "speed"),
    "speed": ("strength", "speed"),
    "defense": ("defense", "dexterity"),
    "dexterity": ("defense", "dexterity"),
}
BUILD_FAMILY_WEIGHTS = {
    "Balanced": {"high": 25.0, "second": 25.0, "medium": 25.0, "low": 25.0},
    "Baldr": {"high": 30.86, "second": 24.69, "medium": 22.22, "low": 22.22},
    "Hank": {"high": 34.72, "second": 27.78, "medium": 27.78, "low": 9.72},
    "Goober Min": {"high": 38.46, "second": 30.77, "medium": 30.77, "low": 0.0},
    "Goober Max": {"high": 42.86, "second": 28.57, "medium": 28.57, "low": 0.0},
}

SPECIALIST_TARGET_OPTIONS = [
    "Auto from build",
    "None",
    "Balboas Gym",
    "Frontline Fitness",
    "Gym 3000",
    "Mr. Isoyamas",
    "Total Rebound",
    "Elites",
    "The Sports Science Lab",
    "Fight Club",
]


def default_specialist_target(family: str, primary: str) -> str:
    if family == "Balanced":
        return "None"
    if family == "Baldr":
        return FRONTLINE_GYM_NAME if primary in {"strength", "speed"} else BALBOAS_GYM_NAME
    if family == "Hank":
        return BALBOAS_GYM_NAME if primary in {"strength", "speed"} else FRONTLINE_GYM_NAME
    if family in {"Goober Min", "Goober Max"}:
        return {
            "strength": GYM3000_NAME,
            "speed": ISOYAMAS_NAME,
            "defense": TOTAL_REBOUND_NAME,
            "dexterity": ELITES_NAME,
        }.get(primary, "None")
    return "None"


def resolve_specialist_target(goal: "GoalSettings") -> str:
    target = getattr(goal, "specialist_gym_target", "Auto from build")
    if target == "Auto from build":
        return default_specialist_target(getattr(goal, "ratio_family", "Custom"), getattr(goal, "ratio_primary_stat", "strength"))
    return target


def ratio_profile_from_build(family: str, primary: str, fallback: Optional[RatioProfile] = None) -> RatioProfile:
    family = family if family in BUILD_FAMILY_OPTIONS else "Custom"
    primary = primary if primary in STAT_KEYS else "strength"
    if family == "Custom":
        return fallback or RatioProfile()
    if family == "Balanced":
        return RatioProfile(25.0, 25.0, 25.0, 25.0)

    weights = BUILD_FAMILY_WEIGHTS[family]
    paired = PAIR_STAT_MAP[primary]
    other_group = [s for s in STAT_KEYS if s not in {primary, paired}]
    values = {stat: weights["medium"] for stat in STAT_KEYS}
    values[primary] = weights["high"]
    values[paired] = weights["second"] if family == "Baldr" else weights["low"]
    for stat in other_group:
        values[stat] = weights["medium"]
    return RatioProfile(**values)


def build_family_specialist_summary(family: str, primary: str) -> str:
    if family == "Balanced":
        return "Balanced keeps all stats even, but does not aim to hold specialist gyms."
    if family == "Baldr":
        if primary in {"strength", "speed"}:
            return "Baldr in the strength/speed family aims to hold Frontline Fitness plus the matching single-stat 50E gym while leaving defense/dexterity on George's."
        return "Baldr in the defense/dexterity family aims to hold Balboas Gym plus the matching single-stat 50E gym while leaving strength/speed on George's."
    if family == "Hank":
        if primary in {"strength", "speed"}:
            return "Hank uses Balboas plus the matching single-stat 50E gym; your paired offensive stat becomes the dump stat."
        return "Hank uses Frontline plus the matching single-stat 50E gym; your paired defensive stat becomes the dump stat."
    if family in {"Goober Min", "Goober Max"}:
        return "Goober-style builds are extreme specialist-gym routes that push one stat very high, dump its paired stat, and keep the opposite pair in the middle."
    return "Custom lets you set any ratio manually."


def build_family_ratio_caption(family: str, primary: str) -> str:
    if family == "Custom":
        return "Custom ratio: edit the percentages directly."
    ratio = ratio_profile_from_build(family, primary)
    return f"{family} preset for {primary.title()}: Str {ratio.strength:.2f}% | Spd {ratio.speed:.2f}% | Def {ratio.defense:.2f}% | Dex {ratio.dexterity:.2f}%"


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
    target_date: date = field(default_factory=lambda: local_today() + timedelta(days=210))
    fhc_allowed: bool = True
    cans_allowed: bool = True
    auto_schedule_happy_jumps: bool = True
    schedule_99k_jump: bool = False
    scheduled_99k_jump_date: date = field(default_factory=lambda: local_today() + timedelta(days=7))
    scheduled_99k_jump_time: dtime = dtime(hour=0, minute=15)
    manual_99k_jump_schedule_text: str = ""
    current_gym_energy_progress: int = 0
    gym_progress_as_of_date: date = field(default_factory=local_today)
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
    daily_refill_available_now: bool = True
    daily_refill_used_today: bool = True
    current_company_stars: int = 0
    planned_10_star_date: date = field(default_factory=lambda: local_today() + timedelta(days=30))
    use_job_points_energy: bool = False
    current_job_points: int = 0
    reserve_job_points: int = 0
    job_points_daily_limit: int = 100
    job_energy_per_point: int = 5
    mcs_ready_claims_now: int = 0
    mcs_energy_per_claim: int = 100
    mcs_next_ready_date: date = field(default_factory=lambda: local_today() + timedelta(days=1))
    mcs_next_ready_time: dtime = dtime(hour=12, minute=0)
    fhc_count_available: int = 0
    fhc_effective_energy: int = 150
    can_count_available: int = 0
    can_energy_per_can: int = 25
    can_cooldown_hours: float = 2.0
    fhc_cooldown_hours: float = 6.0
    max_daily_booster_cooldown_hours: float = 24.0
    sleep_schedule_enabled: bool = False
    sleep_start_time: dtime = dtime(hour=23, minute=0)
    sleep_end_time: dtime = dtime(hour=7, minute=0)
    today_energy_loss_adjustment: int = 0
    forecast_energy_loss_per_day: int = 0
    ratio_family: str = "Baldr"
    ratio_primary_stat: str = "strength"
    ssl_combined_xanax_ecstasy_taken: int = 999
    fight_club_access: bool = False
    specialist_gym_target: str = "Auto from build"


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


@dataclass
class SpecialistGymProjection:
    gym_name: str
    parent_gym: str
    parent_unlocked: bool
    current_value: float
    required_value: float
    remaining_value: float
    estimated_unlock_at: Optional[datetime]
    requirement_text: str = ""


@dataclass
class SupportInventory:
    job_points_remaining: int
    fhc_remaining: int
    cans_remaining: int
    mcs_ready_claims: int
    next_mcs_ready_at: datetime


def company_10_star_activation_date(goal: GoalSettings) -> date:
    if goal.current_company_stars >= 10:
        return local_today()
    return goal.planned_10_star_date


def company_10_star_active_on(goal: GoalSettings, plan_day: date) -> bool:
    return plan_day >= company_10_star_activation_date(goal)


def mcs_next_ready_local(goal: GoalSettings) -> datetime:
    return datetime.combine(goal.mcs_next_ready_date, goal.mcs_next_ready_time).replace(tzinfo=APP_TIMEZONE)


def init_support_inventory(goal: GoalSettings) -> SupportInventory:
    return SupportInventory(
        job_points_remaining=max(0, int(goal.current_job_points)),
        fhc_remaining=max(0, int(goal.fhc_count_available)),
        cans_remaining=max(0, int(goal.can_count_available)),
        mcs_ready_claims=max(0, int(goal.mcs_ready_claims_now)),
        next_mcs_ready_at=mcs_next_ready_local(goal),
    )


def refresh_support_inventory_for_day(goal: GoalSettings, inventory: SupportInventory, plan_day: date) -> None:
    while inventory.next_mcs_ready_at.date() <= plan_day:
        inventory.mcs_ready_claims += 1
        inventory.next_mcs_ready_at = inventory.next_mcs_ready_at + timedelta(days=7)


def apply_energy_losses(goal: GoalSettings, plan_day: date, energy: int) -> int:
    adjusted = max(0, int(energy) - max(0, int(goal.forecast_energy_loss_per_day)))
    if plan_day == local_today():
        adjusted = max(0, adjusted - max(0, int(goal.today_energy_loss_adjustment)))
    return adjusted


def estimate_required_extra_energy_for_day(projected_state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers, plan_day: date) -> int:
    remaining_days = max(1, (goal.target_date - plan_day).days + 1)
    remaining_stats = max(0.0, goal.target_total_stats - projected_state.stats.total())
    if remaining_stats <= 0:
        return 0
    baseline_instruction = build_daily_instruction(projected_state, ratio, goal, plan_day, manual_mods)
    baseline_gain = float(baseline_instruction.estimated_gain)
    required_gain_per_day = remaining_stats / remaining_days
    extra_gain_needed = max(0.0, required_gain_per_day - baseline_gain)
    gain_per_energy = baseline_gain / max(1, int(baseline_instruction.estimated_energy)) if baseline_instruction.estimated_energy > 0 else 0.0
    if gain_per_energy <= 0:
        return 0
    return max(0, math.ceil(extra_gain_needed / gain_per_energy))


def booster_horizon_minutes(goal: GoalSettings) -> int:
    return max(0, int(round(float(goal.max_daily_booster_cooldown_hours) * 60)))


def booster_daily_capacity_minutes(goal: GoalSettings, booster_cd_minutes: int = 0) -> int:
    return max(0, booster_horizon_minutes(goal) - max(0, int(booster_cd_minutes)))



def choose_booster_mix(
    goal: GoalSettings,
    fhc_available: int,
    cans_available: int,
    required_energy: int,
    booster_cd_minutes: int = 0,
) -> Tuple[int, int, Optional[str], int]:
    fhc_cd = max(0, int(round(float(goal.fhc_cooldown_hours) * 60)))
    can_cd = max(0, int(round(float(goal.can_cooldown_hours) * 60)))
    fhc_energy = max(1, int(goal.fhc_effective_energy))
    can_energy = max(1, int(goal.can_energy_per_can))
    capacity = booster_daily_capacity_minutes(goal, booster_cd_minutes)

    best_choice: Optional[Tuple[int, int, Optional[str], int, int, int]] = None
    # tuple: (fhc_count, can_count, last_free, energy, overage, cooldown_used)
    for fhc_count in range(0, max(0, int(fhc_available)) + 1):
        for can_count in range(0, max(0, int(cans_available)) + 1):
            if fhc_count == 0 and can_count == 0:
                continue
            possible_last: List[Optional[str]] = []
            if fhc_count > 0:
                possible_last.append('FHC')
            if can_count > 0:
                possible_last.append('Energy can')
            for last_free in possible_last:
                used_minutes = fhc_cd * fhc_count + can_cd * can_count
                if last_free == 'FHC':
                    used_minutes -= fhc_cd
                elif last_free == 'Energy can':
                    used_minutes -= can_cd
                if used_minutes > capacity:
                    continue
                energy = fhc_count * fhc_energy + can_count * can_energy
                overage = max(0, energy - max(0, int(required_energy)))
                candidate = (fhc_count, can_count, last_free, energy, overage, used_minutes)
                if best_choice is None:
                    best_choice = candidate
                    continue
                _, _, _, best_energy, best_overage, best_used = best_choice
                req = max(0, int(required_energy))
                if energy >= req and best_energy < req:
                    best_choice = candidate
                elif energy >= req and best_energy >= req:
                    if overage < best_overage or (overage == best_overage and used_minutes < best_used) or (overage == best_overage and used_minutes == best_used and energy < best_energy):
                        best_choice = candidate
                elif energy < req and best_energy < req:
                    if energy > best_energy or (energy == best_energy and used_minutes < best_used):
                        best_choice = candidate

    if best_choice is None:
        return 0, 0, None, 0
    fhc_count, can_count, last_free, energy, _overage, _used = best_choice
    return fhc_count, can_count, last_free, energy



def allocate_support_energy(
    goal: GoalSettings,
    inventory: SupportInventory,
    plan_day: date,
    required_extra_energy: int,
    booster_cd_minutes: int = 0,
) -> Tuple[int, List[str], List[Tuple[str, int, int]]]:
    refresh_support_inventory_for_day(goal, inventory, plan_day)
    if required_extra_energy <= 0:
        return 0, [], []

    allocations: List[Tuple[str, int, int]] = []
    notes: List[str] = []
    total_added = 0
    remaining = required_extra_energy

    if inventory.mcs_ready_claims > 0 and remaining > 0:
        claim_energy = max(1, int(goal.mcs_energy_per_claim))
        claims_to_use = min(inventory.mcs_ready_claims, math.ceil(remaining / claim_energy))
        if claims_to_use > 0:
            energy_added = claims_to_use * claim_energy
            inventory.mcs_ready_claims -= claims_to_use
            total_added += energy_added
            remaining -= energy_added
            allocations.append(("MCS stock energy", claims_to_use, energy_added))
            notes.append(f"Use {claims_to_use} MCS stock claim(s) for {energy_added} extra energy.")

    if goal.use_job_points_energy and company_10_star_active_on(goal, plan_day) and remaining > 0:
        jp_available = max(0, inventory.job_points_remaining - max(0, int(goal.reserve_job_points)))
        daily_jp_cap = max(0, int(goal.job_points_daily_limit))
        jp_today = min(jp_available, daily_jp_cap)
        if jp_today > 0:
            jp_needed = math.ceil(remaining / max(1, int(goal.job_energy_per_point)))
            jp_use = min(jp_today, jp_needed)
            if jp_use > 0:
                energy_added = jp_use * int(goal.job_energy_per_point)
                inventory.job_points_remaining -= jp_use
                total_added += energy_added
                remaining -= energy_added
                allocations.append(("Job points", jp_use, energy_added))
                notes.append(f"Spend {jp_use} job points for {energy_added} extra energy once your company is 10★.")

    fhc_available = inventory.fhc_remaining if goal.fhc_allowed else 0
    cans_available = inventory.cans_remaining if goal.cans_allowed else 0
    if (fhc_available > 0 or cans_available > 0) and remaining > 0:
        booster_minutes = booster_cd_minutes if plan_day == local_today() else 0
        fhc_use, can_use, _last_free, booster_energy = choose_booster_mix(goal, fhc_available, cans_available, remaining, booster_cd_minutes=booster_minutes)
        if fhc_use > 0:
            energy_added = fhc_use * max(1, int(goal.fhc_effective_energy))
            inventory.fhc_remaining -= fhc_use
            total_added += energy_added
            remaining -= energy_added
            allocations.append(("FHC", fhc_use, energy_added))
            notes.append(f"Use {fhc_use} FHC(s) for about {energy_added} extra energy within your booster cooldown limit.")
        if can_use > 0:
            energy_added = can_use * max(1, int(goal.can_energy_per_can))
            inventory.cans_remaining -= can_use
            total_added += energy_added
            remaining -= energy_added
            allocations.append(("Energy can", can_use, energy_added))
            notes.append(f"Use {can_use} can(s) for about {energy_added} extra energy within your booster cooldown limit.")
        if booster_energy <= 0 and remaining > 0:
            notes.append("Booster items are capped by your daily booster cooldown limit, so the planner could not add more support energy today.")

    return total_added, notes, allocations



def total_support_energy_available_until_target(goal: GoalSettings) -> Tuple[int, int]:
    total = 0
    inventory = init_support_inventory(goal)
    days_to_target = max(1, (goal.target_date - local_today()).days + 1)
    for offset in range(days_to_target):
        plan_day = local_today() + timedelta(days=offset)
        refresh_support_inventory_for_day(goal, inventory, plan_day)
        total += inventory.mcs_ready_claims * int(goal.mcs_energy_per_claim)
        inventory.mcs_ready_claims = 0
        if goal.use_job_points_energy and company_10_star_active_on(goal, plan_day):
            jp_available = max(0, inventory.job_points_remaining - max(0, int(goal.reserve_job_points)))
            jp_today = min(jp_available, max(0, int(goal.job_points_daily_limit)))
            total += jp_today * int(goal.job_energy_per_point)
            inventory.job_points_remaining -= jp_today
        fhc_use, can_use, _last_free, booster_energy = choose_booster_mix(goal, inventory.fhc_remaining if goal.fhc_allowed else 0, inventory.cans_remaining if goal.cans_allowed else 0, 10**9, booster_cd_minutes=0)
        total += booster_energy
        inventory.fhc_remaining -= fhc_use
        inventory.cans_remaining -= can_use
    return total, max(0, int(total / days_to_target))


def build_gym_db() -> List[Gym]:
    return [
        Gym("Premier Fitness", "light", 5, 10, 200, {"strength": 2.0, "speed": 2.0, "defense": 2.0, "dexterity": 2.0}),
        Gym("Average Joes", "light", 5, 100, 500, {"strength": 2.4, "speed": 2.4, "defense": 2.8, "dexterity": 2.4}),
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
        Gym("Balboas Gym", "specialist", 25, 50_000_000, None, {"strength": 0.0, "speed": 0.0, "defense": 7.5, "dexterity": 7.5}),
        Gym("Frontline Fitness", "specialist", 25, 50_000_000, None, {"strength": 7.5, "speed": 7.5, "defense": 0.0, "dexterity": 0.0}),
        Gym("Gym 3000", "specialist", 50, 100_000_000, None, {"strength": 8.0, "speed": 0.0, "defense": 0.0, "dexterity": 0.0}),
        Gym("Mr. Isoyamas", "specialist", 50, 100_000_000, None, {"strength": 0.0, "speed": 0.0, "defense": 8.0, "dexterity": 0.0}),
        Gym("Total Rebound", "specialist", 50, 100_000_000, None, {"strength": 0.0, "speed": 8.0, "defense": 0.0, "dexterity": 0.0}),
        Gym("Elites", "specialist", 50, 100_000_000, None, {"strength": 0.0, "speed": 0.0, "defense": 0.0, "dexterity": 8.0}),
        Gym("The Sports Science Lab", "specialist", 25, 500_000_000, None, {"strength": 9.0, "speed": 9.0, "defense": 9.0, "dexterity": 9.0}),
        Gym("Fight Club", "specialist", 10, 2_147_483_647, None, {"strength": 10.0, "speed": 10.0, "defense": 10.0, "dexterity": 10.0}),
    ]


GYM_DB = build_gym_db()
GYM_INDEX = {gym.name: gym for gym in GYM_DB}
BALBOAS_GYM_NAME = "Balboas Gym"
FRONTLINE_GYM_NAME = "Frontline Fitness"
GYM3000_NAME = "Gym 3000"
ISOYAMAS_NAME = "Mr. Isoyamas"
TOTAL_REBOUND_NAME = "Total Rebound"
ELITES_NAME = "Elites"
SSL_GYM_NAME = "The Sports Science Lab"
FIGHT_CLUB_NAME = "Fight Club"
LINEAR_GYM_NAMES = [gym.name for gym in GYM_DB if gym.tier != "specialist"]
SPECIALIST_GYM_NAMES = [gym.name for gym in GYM_DB if gym.tier == "specialist"]
AUTO_SPECIALIST_GYM_NAMES = [
    BALBOAS_GYM_NAME,
    FRONTLINE_GYM_NAME,
    GYM3000_NAME,
    ISOYAMAS_NAME,
    TOTAL_REBOUND_NAME,
    ELITES_NAME,
    SSL_GYM_NAME,
]
SPECIALIST_PARENT_MAP = {
    BALBOAS_GYM_NAME: "Cha Cha's",
    FRONTLINE_GYM_NAME: "Cha Cha's",
    GYM3000_NAME: "George's",
    ISOYAMAS_NAME: "George's",
    TOTAL_REBOUND_NAME: "George's",
    ELITES_NAME: "George's",
    SSL_GYM_NAME: "Last Round",
    FIGHT_CLUB_NAME: "George's",
}
SPECIALIST_REQUIREMENT_TEXT = {
    BALBOAS_GYM_NAME: "Defense + Dexterity must be at least 25% higher than Strength + Speed.",
    FRONTLINE_GYM_NAME: "Strength + Speed must be at least 25% higher than Dexterity + Defense.",
    GYM3000_NAME: "Strength must be at least 25% higher than your second-highest stat.",
    ISOYAMAS_NAME: "Defense must be at least 25% higher than your second-highest stat.",
    TOTAL_REBOUND_NAME: "Speed must be at least 25% higher than your second-highest stat.",
    ELITES_NAME: "Dexterity must be at least 25% higher than your second-highest stat.",
    SSL_GYM_NAME: "Last Round must be unlocked and your lifetime Xanax + Ecstasy total must be 150 or less.",
    FIGHT_CLUB_NAME: "Membership is invite-only and the unlock requirement is not publicly documented.",
}


def ordered_gym_names() -> List[str]:
    return [gym.name for gym in GYM_DB]


def linear_gym_names() -> List[str]:
    return list(LINEAR_GYM_NAMES)


def specialist_progress_snapshot(gym_name: str, stats: PlayerStats, goal: Optional[GoalSettings] = None) -> Tuple[float, float, float, bool]:
    goal = goal or GoalSettings()
    if gym_name == FRONTLINE_GYM_NAME:
        current_value = stats.strength + stats.speed
        required_value = max(0.0, 1.25 * (stats.dexterity + stats.defense))
        remaining_value = max(0.0, required_value - current_value)
        return current_value, required_value, remaining_value, current_value >= required_value and required_value > 0
    if gym_name == BALBOAS_GYM_NAME:
        current_value = stats.defense + stats.dexterity
        required_value = max(0.0, 1.25 * (stats.strength + stats.speed))
        remaining_value = max(0.0, required_value - current_value)
        return current_value, required_value, remaining_value, current_value >= required_value and required_value > 0
    if gym_name in {GYM3000_NAME, ISOYAMAS_NAME, TOTAL_REBOUND_NAME, ELITES_NAME}:
        target_stat = {
            GYM3000_NAME: "strength",
            ISOYAMAS_NAME: "defense",
            TOTAL_REBOUND_NAME: "speed",
            ELITES_NAME: "dexterity",
        }[gym_name]
        ordered_values = sorted(stats.as_dict().items(), key=lambda kv: kv[1], reverse=True)
        current_value = stats.get(target_stat)
        second_highest = next((value for key, value in ordered_values if key != target_stat), 0.0)
        required_value = max(0.0, 1.25 * second_highest)
        remaining_value = max(0.0, required_value - current_value)
        return current_value, required_value, remaining_value, current_value >= required_value and required_value > 0
    if gym_name == SSL_GYM_NAME:
        used_total = max(0, int(getattr(goal, "ssl_combined_xanax_ecstasy_taken", 999)))
        current_value = max(0.0, float(150 - used_total))
        required_value = 150.0
        remaining_value = max(0.0, float(used_total - 150))
        return current_value, required_value, remaining_value, used_total <= 150
    if gym_name == FIGHT_CLUB_NAME:
        enabled = bool(getattr(goal, "fight_club_access", False))
        return float(1 if enabled else 0), 1.0, float(0 if enabled else 1), enabled
    return 0.0, 1.0, 1.0, False


def frontline_progress_values(stats: PlayerStats) -> Tuple[float, float, float]:
    current_value, required_value, remaining_value, _ = specialist_progress_snapshot(FRONTLINE_GYM_NAME, stats)
    return current_value, required_value, remaining_value


def specialist_parent_unlocked(unlocked_names: List[str], gym_name: str) -> bool:
    parent = SPECIALIST_PARENT_MAP.get(gym_name)
    return True if not parent else parent in unlocked_names


def specialist_is_available(unlocked_names: List[str], stats: PlayerStats, gym_name: str, goal: Optional[GoalSettings] = None) -> bool:
    if not specialist_parent_unlocked(unlocked_names, gym_name):
        return False
    _current_value, _required_value, _remaining_value, met = specialist_progress_snapshot(gym_name, stats, goal)
    return met


def frontline_is_unlocked_for_stats(unlocked_names: List[str], stats: PlayerStats, goal: Optional[GoalSettings] = None) -> bool:
    return specialist_is_available(unlocked_names, stats, FRONTLINE_GYM_NAME, goal)


def active_unlocked_names_for_stats(base_unlocked_names: List[str], stats: PlayerStats, highest_idx: Optional[int] = None, goal: Optional[GoalSettings] = None) -> List[str]:
    if highest_idx is None:
        linear_names = [name for name in base_unlocked_names if name in LINEAR_GYM_NAMES]
    else:
        linear_names = unlocked_names_through_index(highest_idx)
    names = set(linear_names)
    effective_goal = goal or (st.session_state.get("goal_settings") if getattr(st, "session_state", None) is not None else None)
    for specialist in AUTO_SPECIALIST_GYM_NAMES:
        if specialist_is_available(list(names), stats, specialist, effective_goal):
            names.add(specialist)
    if effective_goal and bool(getattr(effective_goal, "fight_club_access", False)):
        names.add(FIGHT_CLUB_NAME)
    return [name for name in ordered_gym_names() if name in names]


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




def _find_first_bool_for_key(value: Any, target_key: str) -> Optional[bool]:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() == target_key.lower():
                if isinstance(child, bool):
                    return child
                if isinstance(child, (int, float)):
                    return bool(child)
                if isinstance(child, str):
                    lowered = child.strip().lower()
                    if lowered in {"true", "yes", "1", "available", "ready"}:
                        return True
                    if lowered in {"false", "no", "0", "used", "unavailable"}:
                        return False
            found = _find_first_bool_for_key(child, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_first_bool_for_key(child, target_key)
            if found is not None:
                return found
    return None


def _find_first_dict_matching(value: Any, predicate) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        try:
            if predicate(value):
                return value
        except Exception:
            pass
        for child in value.values():
            found = _find_first_dict_matching(child, predicate)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_first_dict_matching(child, predicate)
            if found is not None:
                return found
    return None


def _parse_company_star_count(job_payload: Dict[str, Any]) -> Optional[int]:
    value = _first_present(
        job_payload,
        [
            ("job", "company", "stars"),
            ("company", "stars"),
            ("job", "stars"),
            ("stars",),
        ],
        default=None,
    )
    if value is None:
        value = _find_first_numeric_for_key(job_payload, "stars")
    if value is None:
        return None
    return max(0, min(10, _safe_int(value, 0)))


def _parse_job_points_payload(jobpoints_payload: Dict[str, Any]) -> Optional[int]:
    value = _first_present(
        jobpoints_payload,
        [
            ("jobpoints",),
            ("job_points",),
            ("points",),
            ("job", "points"),
            ("company", "jobpoints"),
        ],
        default=None,
    )
    if value is None:
        value = _find_first_numeric_for_key(jobpoints_payload, "jobpoints")
    if value is None:
        value = _find_first_numeric_for_key(jobpoints_payload, "points")
    if value is None:
        return None
    return max(0, _safe_int(value, 0))


def _parse_refill_state(refills_payload: Dict[str, Any]) -> Tuple[Optional[bool], List[str]]:
    notes: List[str] = []
    candidates = [
        ("energy", "available"),
        ("refills", "energy", "available"),
        ("energy_refill", "available"),
        ("energy_refill_available",),
    ]
    available = None
    for path in candidates:
        value = _first_present(refills_payload, [path], default=None)
        if value is not None:
            if isinstance(value, bool):
                available = value
                break
            if isinstance(value, (int, float)):
                available = bool(value)
                break
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1", "available", "ready"}:
                    available = True
                    break
                if lowered in {"false", "no", "0", "used", "unavailable"}:
                    available = False
                    break
    if available is None:
        used = _first_present(
            refills_payload,
            [
                ("energy", "used"),
                ("refills", "energy", "used"),
                ("energy_refill", "used"),
                ("energy_refill_used",),
            ],
            default=None,
        )
        if isinstance(used, bool):
            available = not used
        elif isinstance(used, (int, float)):
            available = not bool(used)
    if available is None:
        recursive = _find_first_bool_for_key(refills_payload, "available")
        if recursive is not None:
            available = recursive
    if available is None:
        notes.append("Refill endpoint synced, but energy refill availability could not be read cleanly; keeping your manual refill setting.")
    return available, notes


def _parse_mcs_support(stocks_payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[datetime], List[str]]:
    notes: List[str] = []
    mcs_entry = _find_first_dict_matching(
        stocks_payload,
        lambda obj: any(
            token in " ".join(_extract_strings(obj)).upper() for token in ["MCS", "MC SMOOGLE", "MC SMOOGLE CORP"]
        ) or str(obj.get("ticker", "")).upper() == "MCS" or str(obj.get("acronym", "")).upper() == "MCS" or str(obj.get("symbol", "")).upper() == "MCS",
    )
    if mcs_entry is None:
        notes.append("Stocks endpoint synced, but no MCS holding entry was found; keeping your manual MCS settings.")
        return None, None, notes

    ready_claims = None
    for path in [
        ("available",),
        ("ready",),
        ("claims_ready",),
        ("rewards_ready",),
        ("benefits_ready",),
        ("reward_count",),
    ]:
        value = _first_present(mcs_entry, [path], default=None)
        if value is not None:
            ready_claims = max(0, _safe_int(value, 0))
            break
    if ready_claims is None:
        for key in ["available", "ready", "claims_ready", "rewards_ready", "benefits_ready", "reward_count"]:
            value = _find_first_numeric_for_key(mcs_entry, key)
            if value is not None:
                ready_claims = max(0, _safe_int(value, 0))
                break

    next_ready_dt: Optional[datetime] = None
    ts_value = None
    for path in [
        ("next_claim_at",),
        ("next_ready_at",),
        ("ready_at",),
        ("available_at",),
        ("benefit", "next_claim_at"),
    ]:
        ts_value = _first_present(mcs_entry, [path], default=None)
        if ts_value is not None:
            break
    if ts_value is None:
        for key in ["next_claim_at", "next_ready_at", "ready_at", "available_at"]:
            ts_value = _find_first_numeric_for_key(mcs_entry, key)
            if ts_value is not None:
                break
    if ts_value is not None:
        ts_int = _safe_int(ts_value, 0)
        if ts_int > 0:
            try:
                next_ready_dt = datetime.fromtimestamp(ts_int, tz=TORN_TIMEZONE).astimezone(APP_TIMEZONE)
            except Exception:
                next_ready_dt = None

    if ready_claims is None and next_ready_dt is None:
        notes.append("MCS stock entry was found, but the API response did not expose ready-claim timing in a shape the planner could read; keeping your manual MCS settings.")
    return ready_claims, next_ready_dt, notes


def auto_sync_goal_settings_from_api(api_key: str, goal: GoalSettings) -> Tuple[GoalSettings, List[str]]:
    notes: List[str] = []
    updated = GoalSettings(**goal.__dict__)

    payloads: Dict[str, Dict[str, Any]] = {}
    for endpoint in ["/user/job", "/user/jobpoints", "/user/refills", "/user/stocks"]:
        try:
            payloads[endpoint] = _api_get(endpoint, api_key)
        except TornAPIError as exc:
            notes.append(f"{endpoint} auto-sync skipped: {exc}")

    job_payload = payloads.get("/user/job", {})
    stars = _parse_company_star_count(job_payload) if job_payload else None
    if stars is not None:
        updated.current_company_stars = stars
        notes.append(f"Auto-synced company stars: {stars}.")

    jobpoints_payload = payloads.get("/user/jobpoints", {})
    job_points = _parse_job_points_payload(jobpoints_payload) if jobpoints_payload else None
    if job_points is not None:
        updated.current_job_points = job_points
        notes.append(f"Auto-synced job points: {job_points}.")

    refills_payload = payloads.get("/user/refills", {})
    if refills_payload:
        refill_available, refill_notes = _parse_refill_state(refills_payload)
        notes.extend(refill_notes)
        if refill_available is not None:
            updated.daily_refill_available_now = bool(refill_available)
            updated.daily_refill_used_today = not bool(refill_available)
            notes.append("Auto-synced daily refill availability.")

    stocks_payload = payloads.get("/user/stocks", {})
    if stocks_payload:
        ready_claims, next_ready_dt, stock_notes = _parse_mcs_support(stocks_payload)
        notes.extend(stock_notes)
        if ready_claims is not None:
            updated.mcs_ready_claims_now = ready_claims
            notes.append(f"Auto-synced MCS ready claims: {ready_claims}.")
        if next_ready_dt is not None:
            updated.mcs_next_ready_date = next_ready_dt.date()
            updated.mcs_next_ready_time = next_ready_dt.timetz().replace(tzinfo=None)
            notes.append(f"Auto-synced next MCS ready time: {fmt_local(next_ready_dt)}.")

    notes.append("FHC and energy-can counts remain manual for now: the current official v2 Swagger does not expose a stable /user/inventory selection, and Torn's API changes thread on March 12, 2026 said inventory was still being worked on.")
    return updated, notes


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
    today_utc = local_today()
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
        faction_war_days=[local_today() + timedelta(days=2)],
        training_modifiers=TrainingModifiers(
            all_gym_gains_pct=3.0,
            happy_loss_reduction_pct=50.0,
            detected_sources=["Demo Fitness Center (+3% gym gains, -50% gym happy loss)"],
        ),
        api_notes=["Demo data loaded.", "Unlocked gyms are demo values."],
        last_sync=local_now(),
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
            if path_name == "/faction/upgrades" and "Incorrect ID-entity relation" in str(exc):
                pass
            else:
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
        last_sync=local_now(),
    )


def highest_unlocked_gym_index(state: PlayerState) -> int:
    indices = [LINEAR_GYM_NAMES.index(name) for name in state.unlocked_gyms if name in LINEAR_GYM_NAMES]
    return max(indices) if indices else 0


def unlocked_names_through_index(highest_idx: int) -> List[str]:
    names = linear_gym_names()
    highest_idx = max(0, min(highest_idx, len(names) - 1))
    return names[: highest_idx + 1]


def next_gym_name_for_index(highest_idx: int) -> Optional[str]:
    names = linear_gym_names()
    if highest_idx + 1 < len(names):
        return names[highest_idx + 1]
    return None


def next_gym_threshold_for_index(highest_idx: int) -> Optional[int]:
    names = linear_gym_names()
    if 0 <= highest_idx < len(names):
        gym_name = names[highest_idx]
        return GYM_INDEX[gym_name].e_for_next_gym
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
    now_dt = now_dt or local_now()
    blocks: List[Tuple[datetime, int, str]] = []

    if state.recovery.current_energy > 0:
        blocks.append((now_dt, int(state.recovery.current_energy), 'current energy'))

    if state.recovery.daily_refill_enabled:
        refill_time = next_daily_refill_ready_local(goal, now_dt, after_dt=now_dt + timedelta(minutes=10))
        if refill_time.date() == now_dt.date():
            refill_source = 'daily refill' if not getattr(goal, 'daily_refill_used_today', True) else 'daily refill reset (TST midnight)'
            blocks.append((refill_time, int(state.recovery.refill_energy), refill_source))

    eod = end_of_day(now_dt)
    natural_e = natural_energy_between(state, mods, now_dt, eod)
    if natural_e > 0:
        blocks.append((eod, natural_e, 'natural regen through end of day'))

    drug_clear_dt = now_dt + timedelta(minutes=max(0, state.recovery.drug_cd_minutes))
    if drug_clear_dt.date() == now_dt.date() and state.recovery.drug_cd_minutes > 0:
        blocks.append((drug_clear_dt, int(state.recovery.xanax_energy), 'next xanax after cooldown'))

    return blocks


def estimate_today_unlock_from_blocks(state: PlayerState, goal: GoalSettings, mods: TrainingModifiers, now_dt: Optional[datetime] = None) -> Optional[Tuple[datetime, str]]:
    now_dt = now_dt or local_now()
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
    support_bonus_energy: int = 0,
    support_notes: Optional[List[str]] = None,
) -> Tuple[DailyInstruction, PlayerStats, int, int, Optional[datetime]]:
    combined_mods = state.training_modifiers.merge(manual_mods)
    projected_unlocked = active_unlocked_names_for_stats(state.unlocked_gyms, state.stats, highest_idx, goal)
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
    if jump_plan is not None and day_type in {'prep', 'happy_jump', 'super_happy_jump'}:
        target_stat = jump_plan.target_stat
    else:
        target_stat = choose_target_stat(projected_state.stats, ratio)

    if day_type == 'war':
        instruction = DailyInstruction(plan_day, 'war', 'none', 'none', 0, 0, 0.0, 0, 0, ['Faction war day. Training skipped in baseline planner.'])
        return instruction, state.stats, highest_idx, progress_e, None

    day_energy = energy_budget_for_day(projected_state, goal, plan_day, combined_mods, day_type) + max(0, int(support_bonus_energy))
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
    time_cursor = datetime.combine(plan_day, dtime(hour=8, minute=0), tzinfo=APP_TIMEZONE)
    if jump_plan is not None and plan_day == jump_plan.execute_at.date():
        time_cursor = jump_plan.execute_at

    while energy_left > 0:
        unlocked_names = active_unlocked_names_for_stats(state.unlocked_gyms, current_stats, highest_idx, goal)
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

        post_segment_names = active_unlocked_names_for_stats(state.unlocked_gyms, current_stats, highest_idx, goal)
        if FRONTLINE_GYM_NAME in post_segment_names and FRONTLINE_GYM_NAME not in unlocked_names:
            specialist_unlock_time = estimate_segment_end_time(time_cursor, segment_energy, day_energy)
            unlock_notes.append(f'Projected specialist unlock during day: {FRONTLINE_GYM_NAME} at about {specialist_unlock_time.strftime("%H:%M")}.')

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
        f'Expected happy loss per train: {expected_happy_loss_per_train((best_gym_for_stat_from_names(active_unlocked_names_for_stats(state.unlocked_gyms, current_stats, highest_idx, goal), target_stat) or GYM_DB[0]).energy_cost, combined_mods)}.',
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
    if support_notes:
        notes.extend(support_notes)
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





def choose_99k_target_stat(stats: PlayerStats) -> str:
    """For 99k jumps, prioritize the highest current stat for max raw stat gain."""
    stat_map = stats.as_dict()
    return max(stat_map, key=stat_map.get)

def get_unlocked_gyms(state: PlayerState, goal: Optional[GoalSettings] = None) -> List[Gym]:
    active_names = active_unlocked_names_for_stats(state.unlocked_gyms, state.stats, highest_unlocked_gym_index(state), goal)
    return [GYM_INDEX[name] for name in active_names if name in GYM_INDEX]


def best_gym_for_stat(state: PlayerState, stat_key: str, goal: Optional[GoalSettings] = None) -> Optional[Gym]:
    candidates = [gym for gym in get_unlocked_gyms(state, goal) if gym.gain_for(stat_key) > 0]
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
    if plan_day == local_today() and state.recovery.current_happy > 0:
        return state.recovery.current_happy
    if state.recovery.max_happy > 0:
        return state.recovery.max_happy
    return goal.normal_day_start_happy



def parse_manual_99k_schedule_text(schedule_text: str) -> List[datetime]:
    entries: List[datetime] = []
    seen: set[str] = set()
    for raw_line in (schedule_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = line.replace("T", " ")
        parsed: Optional[datetime] = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
            try:
                parsed = datetime.strptime(normalized, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            continue
        parsed = parsed.replace(tzinfo=APP_TIMEZONE)
        key = parsed.isoformat()
        if key not in seen:
            seen.add(key)
            entries.append(parsed)
    return sorted(entries)


def serialize_manual_99k_schedule(datetimes: List[datetime]) -> str:
    normalized = sorted({to_local(dt).replace(second=0, microsecond=0) for dt in datetimes})
    return "\n".join(dt.strftime("%Y-%m-%d %H:%M") for dt in normalized)


def manual_99k_schedule_datetimes(goal: GoalSettings) -> List[datetime]:
    parsed = parse_manual_99k_schedule_text(goal.manual_99k_jump_schedule_text)
    if parsed:
        return parsed
    if goal.schedule_99k_jump:
        fallback = datetime.combine(goal.scheduled_99k_jump_date, goal.scheduled_99k_jump_time).replace(tzinfo=APP_TIMEZONE)
        return [fallback]
    return []


def selected_99k_execute_at(goal: GoalSettings) -> Optional[datetime]:
    now_dt = local_now()
    for dt in manual_99k_schedule_datetimes(goal):
        if to_local(dt) >= now_dt:
            return to_local(dt)
    return None


def scheduled_99k_execute_at_for_day(goal: GoalSettings, plan_day: date) -> Optional[datetime]:
    for dt in manual_99k_schedule_datetimes(goal):
        dt_local = to_local(dt)
        if dt_local.date() == plan_day:
            return dt_local
    return None


def build_specific_jump_plan(
    state: PlayerState,
    ratio: RatioProfile,
    goal: GoalSettings,
    mods: TrainingModifiers,
    jump_type: str,
    execute_at: datetime,
    manual_selected: bool = False,
) -> Optional[JumpPlan]:
    target_stat = choose_99k_target_stat(state.stats) if jump_type == "super_happy_jump" else choose_target_stat(state.stats, ratio)
    gym = best_gym_for_stat(state, target_stat, goal)
    if gym is None:
        return None

    normal_energy = planner_baseline_energy_per_day(state, goal, mods)
    normal_start_happy = max(state.recovery.max_happy, goal.normal_day_start_happy)
    normal_sim = simulate_training_block(state.stats, target_stat, gym, normal_energy, normal_start_happy, mods)

    jump_energy = goal.jump_stack_energy_target + (state.recovery.refill_energy if state.recovery.daily_refill_enabled else 0)
    chosen_gain = 0.0
    if jump_type == "happy_jump":
        chosen_gain = float(simulate_training_block(state.stats, target_stat, gym, jump_energy, goal.happy_jump_start_happy, mods)["total_gain"])
    else:
        chosen_gain = float(simulate_training_block(state.stats, target_stat, gym, jump_energy, goal.super_happy_jump_start_happy, mods)["total_gain"])

    prep_start = to_local(execute_at) - timedelta(hours=goal.jump_prep_hours)
    xanax_times = planned_xanax_stack_times(state, goal, to_local(execute_at))
    final_cd_clear = xanax_times[-1] + timedelta(hours=float(goal.assumed_xanax_cooldown_hours))
    notes: List[str] = []
    if manual_selected and jump_type == "super_happy_jump":
        notes.append("99k jump date/time was manually selected by you.")
    if final_cd_clear > to_local(execute_at):
        notes.append("Warning: current drug cooldown and stack timing push the final Xanax cooldown past the planned jump window.")
    if state.recovery.booster_cd_minutes > 0 and local_now() + timedelta(minutes=state.recovery.booster_cd_minutes) > to_local(execute_at):
        notes.append("Warning: booster cooldown is still active beyond the planned jump window.")
    if to_local(execute_at).date() in state.faction_war_days and not goal.allow_jump_on_war_days:
        notes.append("Warning: this planned jump falls on a war day.")
    notes.extend([
        f"Planner reserves about {goal.jump_prep_hours:.0f} hours of prep time for the jump.",
        f"Planner assumes {goal.jump_stack_xanax_uses} Xanax with roughly {goal.assumed_xanax_cooldown_hours:.1f} hours between doses.",
        "Use the jump just after a 15-minute happy reset mark.",
        "Train immediately after applying happy items / ecstasy, then use daily refill.",
    ])
    return JumpPlan(
        jump_type=jump_type,
        target_stat=target_stat,
        gym_name=gym.name,
        prep_start=prep_start,
        execute_at=to_local(execute_at),
        projected_normal_gain=float(normal_sim["total_gain"]),
        projected_jump_gain=chosen_gain,
        projected_gain_delta=float(chosen_gain - normal_sim["total_gain"]),
        notes=notes,
    )


def next_viable_happy_jump_window(state: PlayerState, goal: GoalSettings) -> datetime:
    now_dt = local_now()
    prep_ready = now_dt + timedelta(minutes=max(state.recovery.drug_cd_minutes, state.recovery.booster_cd_minutes))
    candidate = next_awake_quarter_hour(goal, prep_ready + timedelta(hours=goal.jump_prep_hours)) if getattr(goal, "sleep_schedule_enabled", False) else next_quarter_hour(prep_ready + timedelta(hours=goal.jump_prep_hours))
    while (not goal.allow_jump_on_war_days) and (candidate.date() in state.faction_war_days):
        candidate = next_awake_quarter_hour(goal, candidate + timedelta(days=1)) if getattr(goal, "sleep_schedule_enabled", False) else next_quarter_hour(candidate + timedelta(days=1))
    return candidate


def planned_xanax_stack_times(state: PlayerState, goal: GoalSettings, execute_at: datetime) -> List[datetime]:
    execute_at = to_local(execute_at)
    first_candidate = execute_at - timedelta(hours=goal.jump_prep_hours)
    earliest_allowed = local_now() + timedelta(minutes=max(0, state.recovery.drug_cd_minutes))
    first = max(first_candidate, earliest_allowed)
    first = schedule_action_time(goal, first) if getattr(goal, "sleep_schedule_enabled", False) else first
    times = [first]
    for _ in range(1, int(goal.jump_stack_xanax_uses)):
        next_time = times[-1] + timedelta(hours=float(goal.assumed_xanax_cooldown_hours))
        next_time = schedule_action_time(goal, next_time) if getattr(goal, "sleep_schedule_enabled", False) else next_time
        times.append(next_time)
    return times



def build_jump_plan(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, mods: TrainingModifiers) -> Optional[JumpPlan]:
    manual_99k_dt = selected_99k_execute_at(goal)
    if manual_99k_dt is not None:
        return build_specific_jump_plan(state, ratio, goal, mods, "super_happy_jump", manual_99k_dt, manual_selected=True)

    target_stat = choose_target_stat(state.stats, ratio)
    gym = best_gym_for_stat(state, target_stat, goal)
    if gym is None:
        return None

    normal_energy = planner_baseline_energy_per_day(state, goal, mods)
    normal_start_happy = max(state.recovery.max_happy, goal.normal_day_start_happy)
    normal_sim = simulate_training_block(state.stats, target_stat, gym, normal_energy, normal_start_happy, mods)

    jump_energy = goal.jump_stack_energy_target + (state.recovery.refill_energy if state.recovery.daily_refill_enabled else 0)
    happy_sim = simulate_training_block(state.stats, target_stat, gym, jump_energy, goal.happy_jump_start_happy, mods)
    threshold = 1 + goal.jump_min_extra_gain_pct / 100.0

    if goal.auto_schedule_happy_jumps and happy_sim["total_gain"] > normal_sim["total_gain"] * threshold:
        execute_at = next_viable_happy_jump_window(state, goal)
        return build_specific_jump_plan(state, ratio, goal, mods, "happy_jump", execute_at, manual_selected=False)
    return None


def build_jump_sequence(state: PlayerState, goal: GoalSettings, jump_plan: JumpPlan) -> List[JumpStep]:
    steps: List[JumpStep] = []
    xanax_times = planned_xanax_stack_times(state, goal, jump_plan.execute_at)

    if state.recovery.current_energy > 0 and local_today() <= jump_plan.execute_at.date():
        steps.append(JumpStep(local_now(), "Spend current energy", f"Use your current {state.recovery.current_energy} energy now unless you are already in the final save-for-jump window."))

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
    manual_99k_dt = scheduled_99k_execute_at_for_day(goal, on_day)
    if manual_99k_dt is not None:
        jump_plan = build_specific_jump_plan(state, ratio, goal, mods, "super_happy_jump", manual_99k_dt, manual_selected=True)
        if jump_plan is not None:
            return "super_happy_jump", jump_plan

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
    if plan_day == local_today() and day_type not in {"happy_jump", "super_happy_jump"}:
        blocks = build_today_energy_blocks(state, goal, mods, local_now())
        reachable = max(0, sum(energy for _when, energy, _source in blocks))
        return apply_energy_losses(goal, plan_day, reachable)

    if day_type == "prep":
        return apply_energy_losses(goal, plan_day, min(state.recovery.max_energy, max(state.recovery.current_energy, 0)))
    if day_type in {"happy_jump", "super_happy_jump"}:
        jump_energy = goal.jump_stack_energy_target + (state.recovery.refill_energy if state.recovery.daily_refill_enabled else 0)
        return apply_energy_losses(goal, plan_day, jump_energy)
    baseline = planner_baseline_energy_per_day(state, goal, mods)
    return apply_energy_losses(goal, plan_day, baseline)


def build_daily_instruction(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, plan_day: date, manual_mods: TrainingModifiers) -> DailyInstruction:
    combined_mods = state.training_modifiers.merge(manual_mods)
    day_type, jump_plan = day_type_for_date(state, ratio, goal, plan_day, combined_mods)
    if jump_plan is not None and day_type in {'prep', 'happy_jump', 'super_happy_jump'}:
        target_stat = jump_plan.target_stat
        gym = GYM_INDEX.get(jump_plan.gym_name) or best_gym_for_stat(state, target_stat, goal)
    else:
        target_stat = choose_target_stat(state.stats, ratio)
        gym = best_gym_for_stat(state, target_stat, goal)

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
        notes.append(f"Next jump window: {fmt_local(jump_plan.execute_at)}.")
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
    support_inventory = init_support_inventory(goal)

    for offset in range(days):
        plan_day = local_today() + timedelta(days=offset)
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
        extra_energy_needed = estimate_required_extra_energy_for_day(projected_state, ratio, goal, manual_mods, plan_day)
        support_bonus, support_notes, _allocs = allocate_support_energy(goal, support_inventory, plan_day, extra_energy_needed, booster_cd_minutes=projected_state.recovery.booster_cd_minutes if plan_day == local_today() else 0)
        instruction, projected_stats, highest_idx, progress_e, _unlock_time = simulate_day_with_unlocks(
            projected_state, ratio, goal, plan_day, manual_mods, highest_idx, progress_e, support_bonus_energy=support_bonus, support_notes=support_notes
        )
        plan.append(instruction)

    return plan



def days_until_goal_estimate(state: PlayerState, goal: GoalSettings, manual_mods: TrainingModifiers) -> Optional[int]:
    remaining = goal.target_total_stats - state.stats.total()
    if remaining <= 0:
        return 0

    target_stat = choose_target_stat(state.stats, RatioProfile())
    gym = best_gym_for_stat(state, target_stat, goal)
    if gym is None:
        return None

    combined_mods = state.training_modifiers.merge(manual_mods)
    start_happy = max(state.recovery.max_happy, goal.normal_day_start_happy)
    baseline_energy = apply_energy_losses(goal, local_today(), planner_baseline_energy_per_day(state, goal, combined_mods))
    sim = simulate_training_block(state.stats, target_stat, gym, baseline_energy, start_happy, combined_mods)
    est_daily_gain = float(sim["total_gain"])

    target_days = max(1, (goal.target_date - local_today()).days)
    manual_99k_dts = [dt for dt in manual_99k_schedule_datetimes(goal) if local_today() <= to_local(dt).date() <= goal.target_date]
    if manual_99k_dts:
        jump_plan = build_specific_jump_plan(state, RatioProfile(), goal, combined_mods, "super_happy_jump", manual_99k_dts[0], manual_selected=True)
        if jump_plan is not None:
            est_daily_gain += max(0.0, jump_plan.projected_gain_delta) * len(manual_99k_dts) / target_days
    else:
        jump_plan = build_jump_plan(state, RatioProfile(), goal, combined_mods)
        if jump_plan is not None:
            est_daily_gain += max(0.0, jump_plan.projected_gain_delta) / 7.0

    total_support_energy, avg_support_per_day = total_support_energy_available_until_target(goal)
    est_daily_gain += max(0, avg_support_per_day) * (sim["total_gain"] / max(1, baseline_energy))

    if est_daily_gain <= 0:
        return None

    return math.ceil(remaining / est_daily_gain)


def init_state() -> None:
    if "_notified_events" not in st.session_state:
        st.session_state._notified_events = []
    if "player_state" not in st.session_state:
        st.session_state.player_state = None
    if "goal_settings" not in st.session_state:
        st.session_state.goal_settings = GoalSettings()
    if "ratio_profile" not in st.session_state:
        st.session_state.ratio_profile = RatioProfile()
    if "manual_mods" not in st.session_state:
        st.session_state.manual_mods = TrainingModifiers()
    if "manual_unlocked_gyms" not in st.session_state:
        st.session_state.manual_unlocked_gyms = []
    if "gym_multiselect" not in st.session_state:
        st.session_state.gym_multiselect = []
    if "selected_calendar_date" not in st.session_state:
        st.session_state.selected_calendar_date = None
    if "highest_unlocked_gym_selector" not in st.session_state:
        st.session_state.highest_unlocked_gym_selector = "-- none --"
    if "manual_99k_jump_date" not in st.session_state:
        st.session_state.manual_99k_jump_date = local_today() + timedelta(days=7)
    if "manual_99k_jump_entries" not in st.session_state:
        st.session_state.manual_99k_jump_entries = manual_99k_schedule_datetimes(st.session_state.goal_settings)
    if "preview_days" not in st.session_state:
        st.session_state.preview_days = 30
    if "display_timezone_name" not in st.session_state:
        st.session_state.display_timezone_name = DEFAULT_APP_TIMEZONE_NAME
    set_app_timezone(st.session_state.display_timezone_name)
    if "use_tct_times" not in st.session_state:
        st.session_state.use_tct_times = False
    if "sleep_schedule_enabled" not in st.session_state:
        st.session_state.sleep_schedule_enabled = False
    if "sleep_start_time" not in st.session_state:
        st.session_state.sleep_start_time = dtime(hour=23, minute=0)
    if "sleep_end_time" not in st.session_state:
        st.session_state.sleep_end_time = dtime(hour=7, minute=0)
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""
    if "_loaded_persistence_namespace" not in st.session_state:
        st.session_state._loaded_persistence_namespace = None
    if "_persistence_error" not in st.session_state:
        st.session_state._persistence_error = None
    if "active_section" not in st.session_state:
        st.session_state.active_section = "Calendar"
    if "notifications_enabled" not in st.session_state:
        st.session_state.notifications_enabled = True
    if "notification_toasts_enabled" not in st.session_state:
        st.session_state.notification_toasts_enabled = True
    if "notification_browser_enabled" not in st.session_state:
        st.session_state.notification_browser_enabled = False
    if "notification_lead_minutes" not in st.session_state:
        st.session_state.notification_lead_minutes = 10
    if "notify_refill_ready" not in st.session_state:
        st.session_state.notify_refill_ready = True
    if "notify_drug_clear" not in st.session_state:
        st.session_state.notify_drug_clear = True
    if "notify_booster_clear" not in st.session_state:
        st.session_state.notify_booster_clear = True
    if "notify_jump_prep" not in st.session_state:
        st.session_state.notify_jump_prep = True
    if "notify_jump_execute" not in st.session_state:
        st.session_state.notify_jump_execute = True
    if "notify_gym_unlock" not in st.session_state:
        st.session_state.notify_gym_unlock = True


def render_sidebar() -> Tuple[str, int]:
    st.sidebar.header("Connection")
    api_key = st.sidebar.text_input(
        "Torn API key",
        type="password",
        key="api_key_input",
        help="This API key is also used as your private planner profile key. The app stores only a one-way hash of it, not the key itself.",
    )
    load_persistent_state_for_api(api_key)

    preview_days = st.sidebar.slider("Preview days", min_value=7, max_value=90, value=int(st.session_state.preview_days), step=1)
    st.session_state.preview_days = preview_days

    st.sidebar.header("Saved planner data")
    if api_key:
        st.sidebar.caption("This planner auto-saves to a private profile keyed by your API. The raw API key is not written to disk.")
    else:
        st.sidebar.caption("Enter your API key to load and save your private planner profile.")
    loaded_namespace = st.session_state.get("_loaded_persistence_namespace")
    if api_key and loaded_namespace:
        st.sidebar.caption(f"Profile loaded: {loaded_namespace[:8]}…")
    if st.session_state.get("player_state") and st.session_state.player_state.last_sync:
        st.sidebar.caption(f"Last saved snapshot: {fmt_local(st.session_state.player_state.last_sync)}")
    if st.sidebar.button("Clear saved planner data", use_container_width=True, disabled=not bool(api_key)):
        clear_persistent_state(api_key)
        reset_runtime_state(keep_api_fields=True)
        st.session_state._loaded_persistence_namespace = _api_namespace(api_key)
        st.rerun()
    if st.session_state.get("_persistence_error"):
        st.sidebar.warning(f"Saved data could not be loaded: {st.session_state['_persistence_error']}")

    st.sidebar.header("Time settings")
    selected_timezone = st.sidebar.selectbox(
        "Display timezone",
        options=ALL_TIMEZONE_OPTIONS,
        index=ALL_TIMEZONE_OPTIONS.index(st.session_state.display_timezone_name) if st.session_state.display_timezone_name in ALL_TIMEZONE_OPTIONS else ALL_TIMEZONE_OPTIONS.index(DEFAULT_APP_TIMEZONE_NAME),
        help="Choose the timezone used for displayed planner times and manual scheduling inputs when TCT override is off.",
    )
    st.session_state.display_timezone_name = selected_timezone
    set_app_timezone(st.session_state.display_timezone_name)
    st.session_state.use_tct_times = st.sidebar.checkbox(
        "Use TCT for displayed times",
        value=bool(st.session_state.use_tct_times),
        help="When enabled, displayed planner times switch to Torn City Time (TCT / UTC).",
    )
    if st.session_state.use_tct_times:
        st.sidebar.caption(f"Displayed times are using {TORN_TIMEZONE_LABEL}. Selected local timezone remains saved as {selected_timezone}.")
    else:
        st.sidebar.caption(f"Displayed times are using {get_app_timezone_label()}.")

    st.sidebar.header("Notifications")
    st.session_state.notifications_enabled = st.sidebar.checkbox("Enable notifications", value=bool(st.session_state.get("notifications_enabled", True)))
    st.session_state.notification_toasts_enabled = st.sidebar.checkbox("In-app toasts", value=bool(st.session_state.get("notification_toasts_enabled", True)), disabled=not bool(st.session_state.get("notifications_enabled", True)))
    st.session_state.notification_browser_enabled = st.sidebar.checkbox("Browser notifications", value=bool(st.session_state.get("notification_browser_enabled", False)), disabled=not bool(st.session_state.get("notifications_enabled", True)), help="Works while the tab is open. Your browser may ask for permission the first time a notification fires.")
    st.session_state.notification_lead_minutes = int(st.sidebar.number_input("Lead time (minutes)", min_value=0, max_value=240, value=int(st.session_state.get("notification_lead_minutes", 10)), step=5, disabled=not bool(st.session_state.get("notifications_enabled", True))))
    cna, cnb = st.sidebar.columns(2)
    with cna:
        st.session_state.notify_refill_ready = st.checkbox("Refill", value=bool(st.session_state.get("notify_refill_ready", True)), disabled=not bool(st.session_state.get("notifications_enabled", True)))
        st.session_state.notify_drug_clear = st.checkbox("Drug", value=bool(st.session_state.get("notify_drug_clear", True)), disabled=not bool(st.session_state.get("notifications_enabled", True)))
        st.session_state.notify_booster_clear = st.checkbox("Booster", value=bool(st.session_state.get("notify_booster_clear", True)), disabled=not bool(st.session_state.get("notifications_enabled", True)))
    with cnb:
        st.session_state.notify_jump_prep = st.checkbox("Jump prep", value=bool(st.session_state.get("notify_jump_prep", True)), disabled=not bool(st.session_state.get("notifications_enabled", True)))
        st.session_state.notify_jump_execute = st.checkbox("Jump execute", value=bool(st.session_state.get("notify_jump_execute", True)), disabled=not bool(st.session_state.get("notifications_enabled", True)))
        st.session_state.notify_gym_unlock = st.checkbox("Gym unlock", value=bool(st.session_state.get("notify_gym_unlock", True)), disabled=not bool(st.session_state.get("notifications_enabled", True)))

    st.sidebar.header("Planner assumptions")
    st.sidebar.caption("These match the locked v2 rules.")
    st.sidebar.write("- 3 Xanax/day")
    st.sidebar.write("- Daily refill used")
    st.sidebar.write("- Booster cooldown tracked")
    st.sidebar.write("- Auto jump scheduling enabled")
    st.sidebar.write("- War days can block training")
    st.sidebar.write("- Happy loss modeled with expected value")

    return api_key, preview_days


def auto_advance_gym_energy_progress(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> GoalSettings:
    """Auto-advance the manually seeded gym progress by completed planner days.

    Torn does not expose gym unlock energy progress through the API, so this uses
    the planner's completed-day assumptions to keep the value moving after the user
    seeds it once. It only advances through fully completed local days to avoid
    claiming intra-day precision that the API cannot verify.
    """
    as_of = getattr(goal, "gym_progress_as_of_date", local_today())
    if as_of >= local_today():
        return goal

    highest_idx = highest_unlocked_gym_index(state)
    projected_progress = max(0, int(goal.current_gym_energy_progress))
    projected_stats = state.stats
    projected_highest_idx = highest_idx

    day = as_of
    while day < local_today():
        projected_state = PlayerState(
            stats=projected_stats,
            recovery=state.recovery,
            unlocked_gyms=active_unlocked_names_for_stats(state.unlocked_gyms, projected_stats, projected_highest_idx, goal),
            faction_war_days=list(state.faction_war_days),
            torn_name=state.torn_name,
            torn_id=state.torn_id,
            faction_id=state.faction_id,
            faction_name=state.faction_name,
            training_modifiers=state.training_modifiers,
            api_notes=list(state.api_notes),
            last_sync=state.last_sync,
        )
        _, projected_stats, projected_highest_idx, projected_progress, _ = simulate_day_with_unlocks(
            projected_state, ratio, goal, day, manual_mods, projected_highest_idx, projected_progress
        )
        day += timedelta(days=1)

    updated = GoalSettings(**{**goal.__dict__, "current_gym_energy_progress": int(projected_progress), "gym_progress_as_of_date": local_today()})
    return updated


def estimate_next_gym_unlock(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers, days: int = 90) -> GymUnlockProjection:
    highest_idx = highest_unlocked_gym_index(state)
    current_gym = linear_gym_names()[highest_idx]
    next_gym = next_gym_name_for_index(highest_idx)
    threshold = next_gym_threshold_for_index(highest_idx)
    progress_e = max(0, int(goal.current_gym_energy_progress))
    if next_gym is None or threshold is None:
        return GymUnlockProjection(current_gym=current_gym, next_gym=None, current_progress=progress_e, required_progress=None, remaining_energy=None, estimated_unlock_at=None)

    projected_stats = state.stats
    projected_highest_idx = highest_idx
    projected_progress = progress_e
    for offset in range(days):
        plan_day = local_today() + timedelta(days=offset)
        projected_state = PlayerState(
            stats=projected_stats,
            recovery=state.recovery,
            unlocked_gyms=active_unlocked_names_for_stats(state.unlocked_gyms, projected_stats, projected_highest_idx, goal),
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


def estimate_specialist_unlock(gym_name: str, state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers, days: int = 365) -> SpecialistGymProjection:
    highest_idx = highest_unlocked_gym_index(state)
    base_names = active_unlocked_names_for_stats(state.unlocked_gyms, state.stats, highest_idx, goal)
    current_value, required_value, remaining_value, met = specialist_progress_snapshot(gym_name, state.stats, goal)
    parent_gym = SPECIALIST_PARENT_MAP.get(gym_name, "")
    parent_unlocked = specialist_parent_unlocked(base_names, gym_name)
    requirement_text = SPECIALIST_REQUIREMENT_TEXT.get(gym_name, "")

    if specialist_is_available(base_names, state.stats, gym_name, goal):
        return SpecialistGymProjection(
            gym_name=gym_name,
            parent_gym=parent_gym,
            parent_unlocked=parent_unlocked,
            current_value=current_value,
            required_value=required_value,
            remaining_value=remaining_value,
            estimated_unlock_at=local_now(),
            requirement_text=requirement_text,
        )

    if gym_name in {SSL_GYM_NAME, FIGHT_CLUB_NAME}:
        return SpecialistGymProjection(
            gym_name=gym_name,
            parent_gym=parent_gym,
            parent_unlocked=parent_unlocked,
            current_value=current_value,
            required_value=required_value,
            remaining_value=remaining_value,
            estimated_unlock_at=None,
            requirement_text=requirement_text,
        )

    projected_stats = state.stats
    projected_highest_idx = highest_idx
    projected_progress = max(0, int(goal.current_gym_energy_progress))
    for offset in range(days):
        plan_day = local_today() + timedelta(days=offset)
        projected_state = PlayerState(
            stats=projected_stats,
            recovery=state.recovery,
            unlocked_gyms=active_unlocked_names_for_stats(state.unlocked_gyms, projected_stats, projected_highest_idx, goal),
            faction_war_days=list(state.faction_war_days),
            torn_name=state.torn_name,
            torn_id=state.torn_id,
            faction_id=state.faction_id,
            faction_name=state.faction_name,
            training_modifiers=state.training_modifiers,
            api_notes=list(state.api_notes),
            last_sync=state.last_sync,
        )
        _instruction, projected_stats, projected_highest_idx, projected_progress, unlock_time = simulate_day_with_unlocks(
            projected_state, ratio, goal, plan_day, manual_mods, projected_highest_idx, projected_progress
        )
        projected_names = active_unlocked_names_for_stats(state.unlocked_gyms, projected_stats, projected_highest_idx, goal)
        if specialist_is_available(projected_names, projected_stats, gym_name, goal):
            est_time = unlock_time or datetime.combine(plan_day, dtime(hour=20, minute=0), tzinfo=APP_TIMEZONE)
            current_value, required_value, remaining_value, _ = specialist_progress_snapshot(gym_name, projected_stats, goal)
            return SpecialistGymProjection(
                gym_name=gym_name,
                parent_gym=parent_gym,
                parent_unlocked=specialist_parent_unlocked(projected_names, gym_name),
                current_value=current_value,
                required_value=required_value,
                remaining_value=remaining_value,
                estimated_unlock_at=est_time,
                requirement_text=requirement_text,
            )

    return SpecialistGymProjection(
        gym_name=gym_name,
        parent_gym=parent_gym,
        parent_unlocked=parent_unlocked,
        current_value=current_value,
        required_value=required_value,
        remaining_value=remaining_value,
        estimated_unlock_at=None,
        requirement_text=requirement_text,
    )


def estimate_frontline_unlock(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers, days: int = 365) -> SpecialistGymProjection:
    return estimate_specialist_unlock(FRONTLINE_GYM_NAME, state, ratio, goal, manual_mods, days)


def render_specialist_gyms_progress(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    st.subheader("Specialist gym progress")
    resolved_target = resolve_specialist_target(goal)
    configured_target = getattr(goal, "specialist_gym_target", "Auto from build")
    if configured_target == "Auto from build":
        st.caption(f"Showing the specialist gym implied by your current build: **{resolved_target}**.")
    elif resolved_target == "None":
        st.caption("No specialist gym target selected.")
    else:
        st.caption(f"Showing progress only for your selected specialist gym: **{resolved_target}**.")

    if resolved_target == "None":
        st.info("Pick a specialist gym target in Setup → Build structure to see its progress here.")
        return

    projection = estimate_specialist_unlock(resolved_target, state, ratio, goal, manual_mods)
    st.markdown(f"### {projection.gym_name}")
    st.caption(projection.requirement_text)
    if projection.parent_gym:
        st.write(f"Parent gym: **{projection.parent_gym}**")
    if not projection.parent_unlocked:
        st.info("Parent not unlocked yet.")
    if projection.gym_name == SSL_GYM_NAME:
        used_total = int(getattr(goal, 'ssl_combined_xanax_ecstasy_taken', 999))
        st.write(f"Combined Xanax + Ecstasy taken: **{used_total:,} / 150**")
        st.progress(min(1.0, max(0.0, (150 - min(used_total, 150)) / 150.0)))
    elif projection.gym_name == FIGHT_CLUB_NAME:
        st.write(f"Manual access flag: **{'On' if getattr(goal, 'fight_club_access', False) else 'Off'}**")
        st.progress(1.0 if getattr(goal, 'fight_club_access', False) else 0.0)
    else:
        required_value = max(projection.required_value, 1.0)
        st.write(f"Current requirement value: **{projection.current_value:,.0f}**")
        st.write(f"Required value: **{required_value:,.0f}**")
        st.progress(min(1.0, projection.current_value / required_value))
        if projection.remaining_value > 0:
            st.caption(f"Remaining gap: {projection.remaining_value:,.0f}")
    if projection.estimated_unlock_at is not None:
        unlock_at = to_local(projection.estimated_unlock_at)
        if unlock_at <= local_now() + timedelta(minutes=1):
            st.success("Available now in the planner.")
        else:
            st.success(f"Projected unlock: {fmt_local(unlock_at)}")
    elif projection.gym_name in {SSL_GYM_NAME, FIGHT_CLUB_NAME}:
        st.info("No automatic date projection for this requirement. Update the manual setting when it changes.")
    else:
        st.info("Unlock is beyond the current forecast window.")


def render_frontline_progress(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    render_specialist_gyms_progress(state, ratio, goal, manual_mods)
def render_goal_controls(goal: GoalSettings) -> GoalSettings:
    st.subheader("Goal setup")
    c1, c2 = st.columns(2)
    with c1:
        target_total = st.number_input("Target total battle stats", min_value=1_000_000, value=int(goal.target_total_stats), step=1_000_000)
    with c2:
        target_date = st.date_input("Target date", value=goal.target_date)

    c3, c4, c5, c6, c7a = st.columns(5)
    with c3:
        fhc_allowed = st.checkbox("FHC allowed", value=goal.fhc_allowed)
    with c4:
        cans_allowed = st.checkbox("Cans allowed", value=goal.cans_allowed)
    with c5:
        auto_schedule_happy_jumps = st.checkbox("Auto-schedule happy jumps", value=goal.auto_schedule_happy_jumps)
    with c6:
        schedule_99k_jump = st.checkbox("Schedule 99k jump", value=goal.schedule_99k_jump)
    with c7a:
        daily_refill_used_today = st.checkbox("I already used today's refill", value=getattr(goal, "daily_refill_used_today", True), help="Turn this on if your daily refill has already been used and won't be ready again until the next TST midnight reset.")

    skip_war_days = st.checkbox("Skip war days", value=goal.skip_war_days)
    refill_reset_local = next_tst_midnight_local()
    st.caption(f"App times are shown in {get_app_timezone_label()}. Torn resets use {TORN_TIMEZONE_LABEL}. {app_vs_tst_text()}. Next TST midnight / refill reset in your selected timezone: {fmt_local(refill_reset_local)} ({fmt_tst(refill_reset_local)}).")

    st.markdown("**Sleep schedule**")
    st.caption("Use this so the planner avoids scheduling training, Xanax, refills, and other timed actions while you are asleep.")
    s1, s2, s3 = st.columns(3)
    with s1:
        sleep_schedule_enabled = st.checkbox("Respect sleep schedule", value=getattr(goal, "sleep_schedule_enabled", False), key="sleep_schedule_enabled")
    with s2:
        sleep_start_time = st.time_input("Sleep start", value=getattr(goal, "sleep_start_time", dtime(hour=23, minute=0)), step=timedelta(minutes=15), key="sleep_start_time")
    with s3:
        sleep_end_time = st.time_input("Wake time", value=getattr(goal, "sleep_end_time", dtime(hour=7, minute=0)), step=timedelta(minutes=15), key="sleep_end_time")
    if sleep_schedule_enabled:
        preview_goal = GoalSettings(sleep_schedule_enabled=True, sleep_start_time=sleep_start_time, sleep_end_time=sleep_end_time, assumed_xanax_cooldown_hours=float(goal.assumed_xanax_cooldown_hours))
        st.caption(sleep_schedule_summary(preview_goal, assumed_xanax_cooldown_hours=float(goal.assumed_xanax_cooldown_hours)))

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
    manual_99k_jump_schedule_text = goal.manual_99k_jump_schedule_text
    if schedule_99k_jump:
        saved_entries = manual_99k_schedule_datetimes(goal)
        if saved_entries and not st.session_state.manual_99k_jump_entries:
            st.session_state.manual_99k_jump_entries = saved_entries
        elif not st.session_state.manual_99k_jump_entries:
            st.session_state.manual_99k_jump_entries = [datetime.combine(goal.scheduled_99k_jump_date, goal.scheduled_99k_jump_time).replace(tzinfo=APP_TIMEZONE)]

        st.markdown("**Manual 99k jump schedule (Central Time)**")
        st.caption("Add exact 99k jumps with the controls below. The app sorts them automatically and prevents duplicates.")

        c16, c17, c18, c19 = st.columns([1.2, 1, 0.8, 0.7])
        with c16:
            scheduled_99k_jump_date = st.date_input("99k jump date", value=goal.scheduled_99k_jump_date, key="manual_99k_jump_date")
        with c17:
            scheduled_99k_jump_time = st.time_input("99k jump time", value=goal.scheduled_99k_jump_time, step=timedelta(minutes=15), key="manual_99k_jump_time")
        with c18:
            add_jump = st.button("Add jump", use_container_width=True)
        with c19:
            clear_jumps = st.button("Clear all", use_container_width=True)

        if add_jump:
            new_dt = datetime.combine(scheduled_99k_jump_date, scheduled_99k_jump_time).replace(tzinfo=APP_TIMEZONE, second=0, microsecond=0)
            existing = {to_local(dt).replace(second=0, microsecond=0).isoformat() for dt in st.session_state.manual_99k_jump_entries}
            if new_dt.isoformat() not in existing:
                st.session_state.manual_99k_jump_entries = sorted(st.session_state.manual_99k_jump_entries + [new_dt], key=lambda d: to_local(d))
        if clear_jumps:
            st.session_state.manual_99k_jump_entries = []

        entries = sorted(st.session_state.manual_99k_jump_entries, key=lambda d: to_local(d))
        if entries:
            remove_options = [to_local(dt).strftime("%Y-%m-%d %H:%M") for dt in entries]
            r1, r2 = st.columns([3, 1])
            with r1:
                remove_choice = st.selectbox("Scheduled 99k jumps", options=remove_options, index=0, help="Pick an existing scheduled jump if you want to remove it.")
            with r2:
                remove_jump = st.button("Remove selected", use_container_width=True)
            if remove_jump:
                st.session_state.manual_99k_jump_entries = [dt for dt in entries if to_local(dt).strftime("%Y-%m-%d %H:%M") != remove_choice]
                entries = sorted(st.session_state.manual_99k_jump_entries, key=lambda d: to_local(d))

            schedule_rows = [{"Date": to_local(dt).strftime("%Y-%m-%d"), "Time": to_local(dt).strftime("%H:%M"), "Weekday": to_local(dt).strftime("%a")} for dt in entries]
            st.dataframe(schedule_rows, use_container_width=True, hide_index=True)
            manual_99k_jump_schedule_text = serialize_manual_99k_schedule(entries)
            st.caption(f"Scheduled 99k jumps loaded: {len(entries)}")
        else:
            fallback_dt = datetime.combine(scheduled_99k_jump_date, scheduled_99k_jump_time).replace(tzinfo=APP_TIMEZONE, second=0, microsecond=0)
            manual_99k_jump_schedule_text = serialize_manual_99k_schedule([fallback_dt])
            st.info("No manual 99k jumps added yet. Use the controls above to add one or more exact jump times.")
    else:
        manual_99k_jump_schedule_text = ""
        st.session_state.manual_99k_jump_entries = []

    st.caption(f"Gym unlock progress auto-tracks once seeded. Current auto-tracked estimate is as of **{getattr(goal, 'gym_progress_as_of_date', local_today()).isoformat()}**.")
    current_gym_energy_progress = st.number_input(
        "Gym energy progress seed / manual override",
        min_value=0,
        value=int(goal.current_gym_energy_progress),
        step=10,
        help="Seed this once from Torntools, then the app will auto-advance it by completed planner days. Change it again any time you want to re-anchor the estimate.",
    )

    st.subheader("Extra energy sources & setbacks")
    st.caption("These fields can still be edited manually, but API sync now auto-refreshes company stars, job points, refill state, and MCS claim timing when those endpoints are available. FHC and can counts stay manual until a stable inventory endpoint exists.")

    e1, e2, e3 = st.columns(3)
    with e1:
        current_company_stars = st.number_input("Current company stars", min_value=0, max_value=10, value=int(goal.current_company_stars), step=1)
        planned_10_star_date = st.date_input("Planned 10★ company date", value=goal.planned_10_star_date)
        use_job_points_energy = st.checkbox("Use job points for energy at 10★", value=goal.use_job_points_energy)
    with e2:
        current_job_points = st.number_input("Current job points", min_value=0, value=int(goal.current_job_points), step=10)
        reserve_job_points = st.number_input("Reserve job points", min_value=0, value=int(goal.reserve_job_points), step=10)
        job_points_daily_limit = st.number_input("Job points daily limit", min_value=0, value=int(goal.job_points_daily_limit), step=10)
    with e3:
        job_energy_per_point = st.number_input("Energy per job point", min_value=1, value=int(goal.job_energy_per_point), step=1)
        today_energy_loss_adjustment = st.number_input("Energy lost today (OD / missed use)", min_value=0, value=int(goal.today_energy_loss_adjustment), step=5)
        forecast_energy_loss_per_day = st.number_input("Avg daily energy loss for forecast", min_value=0, value=int(goal.forecast_energy_loss_per_day), step=5)

    e4, e5, e6 = st.columns(3)
    with e4:
        mcs_ready_claims_now = st.number_input("MCS claims ready now", min_value=0, value=int(goal.mcs_ready_claims_now), step=1)
        mcs_energy_per_claim = st.number_input("Energy per MCS claim", min_value=0, value=int(goal.mcs_energy_per_claim), step=10)
    with e5:
        mcs_next_ready_date = st.date_input("Next MCS ready date", value=goal.mcs_next_ready_date)
        mcs_next_ready_time = st.time_input("Next MCS ready time", value=goal.mcs_next_ready_time, step=timedelta(minutes=15), key="mcs_next_ready_time")
    with e6:
        fhc_count_available = st.number_input("FHCs available", min_value=0, value=int(goal.fhc_count_available), step=1)
        fhc_effective_energy = st.number_input("Effective energy per FHC", min_value=0, value=int(goal.fhc_effective_energy), step=10)

    e7, e8, e9, e10 = st.columns(4)
    with e7:
        can_count_available = st.number_input("Energy cans available", min_value=0, value=int(goal.can_count_available), step=1)
    with e8:
        can_energy_per_can = st.number_input("Effective energy per can", min_value=0, value=int(goal.can_energy_per_can), step=5)
    with e9:
        can_cooldown_hours = st.number_input("Booster cooldown per can (hours)", min_value=0.0, value=float(goal.can_cooldown_hours), step=0.5)
    with e10:
        fhc_cooldown_hours = st.number_input("Booster cooldown per FHC (hours)", min_value=0.0, value=float(goal.fhc_cooldown_hours), step=0.5)

    max_daily_booster_cooldown_hours = st.number_input("Max booster cooldown window (hours)", min_value=0.0, value=float(goal.max_daily_booster_cooldown_hours), step=1.0, help="Planner cap for total booster-use spacing over a rolling day. For your current setup this should be 24 hours.")

    return GoalSettings(
        target_total_stats=float(target_total),
        target_date=target_date,
        fhc_allowed=fhc_allowed,
        cans_allowed=cans_allowed,
        auto_schedule_happy_jumps=auto_schedule_happy_jumps,
        schedule_99k_jump=schedule_99k_jump,
        scheduled_99k_jump_date=scheduled_99k_jump_date,
        scheduled_99k_jump_time=scheduled_99k_jump_time,
        manual_99k_jump_schedule_text=manual_99k_jump_schedule_text,
        current_gym_energy_progress=int(current_gym_energy_progress),
        gym_progress_as_of_date=(local_today() if int(current_gym_energy_progress) != int(goal.current_gym_energy_progress) else getattr(goal, 'gym_progress_as_of_date', local_today())),
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
        daily_refill_available_now=(not daily_refill_used_today),
        daily_refill_used_today=daily_refill_used_today,
        current_company_stars=int(current_company_stars),
        planned_10_star_date=planned_10_star_date,
        use_job_points_energy=use_job_points_energy,
        current_job_points=int(current_job_points),
        reserve_job_points=int(reserve_job_points),
        job_points_daily_limit=int(job_points_daily_limit),
        job_energy_per_point=int(job_energy_per_point),
        mcs_ready_claims_now=int(mcs_ready_claims_now),
        mcs_energy_per_claim=int(mcs_energy_per_claim),
        mcs_next_ready_date=mcs_next_ready_date,
        mcs_next_ready_time=mcs_next_ready_time,
        fhc_count_available=int(fhc_count_available),
        fhc_effective_energy=int(fhc_effective_energy),
        can_count_available=int(can_count_available),
        can_energy_per_can=int(can_energy_per_can),
        can_cooldown_hours=float(can_cooldown_hours),
        fhc_cooldown_hours=float(fhc_cooldown_hours),
        max_daily_booster_cooldown_hours=float(max_daily_booster_cooldown_hours),
        sleep_schedule_enabled=bool(sleep_schedule_enabled),
        sleep_start_time=sleep_start_time,
        sleep_end_time=sleep_end_time,
        today_energy_loss_adjustment=int(today_energy_loss_adjustment),
        forecast_energy_loss_per_day=int(forecast_energy_loss_per_day),
    )


def render_ratio_controls(goal: GoalSettings, ratio: RatioProfile) -> Tuple[GoalSettings, RatioProfile]:
    st.subheader("Build structure")
    family = st.selectbox("Build family", options=BUILD_FAMILY_OPTIONS, index=BUILD_FAMILY_OPTIONS.index(goal.ratio_family) if goal.ratio_family in BUILD_FAMILY_OPTIONS else 0)
    primary = st.selectbox("Primary stat", options=STAT_KEYS, index=STAT_KEYS.index(goal.ratio_primary_stat) if goal.ratio_primary_stat in STAT_KEYS else 0, format_func=lambda x: x.title())
    st.caption(build_family_ratio_caption(family, primary))
    st.caption(build_family_specialist_summary(family, primary))

    default_target = default_specialist_target(family, primary)
    target_help = f"Auto from build currently points to: {default_target}. Choose a specific specialist gym if you only want to track progress toward that one."
    current_target = getattr(goal, "specialist_gym_target", "Auto from build")
    specialist_target = st.selectbox(
        "Specialist gym target",
        options=SPECIALIST_TARGET_OPTIONS,
        index=SPECIALIST_TARGET_OPTIONS.index(current_target) if current_target in SPECIALIST_TARGET_OPTIONS else 0,
        help=target_help,
    )
    resolved_target = default_target if specialist_target == "Auto from build" else specialist_target
    st.caption(f"Current specialist target for the Gyms tab: **{resolved_target}**")

    display_ratio = ratio_profile_from_build(family, primary, ratio) if family != "Custom" else ratio
    disabled = family != "Custom"
    c1, c2, c3, c4 = st.columns(4)
    strength = c1.number_input("Strength %", min_value=0.0, max_value=100.0, value=float(display_ratio.strength), step=0.01, disabled=disabled)
    speed = c2.number_input("Speed %", min_value=0.0, max_value=100.0, value=float(display_ratio.speed), step=0.01, disabled=disabled)
    defense = c3.number_input("Defense %", min_value=0.0, max_value=100.0, value=float(display_ratio.defense), step=0.01, disabled=disabled)
    dexterity = c4.number_input("Dexterity %", min_value=0.0, max_value=100.0, value=float(display_ratio.dexterity), step=0.01, disabled=disabled)
    total = strength + speed + defense + dexterity
    if abs(total - 100.0) > 0.05:
        st.warning(f"Ratio totals {total:.2f}%. Ideally this should equal 100%.")
    updated_goal = GoalSettings(**{**goal.__dict__, "ratio_family": family, "ratio_primary_stat": primary, "specialist_gym_target": specialist_target})
    updated_ratio = RatioProfile(strength=strength, speed=speed, defense=defense, dexterity=dexterity)
    return updated_goal, updated_ratio


def render_manual_modifier_controls() -> TrainingModifiers:
    st.subheader("Manual training modifier overrides")
    st.caption("Use these when the API does not expose a training bonus cleanly, or to test scenarios.")
    current = st.session_state.manual_mods

    c1, c2, c3 = st.columns(3)
    with c1:
        all_gym = st.number_input("All gym gains %", value=float(current.all_gym_gains_pct), step=0.5)
        happy_loss_reduction = st.number_input("Gym happy loss reduction %", value=float(current.happy_loss_reduction_pct), step=5.0)
    with c2:
        strength = st.number_input("Strength gym gains %", value=float(current.strength_gym_gains_pct), step=0.5)
        speed = st.number_input("Speed gym gains %", value=float(current.speed_gym_gains_pct), step=0.5)
    with c3:
        defense = st.number_input("Defense gym gains %", value=float(current.defense_gym_gains_pct), step=0.5)
        dexterity = st.number_input("Dexterity gym gains %", value=float(current.dexterity_gym_gains_pct), step=0.5)

    energy_regen_bonus = st.number_input("Energy regeneration bonus %", value=float(current.energy_regen_bonus_pct), step=5.0)

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
        st.success(f"Projected unlock: {fmt_local(projection.estimated_unlock_at)}")
    else:
        st.info("Projected unlock is beyond the current preview window.")


def render_support_status(goal: GoalSettings) -> None:
    st.subheader("Extra energy planner summary")
    total_support_energy, avg_support_per_day = total_support_energy_available_until_target(goal)
    activation_date = company_10_star_activation_date(goal)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Job points", f"{goal.current_job_points:,}")
    c2.metric("FHCs", int(goal.fhc_count_available))
    c3.metric("Cans", int(goal.can_count_available))
    c4.metric("Support E until target", f"{total_support_energy:,}")
    st.caption(f"10★ company activation date used by planner: {activation_date.isoformat()}. Average support energy available per day until target: {avg_support_per_day:,} E.")
    st.caption(f"MCS ready now: {goal.mcs_ready_claims_now} claim(s). Next MCS ready: {fmt_local(mcs_next_ready_local(goal))}.")
    st.caption(f"Booster planning cap: {goal.max_daily_booster_cooldown_hours:.0f}h total, {goal.can_cooldown_hours:.1f}h per can, {goal.fhc_cooldown_hours:.1f}h per FHC.")
    if goal.today_energy_loss_adjustment or goal.forecast_energy_loss_per_day:
        st.caption(f"Loss adjustments: today {goal.today_energy_loss_adjustment:,} E, forecast {goal.forecast_energy_loss_per_day:,} E/day.")


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
        f"Baseline daily energy: {planner_baseline_energy_per_day(state, goal, combined_mods):,} "
        f"(natural {state.recovery.natural_energy_per_day(energy_regen_bonus_pct=combined_mods.energy_regen_bonus_pct):,} + xanax + refill)"
    )

    if state.last_sync is not None:
        st.caption(f"Last sync: {fmt_local(state.last_sync)}")

    next_reset_local = next_tst_midnight_local()
    st.caption(f"Times shown in {get_app_timezone_label()}. {app_vs_tst_text()}. Next TST midnight / daily refill reset: {fmt_local(next_reset_local)} ({fmt_tst(next_reset_local)}).")

    if combined_mods.detected_sources:
        st.caption("Training modifiers in effect: " + "; ".join(combined_mods.detected_sources))

    if state.api_notes:
        for note in state.api_notes:
            st.info(note)


def render_unlocked_gym_editor(state: PlayerState) -> None:
    st.subheader("Unlocked gyms")
    st.caption("Live API sync may not expose unlocked gyms directly. Use this as the source of truth whenever needed.")

    gym_names = ordered_gym_names()
    fill_names = linear_gym_names()

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
            options=["-- none --"] + fill_names,
            index=0,
            key="highest_unlocked_gym_selector",
        )

    with c3:
        apply_fill = st.button("Apply quick fill", use_container_width=True)

    if apply_fill and highest_gym != "-- none --":
        highest_idx = fill_names.index(highest_gym)
        selected = fill_names[: highest_idx + 1]
        st.session_state.gym_multiselect = selected
        st.session_state.manual_unlocked_gyms = selected
        state.unlocked_gyms = selected
        st.rerun()
    else:
        st.session_state.gym_multiselect = selected

    ordered_selection = [name for name in gym_names if name in st.session_state.gym_multiselect]
    linear_selection = [name for name in ordered_selection if name in LINEAR_GYM_NAMES]
    if FRONTLINE_GYM_NAME in ordered_selection and not frontline_is_unlocked_for_stats(linear_selection, state.stats):
        ordered_selection = [name for name in ordered_selection if name != FRONTLINE_GYM_NAME]
        st.info(f"{FRONTLINE_GYM_NAME} will appear automatically once {FRONTLINE_PARENT_GYM_NAME} is unlocked and your Strength + Speed requirement is met.")

    st.session_state.manual_unlocked_gyms = ordered_selection
    state.unlocked_gyms = ordered_selection

    active_names = active_unlocked_names_for_stats(state.unlocked_gyms, state.stats, highest_unlocked_gym_index(state))
    if active_names:
        st.caption(f"Highest unlocked gym in planner: {active_names[-1]}")
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
    st.write(f"**Prep starts:** {fmt_local(jump_plan.prep_start)}")
    st.write(f"**Execute at:** {fmt_local(jump_plan.execute_at)}")
    st.write(f"**Projected normal gain:** {jump_plan.projected_normal_gain:,.2f}")
    st.write(f"**Projected jump gain:** {jump_plan.projected_jump_gain:,.2f}")
    st.write(f"**Projected extra gain:** {jump_plan.projected_gain_delta:,.2f}")

    for note in jump_plan.notes:
        st.write(f"- {note}")



def estimate_optimal_99k_jump_count(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> Tuple[int, int, float]:
    combined_mods = state.training_modifiers.merge(manual_mods)
    target_days = max(0, (goal.target_date - local_today()).days)
    if target_days <= 0:
        return 0, 0, 0.0

    baseline_goal = GoalSettings(**{**goal.__dict__, "schedule_99k_jump": False, "manual_99k_jump_schedule_text": ""})
    baseline_days = days_until_goal_estimate(state, baseline_goal, manual_mods)
    if baseline_days is None:
        return 0, 0, 0.0

    candidate_execute_at = local_now() + timedelta(hours=max(1.0, goal.jump_prep_hours))
    candidate_execute_at = next_quarter_hour(candidate_execute_at)
    per_jump_plan = build_specific_jump_plan(state, ratio, goal, combined_mods, "super_happy_jump", candidate_execute_at, manual_selected=True)
    per_jump_delta = max(0.0, per_jump_plan.projected_gain_delta if per_jump_plan is not None else 0.0)

    max_slots = max(0, int(((goal.target_date - local_today()).days * 24) // max(1.0, goal.jump_prep_hours)))
    if baseline_days <= target_days or per_jump_delta <= 0:
        return 0, max_slots, per_jump_delta

    remaining = max(0.0, goal.target_total_stats - state.stats.total())
    avg_gain_without_manual = remaining / max(1, baseline_days)
    shortfall_stats = max(0.0, remaining - (avg_gain_without_manual * target_days))
    recommended = math.ceil(shortfall_stats / per_jump_delta) if per_jump_delta > 0 else 0
    recommended = max(0, min(recommended, max_slots))
    return recommended, max_slots, per_jump_delta


def render_99k_optimizer_panel(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    st.subheader("99k jump optimizer")
    scheduled = manual_99k_schedule_datetimes(goal)
    recommended, max_slots, per_jump_delta = estimate_optimal_99k_jump_count(state, ratio, goal, manual_mods)
    scheduled_until_target = [dt for dt in scheduled if local_today() <= to_local(dt).date() <= goal.target_date]

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended 99k jumps", recommended)
    c2.metric("Currently scheduled", len(scheduled_until_target))
    c3.metric("Max feasible slots", max_slots)

    if per_jump_delta > 0:
        st.caption(f"Current estimated extra gain per 99k jump: {per_jump_delta:,.2f}")
    if scheduled_until_target:
        st.caption("Scheduled 99k jumps: " + ", ".join(fmt_local(dt) for dt in scheduled_until_target[:8]) + (" ..." if len(scheduled_until_target) > 8 else ""))

    if len(scheduled_until_target) < recommended:
        st.warning(f"You are currently short {recommended - len(scheduled_until_target)} manual 99k jump(s) versus the optimizer's current estimate.")
    elif len(scheduled_until_target) > recommended and recommended >= 0:
        st.info("You have at least as many manual 99k jumps scheduled as the optimizer currently recommends.")
    else:
        st.success("Your currently scheduled 99k jumps match the optimizer's current estimate.")


def build_today_action_plan(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> List[JumpStep]:
    combined_mods = state.training_modifiers.merge(manual_mods)
    today_type, jump_plan = day_type_for_date(state, ratio, goal, local_today(), combined_mods)
    target_stat = choose_target_stat(state.stats, ratio)
    gym = best_gym_for_stat(state, target_stat, goal)
    now_dt = local_now()
    actions: List[JumpStep] = []

    if gym is None:
        return actions

    if jump_plan is not None and today_type in {"prep", "happy_jump", "super_happy_jump"}:
        return [step for step in build_jump_sequence(state, goal, jump_plan) if step.when.date() == local_today()]

    if state.recovery.current_energy > 0:
        actions.append(JumpStep(now_dt, "Train current energy", f"Train your current {state.recovery.current_energy} energy into {target_stat.title()} at {gym.name}."))

    if state.recovery.daily_refill_enabled:
        refill_dt = next_daily_refill_ready_local(goal, now_dt, after_dt=now_dt + timedelta(minutes=10))
        if not getattr(goal, "daily_refill_used_today", True):
            actions.append(JumpStep(refill_dt, "Use daily refill", "Use your daily refill after your current energy block if you are training today."))
            actions.append(JumpStep(refill_dt + timedelta(minutes=1), "Train refill energy", f"Train the refill energy into {target_stat.title()} at {gym.name}."))
        else:
            actions.append(JumpStep(refill_dt, "Daily refill resets (TST midnight)", f"Your daily refill resets at Torn midnight. That is {fmt_local(refill_dt)} in your selected timezone / {fmt_tst(refill_dt)} in Torn Standard Time."))
            if refill_dt.date() == now_dt.date():
                actions.append(JumpStep(refill_dt + timedelta(minutes=1), "Use daily refill after reset", "Use your daily refill once the TST reset hits if you are still training today."))
                actions.append(JumpStep(refill_dt + timedelta(minutes=2), "Train refill energy", f"Train the refill energy into {target_stat.title()} at {gym.name}."))

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
        actions.append(JumpStep(unlock_dt, "Next gym unlocks", f"Projected unlock: {next_gym}. This estimate only uses today's reachable energy and current loss adjustments."))

    # Extra energy optimizer for today
    today_inventory = init_support_inventory(goal)
    required_extra_energy = estimate_required_extra_energy_for_day(state, ratio, goal, manual_mods, local_today())
    bonus_energy, _support_notes, allocations = allocate_support_energy(goal, today_inventory, local_today(), required_extra_energy, booster_cd_minutes=state.recovery.booster_cd_minutes)
    if bonus_energy > 0:
        actions.append(JumpStep(now_dt, "Optimizer target", f"To stay closer to your goal date, the planner wants about {bonus_energy} extra energy today from support sources."))
        cursor = max(now_dt + timedelta(minutes=15), now_dt)
        booster_ready = now_dt + timedelta(minutes=max(0, state.recovery.booster_cd_minutes))
        fhc_units = 0
        can_units = 0
        for source_name, units, _energy_added in allocations:
            if source_name == "FHC":
                fhc_units = units
            elif source_name == "Energy can":
                can_units = units
            elif source_name == "MCS stock energy":
                when = now_dt if goal.mcs_ready_claims_now > 0 else mcs_next_ready_local(goal)
                actions.append(JumpStep(when, "Claim MCS stock energy", f"Claim {units} MCS stock energy reward(s) for {units * goal.mcs_energy_per_claim} energy, then train it in {gym.name}."))
            elif source_name == "Job points":
                activation = datetime.combine(company_10_star_activation_date(goal), dtime(hour=9, minute=0)).replace(tzinfo=APP_TIMEZONE)
                when = max(now_dt, activation)
                actions.append(JumpStep(when, "Spend job points for energy", f"Use {units} job points from your 10★ Game Shop for {units * goal.job_energy_per_point} energy, then train it in {gym.name}."))

        if fhc_units > 0 or can_units > 0:
            fhc_cd = timedelta(hours=max(0.0, float(goal.fhc_cooldown_hours)))
            can_cd = timedelta(hours=max(0.0, float(goal.can_cooldown_hours)))
            available_minutes = booster_daily_capacity_minutes(goal, state.recovery.booster_cd_minutes)
            # Build the exact booster sequence within the daily cooldown cap.
            remaining_fhc = fhc_units
            remaining_can = can_units
            sequence: List[str] = []
            while remaining_fhc > 0 or remaining_can > 0:
                next_type = None
                if remaining_fhc > 0 and remaining_can > 0:
                    fhc_rate = goal.fhc_effective_energy / max(0.1, float(goal.fhc_cooldown_hours))
                    can_rate = goal.can_energy_per_can / max(0.1, float(goal.can_cooldown_hours))
                    next_type = "FHC" if fhc_rate <= can_rate else "Energy can"
                elif remaining_fhc > 0:
                    next_type = "FHC"
                else:
                    next_type = "Energy can"
                sequence.append(next_type)
                if next_type == "FHC":
                    remaining_fhc -= 1
                else:
                    remaining_can -= 1
            # Highest cooldown item should be the last use because its cooldown is effectively free in the 24h window.
            sequence.sort(key=lambda s: float(goal.fhc_cooldown_hours) if s == "FHC" else float(goal.can_cooldown_hours))
            booster_time = booster_ready
            for idx, source_name in enumerate(sequence):
                if source_name == "FHC":
                    actions.append(JumpStep(booster_time, f"Use FHC #{sum(1 for s in sequence[:idx+1] if s == 'FHC')}", f"Use one FHC for about {goal.fhc_effective_energy} energy, then train it in {gym.name}."))
                    if idx < len(sequence) - 1:
                        booster_time = booster_time + fhc_cd
                else:
                    actions.append(JumpStep(booster_time, f"Use energy can #{sum(1 for s in sequence[:idx+1] if s == 'Energy can')}", f"Use one can for about {goal.can_energy_per_can} energy, then train it in {gym.name}."))
                    if idx < len(sequence) - 1:
                        booster_time = booster_time + can_cd
            max_booster_energy = fhc_units * int(goal.fhc_effective_energy) + can_units * int(goal.can_energy_per_can)
            if required_extra_energy > bonus_energy:
                actions.append(JumpStep(now_dt, "Booster cap reached", f"Your daily booster limit only allows about {max_booster_energy} energy from FHC/cans in the next {goal.max_daily_booster_cooldown_hours:.0f} hours, so the optimizer cannot reach the full extra-energy target today."))

    if goal.today_energy_loss_adjustment > 0:
        actions.append(JumpStep(now_dt, "Energy loss adjustment", f"Planner is subtracting {goal.today_energy_loss_adjustment} energy today to account for overdoses or missed usage."))

    return actions




def build_action_plan_for_date(state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers, plan_day: date, instruction: Optional[DailyInstruction] = None) -> List[JumpStep]:
    if plan_day == local_today():
        return build_today_action_plan(state, ratio, goal, manual_mods)

    combined_mods = state.training_modifiers.merge(manual_mods)
    instruction = instruction or build_daily_instruction(state, ratio, goal, plan_day, manual_mods)
    day_type, jump_plan = day_type_for_date(state, ratio, goal, plan_day, combined_mods)
    target_stat = instruction.target_stat if instruction.target_stat != "none" else choose_target_stat(state.stats, ratio)
    gym = GYM_INDEX.get(instruction.gym_name) or best_gym_for_stat(state, target_stat, goal)
    day_start = datetime.combine(plan_day, dtime(hour=8, minute=0), tzinfo=APP_TIMEZONE)
    actions: List[JumpStep] = []

    if instruction.day_type == "war":
        return [JumpStep(day_start, "War / non-training day", "Training is skipped today because this date is marked as a war or non-training day.")]
    if gym is None:
        return [JumpStep(day_start, "Gym selection needed", "No valid gym is available for this day. Set your unlocked gyms first.")]
    if jump_plan is not None and day_type in {"prep", "happy_jump", "super_happy_jump"}:
        day_steps = [step for step in build_jump_sequence(state, goal, jump_plan) if to_local(step.when).date() == plan_day]
        if day_steps:
            return day_steps

    actions.append(JumpStep(day_start, "Start main training block", f"Train {instruction.target_stat.title()} in {gym.name}. Planned energy for the day is about {instruction.estimated_energy:,}."))

    # Assumed baseline schedule for future normal days.
    if instruction.day_type == "normal":
        if state.recovery.current_energy > 0:
            actions.append(JumpStep(day_start, "Spend opening energy", f"Use your opening energy toward {instruction.target_stat.title()} at {gym.name}."))
        if state.recovery.daily_refill_enabled:
            refill_at = datetime.combine(plan_day, dtime(hour=0, minute=1), tzinfo=TORN_TIMEZONE).astimezone(APP_TIMEZONE)
            actions.append(JumpStep(refill_at, "TST refill reset", f"Daily refill resets at {fmt_tst(refill_at)} / {fmt_local(refill_at)} in your selected timezone."))
            actions.append(JumpStep(refill_at + timedelta(minutes=1), "Use daily refill", f"Use the daily refill and train that energy into {instruction.target_stat.title()} at {gym.name}."))
        # Three baseline xanax windows.
        xanax_times = [
            datetime.combine(plan_day, dtime(hour=8, minute=5), tzinfo=APP_TIMEZONE),
            datetime.combine(plan_day, dtime(hour=15, minute=5), tzinfo=APP_TIMEZONE),
            datetime.combine(plan_day, dtime(hour=22, minute=5), tzinfo=APP_TIMEZONE),
        ]
        for idx, when in enumerate(xanax_times, start=1):
            actions.append(JumpStep(when, f"Baseline Xanax #{idx}", f"Take Xanax #{idx} if you are following the baseline plan, then train the extra {state.recovery.xanax_energy} energy in {gym.name}."))
        end_dt = datetime.combine(plan_day, dtime(hour=23, minute=59), tzinfo=APP_TIMEZONE)
        natural_e = natural_energy_between(state, combined_mods, day_start, end_dt)
        actions.append(JumpStep(end_dt, "Natural regen through end of day", f"About {natural_e} natural energy will regenerate through the end of this day at your current regen rate."))
    elif instruction.day_type == "prep":
        actions.append(JumpStep(day_start + timedelta(minutes=5), "Hold most energy", "Save as much energy as possible today so the scheduled jump can go off on time."))
        if jump_plan is not None:
            actions.append(JumpStep(jump_plan.execute_at - timedelta(hours=1), "Final jump prep check", f"Confirm items, cooldowns, and stack before the jump at {fmt_local(jump_plan.execute_at)}."))
    else:
        actions.append(JumpStep(day_start + timedelta(minutes=5), "Follow jump sequence", "This day is part of a jump sequence. Use the timed steps shown here in order."))

    return sorted(actions, key=lambda x: x.when)

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
    baseline_goal = GoalSettings(**{**goal.__dict__, "use_job_points_energy": False, "mcs_ready_claims_now": 0, "fhc_count_available": 0, "can_count_available": 0})
    baseline_days = days_until_goal_estimate(state, baseline_goal, manual_mods)
    optimized_days = days_until_goal_estimate(state, goal, manual_mods)
    target_days = (goal.target_date - local_today()).days
    if optimized_days is None:
        st.error("Forecast unavailable until a valid gym path and gain model are set.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline days to goal", baseline_days if baseline_days is not None else "N/A")
    c2.metric("Optimized days to goal", optimized_days)
    c3.metric("Days until target date", target_days)

    total_support_energy, avg_support_per_day = total_support_energy_available_until_target(goal)
    st.caption(f"Support energy budget until target: {total_support_energy:,} total ({avg_support_per_day:,} E/day average).")

    if optimized_days <= target_days:
        st.success("With the current extra-energy plan, the projection is on pace for the selected goal.")
    else:
        st.warning("Current baseline projection is behind the selected goal. Extra energy sources, jumps, and lost-energy adjustments will matter.")


def render_war_calendar_editor(state: PlayerState) -> None:
    st.subheader("War / non-training days")
    st.caption("Use quick-add controls instead of typing a long comma-separated list. Manual edits become the planner source of truth for this session.")

    if "war_days_editor" not in st.session_state:
        st.session_state.war_days_editor = sorted(set(state.faction_war_days))

    current_days: List[date] = sorted(set(st.session_state.war_days_editor))

    c1, c2 = st.columns(2)
    with c1:
        single_day = st.date_input("Add single day", value=local_today(), key="war_day_single_input")
        if st.button("Add day", key="war_day_add_button", use_container_width=True):
            current_days.append(single_day)
            st.session_state.war_days_editor = sorted(set(current_days))
            st.rerun()

    with c2:
        default_range = (local_today(), local_today() + timedelta(days=2))
        range_value = st.date_input("Add date range", value=default_range, key="war_day_range_input")
        if st.button("Add range", key="war_day_add_range_button", use_container_width=True):
            dates_to_add: List[date] = []
            if isinstance(range_value, tuple) and len(range_value) == 2:
                start_day, end_day = range_value
                if start_day > end_day:
                    start_day, end_day = end_day, start_day
                span = min((end_day - start_day).days, 120)
                dates_to_add = [start_day + timedelta(days=i) for i in range(span + 1)]
            elif isinstance(range_value, date):
                dates_to_add = [range_value]
            current_days.extend(dates_to_add)
            st.session_state.war_days_editor = sorted(set(current_days))
            st.rerun()

    c3, c4 = st.columns([3, 1])
    with c3:
        remove_options = ["-- none --"] + [d.isoformat() for d in current_days]
        remove_choice = st.selectbox("Remove a saved day", options=remove_options, key="war_day_remove_choice")
    with c4:
        st.write("")
        st.write("")
        if st.button("Remove day", key="war_day_remove_button", use_container_width=True):
            if remove_choice != "-- none --":
                st.session_state.war_days_editor = [d for d in current_days if d.isoformat() != remove_choice]
                st.rerun()

    c5, c6 = st.columns(2)
    with c5:
        if st.button("Use API-loaded days", key="war_day_reset_api_button", use_container_width=True):
            st.session_state.war_days_editor = sorted(set(state.faction_war_days))
            st.rerun()
    with c6:
        if st.button("Clear all days", key="war_day_clear_button", use_container_width=True):
            st.session_state.war_days_editor = []
            st.rerun()

    with st.expander("Paste / edit raw dates manually"):
        raw_default = ", ".join(d.isoformat() for d in current_days)
        raw = st.text_input("War days (comma-separated YYYY-MM-DD)", value=raw_default, key="war_day_raw_input")
        if st.button("Apply raw dates", key="war_day_apply_raw_button"):
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
            st.session_state.war_days_editor = sorted(set(parsed_days))
            st.rerun()

    final_days = sorted(set(st.session_state.war_days_editor))
    state.faction_war_days = final_days
    if final_days:
        st.caption("Current war / non-training days: " + " • ".join(d.strftime("%b %d") for d in final_days))
    else:
        st.caption("No war / non-training days currently set.")


def render_gain_debug_panel(state: PlayerState, goal: GoalSettings, ratio: RatioProfile, manual_mods: TrainingModifiers) -> None:
    st.subheader("Gain engine debug")
    target_stat = choose_target_stat(state.stats, ratio)
    gym = best_gym_for_stat(state, target_stat, goal)
    if gym is None:
        st.warning("No gym selected yet.")
        return

    combined_mods = state.training_modifiers.merge(manual_mods)
    start_happy = max(state.recovery.max_happy, goal.normal_day_start_happy)
    sim = simulate_training_block(
        base_stats=state.stats,
        stat_key=target_stat,
        gym=gym,
        total_energy=min(150, planner_baseline_energy_per_day(state, goal, combined_mods)),
        starting_happy=start_happy,
        mods=combined_mods,
    )

    st.caption(f"Previewing first {min(150, planner_baseline_energy_per_day(state, goal, combined_mods))} energy into {target_stat.title()} at {gym.name}.")
    rows = []
    for item in sim["per_train_preview"]:
        rows.append({"Train #": item["train"], "Happy before": round(item["happy_before"], 2), "Gain": round(item["gain"], 4)})
    st.dataframe(rows, use_container_width=True)




def inject_torn_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 18% 22%, rgba(120,120,120,0.08), transparent 28%),
                radial-gradient(circle at 82% 26%, rgba(255,255,255,0.05), transparent 20%),
                linear-gradient(180deg, #151515 0%, #0f1114 46%, #090a0c 100%);
            color: #e2e6ea;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1680px;
        }
        .stSidebar {
            background: linear-gradient(180deg, #1a1d21 0%, #121417 100%);
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        h1, h2, h3, [data-testid="stMarkdownContainer"] h4 {
            color: #f1f3f5 !important;
            letter-spacing: 0.01em;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(53,57,64,0.75) 0%, rgba(22,24,28,0.96) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        [data-testid="stMetricLabel"] {
            color: #cfd4da !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.7rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #8ef041 !important;
            text-shadow: 0 0 14px rgba(107,255,75,0.12);
        }
        div[data-baseweb="tab-list"] {
            gap: 0.3rem;
            background: linear-gradient(180deg, rgba(69,73,79,0.65) 0%, rgba(27,29,33,0.96) 100%);
            padding: 0.42rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        button[data-baseweb="tab"] {
            background: linear-gradient(180deg, #3b3f45 0%, #1b1d22 100%) !important;
            border-radius: 9px !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            color: #d9dde2 !important;
            font-weight: 700 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, #8ecb31 0%, #5e9f14 100%) !important;
            color: #0d1108 !important;
            border-color: rgba(255,255,255,0.15) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 0 0 1px rgba(122,211,55,0.15);
        }
        .torn-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            background: linear-gradient(180deg, #282c31 0%, #1c1f24 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.6rem 0.95rem;
            margin-bottom: 0.8rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .torn-logo {
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            color: #f7f7f7;
            text-shadow: 0 2px 0 rgba(0,0,0,0.55);
        }
        .torn-nav {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            color: #d2d7dd;
            font-size: 0.86rem;
            font-weight: 700;
        }
        .torn-nav span { opacity: 0.92; }
        .torn-clock {
            color: #d4dbe2;
            font-size: 0.82rem;
            font-weight: 700;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 999px;
            padding: 0.28rem 0.75rem;
        }
        .torn-hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 75% 35%, rgba(255,255,255,0.08), transparent 24%),
                linear-gradient(120deg, rgba(32,35,40,0.97) 0%, rgba(19,21,24,0.98) 42%, rgba(11,12,14,0.99) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 38px rgba(0,0,0,0.35);
        }
        .torn-hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(173,17,17,0.10), transparent 24%, transparent 78%, rgba(255,255,255,0.03));
            pointer-events: none;
        }
        .torn-hero-title {
            font-size: 1.8rem;
            font-weight: 900;
            color: #f6f6f6;
            margin-bottom: 0.2rem;
            letter-spacing: 0.02em;
        }
        .torn-hero-sub {
            color: #c8d0d8;
            font-size: 0.94rem;
            max-width: 74rem;
        }
        .torn-chip-row {
            margin-top: 0.7rem;
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
        }
        .torn-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            background: rgba(142, 203, 49, 0.12);
            border: 1px solid rgba(142, 203, 49, 0.28);
            color: #c7f29c;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stButton"] > button,
        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(180deg, #4f555e 0%, #1e232a 100%);
            color: #f0f3f5;
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 10px;
            font-weight: 700;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        [data-testid="stButton"] > button:hover,
        [data-testid="stDownloadButton"] > button:hover {
            border-color: rgba(142,203,49,0.38);
            color: #e9ffd2;
        }
        [data-testid="stTextInputRootElement"], [data-baseweb="select"], textarea, .stDateInput > div > div, .stNumberInput > div > div {
            background: linear-gradient(180deg, rgba(58,62,69,0.78) 0%, rgba(22,24,28,0.98) 100%) !important;
            border-radius: 10px !important;
        }
        [data-baseweb="input"] input, textarea {
            color: #f2f5f7 !important;
        }
        .stDataFrame, .stTable {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(43,47,53,0.72) 0%, rgba(16,18,21,0.96) 100%);
        }
        [data-testid="stMarkdownContainer"] code {
            color: #c5f48a;
        }
        .calendar-cell {
            background: linear-gradient(180deg, rgba(52,56,62,0.66) 0%, rgba(16,18,21,0.98) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 4px solid #5f646b;
            border-radius: 14px;
            padding: 0.68rem 0.74rem;
            min-height: 156px;
            margin-bottom: 0.4rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .calendar-empty {
            background: linear-gradient(180deg, rgba(36,38,42,0.35) 0%, rgba(14,15,17,0.76) 100%);
            border: 1px dashed rgba(255,255,255,0.06);
            border-radius: 14px;
            min-height: 156px;
            margin-bottom: 0.4rem;
        }
        .calendar-detail-shell {
            margin-top: 0.5rem;
            padding: 0.7rem;
            border-radius: 12px;
            border: 1px solid rgba(142,203,49,0.22);
            background: linear-gradient(180deg, rgba(38,42,31,0.72) 0%, rgba(12,14,11,0.94) 100%);
        }
        .calendar-detail-title {
            font-size: 0.95rem;
            font-weight: 800;
            color: #f2f4f6;
            margin-bottom: 0.45rem;
        }
        .calendar-detail-meta {
            font-size: 0.78rem;
            color: #d0d8df;
            margin-bottom: 0.18rem;
        }
        .calendar-step {
            display: flex;
            gap: 0.55rem;
            align-items: flex-start;
            margin-top: 0.4rem;
            padding: 0.52rem;
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(60,64,70,0.32) 0%, rgba(18,20,24,0.88) 100%);
            border: 1px solid rgba(255,255,255,0.06);
        }
        .calendar-step-index {
            min-width: 1.45rem;
            height: 1.45rem;
            border-radius: 999px;
            background: rgba(142, 203, 49, 0.18);
            color: #d6ffad;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 800;
        }
        .calendar-step-when {
            font-size: 0.72rem;
            color: #9ac85f;
            margin-bottom: 0.12rem;
        }
        .calendar-step-action {
            font-size: 0.8rem;
            color: #f4f6fb;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .calendar-step-details {
            font-size: 0.73rem;
            color: #cad3dd;
            line-height: 1.35;
        }
        .calendar-date {
            font-weight: 900;
            color: #ffffff;
            margin-bottom: 0.32rem;
            font-size: 0.98rem;
        }
        .calendar-meta {
            color: #dfe4e8;
            font-size: 0.8rem;
            line-height: 1.35;
            margin-top: 0.18rem;
        }
        .calendar-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.14rem 0.48rem;
            font-size: 0.7rem;
            font-weight: 800;
            margin-bottom: 0.42rem;
            background: rgba(255,255,255,0.08);
            color: #f5f5f5;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .day-normal { border-left-color: #8ea0b5; }
        .day-prep { border-left-color: #d4a63b; }
        .day-happy_jump { border-left-color: #b46dff; }
        .day-super_happy_jump { border-left-color: #8ecb31; }
        .day-war { border-left-color: #bf3131; }
        .day-blocked { border-left-color: #5d5d5d; }
        .day-happy_jump .calendar-badge { background: rgba(180,109,255,0.22); }
        .day-super_happy_jump .calendar-badge { background: rgba(142,203,49,0.20); color: #d8f8b2; }
        .day-prep .calendar-badge { background: rgba(212,166,59,0.22); }
        .day-war .calendar-badge { background: rgba(191,49,49,0.28); }
        .section-chip {
            display: inline-block;
            margin-left: 0.45rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: rgba(142,203,49,0.18);
            color: #dbffb8;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .stAlert {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.07);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_torn_hero() -> None:
    now_label = fmt_local(local_now())
    st.markdown(
        f"""
        <div class="torn-topbar">
            <div class="torn-logo">TORN</div>
            <div></div>
            <div class="torn-clock">{now_label}</div>
        </div>
        <div class="torn-hero">
            <div class="torn-hero-title">Stat Tracker Command Console</div>
            <div class="torn-hero-sub">Torn-inspired planning console for training, jumps, unlocks, support energy, and day-by-day execution. Times shown in {get_app_timezone_label()} and aligned against Torn reset logic.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_section_nav() -> str:
    st.caption("Navigate sections")
    selected = st.radio(
        "Primary navigation",
        ["Overview", "Setup", "Progress", "Gyms", "Calendar"],
        index=["Overview", "Setup", "Progress", "Gyms", "Calendar"].index(st.session_state.active_section) if st.session_state.active_section in ["Overview", "Setup", "Progress", "Gyms", "Calendar"] else 4,
        horizontal=True,
        key="section_nav_radio",
        label_visibility="collapsed",
    )
    st.session_state.active_section = selected
    return selected

def _calendar_day_html(item: DailyInstruction, today: date) -> str:
    day_class = item.day_type.replace(' ', '_')
    date_str = item.plan_date.strftime('%b %d')
    badge = item.day_type.replace('_', ' ').title()
    today_chip = ' <span class="section-chip">Today</span>' if item.plan_date == today else ''
    return f"""
    <div class="calendar-cell day-{day_class}">
        <div class="calendar-date">{date_str}{today_chip}</div>
        <div class="calendar-badge">{badge}</div>
        <div class="calendar-meta"><strong>Gym:</strong> {item.gym_name}</div>
        <div class="calendar-meta"><strong>Train:</strong> {item.target_stat.title() if item.target_stat != 'none' else 'None'}</div>
        <div class="calendar-meta"><strong>Energy:</strong> {item.estimated_energy:,}</div>
        <div class="calendar-meta"><strong>Gain:</strong> {item.estimated_gain:,.0f}</div>
    </div>
    """


def render_calendar_tab(plan: List[DailyInstruction], state: PlayerState, ratio: RatioProfile, goal: GoalSettings, manual_mods: TrainingModifiers) -> None:
    st.subheader("Training calendar")
    st.caption("Day details stay hidden until you click Open day.")
    if not plan:
        st.info("No calendar data available yet.")
        return

    today = local_today()
    plan_map = {item.plan_date: item for item in plan}
    # Keep all day details hidden until the user explicitly opens a day.
    # Do not auto-expand today or the first day on initial load.

    first_day = plan[0].plan_date
    last_day = plan[-1].plan_date
    grid_start = first_day - timedelta(days=first_day.weekday())
    grid_end = last_day + timedelta(days=(6 - last_day.weekday()))

    selected_day: Optional[date] = None
    selected_day_str = st.session_state.selected_calendar_date
    if selected_day_str:
        try:
            selected_day = date.fromisoformat(selected_day_str)
        except ValueError:
            selected_day = None
            st.session_state.selected_calendar_date = None

    header_cols = st.columns(7)
    for col, label in zip(header_cols, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        with col:
            st.caption(f"**{label}**")

    cursor = grid_start
    while cursor <= grid_end:
        cols = st.columns(7)
        for col_idx in range(7):
            day = cursor + timedelta(days=col_idx)
            with cols[col_idx]:
                item = plan_map.get(day)
                if item is None:
                    st.markdown('<div class="calendar-empty"></div>', unsafe_allow_html=True)
                else:
                    st.markdown(_calendar_day_html(item, today), unsafe_allow_html=True)
                    is_selected = day == selected_day
                    button_label = "Hide day" if is_selected else "Open day"
                    if st.button(button_label, key=f"cal_open_{day.isoformat()}", use_container_width=True):
                        if is_selected:
                            st.session_state.selected_calendar_date = None
                            st.rerun()
                        else:
                            st.session_state.selected_calendar_date = day.isoformat()
                            st.rerun()

                    if is_selected:
                        steps = build_action_plan_for_date(state, ratio, goal, manual_mods, day, item)
                        st.markdown(
                            f"""
                            <div class="calendar-detail-shell">
                                <div class="calendar-detail-title">{day.strftime('%a, %b %d')} plan</div>
                                <div class="calendar-detail-meta"><strong>Type:</strong> {item.day_type.replace('_', ' ').title()}</div>
                                <div class="calendar-detail-meta"><strong>Gym:</strong> {item.gym_name}</div>
                                <div class="calendar-detail-meta"><strong>Train:</strong> {item.target_stat.title() if item.target_stat != 'none' else 'None'}</div>
                                <div class="calendar-detail-meta"><strong>Planned energy:</strong> {item.estimated_energy:,}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        for idx, step in enumerate(sorted(steps, key=lambda x: x.when), start=1):
                            st.markdown(
                                f"""
                                <div class="calendar-step">
                                    <div class="calendar-step-index">{idx}</div>
                                    <div class="calendar-step-body">
                                        <div class="calendar-step-when">{fmt_local(step.when)}</div>
                                        <div class="calendar-step-action">{step.action}</div>
                                        <div class="calendar-step-details">{step.details}</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
        cursor += timedelta(days=7)

def render_setup_tab() -> TrainingModifiers:
    st.subheader("Planner setup")
    st.caption("All planning inputs live here so the rest of the app can focus on results.")
    st.session_state.goal_settings = render_goal_controls(st.session_state.goal_settings)
    st.session_state.goal_settings, st.session_state.ratio_profile = render_ratio_controls(st.session_state.goal_settings, st.session_state.ratio_profile)
    st.subheader("Specialist gym settings")
    c1, c2 = st.columns(2)
    with c1:
        ssl_taken = st.number_input("Lifetime Xanax + Ecstasy taken (for SSL)", min_value=0, value=int(st.session_state.goal_settings.ssl_combined_xanax_ecstasy_taken), step=1)
    with c2:
        fight_access = st.checkbox("I have Fight Club access", value=bool(st.session_state.goal_settings.fight_club_access))
    st.caption("SSL only opens if Last Round is unlocked and your combined Xanax + Ecstasy total is 150 or less. Fight Club is manual because its requirement is not public.")
    st.session_state.goal_settings = GoalSettings(**{**st.session_state.goal_settings.__dict__, "ssl_combined_xanax_ecstasy_taken": int(ssl_taken), "fight_club_access": bool(fight_access)})
    manual_mods = render_manual_modifier_controls()
    st.session_state.manual_mods = manual_mods
    return manual_mods
def main() -> None:
    st.set_page_config(page_title="Torn Stat Tracker v2", layout="wide")
    init_state()
    inject_torn_theme()
    render_torn_hero()

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
                st.session_state.goal_settings, auto_sync_notes = auto_sync_goal_settings_from_api(api_key, st.session_state.goal_settings)
                if auto_sync_notes:
                    synced.api_notes.extend(auto_sync_notes)
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
        save_persistent_state(api_key)
        st.info("Load demo data or sync with your API key to begin.")
        return

    selected_section = render_section_nav()

    if selected_section == "Setup":
        manual_mods = render_setup_tab()
        st.subheader("War and schedule inputs")
        render_war_calendar_editor(st.session_state.player_state)

    player_state: PlayerState = st.session_state.player_state
    if "manual_mods" not in st.session_state:
        st.session_state.manual_mods = TrainingModifiers()
    manual_mods = st.session_state.manual_mods

    plan = build_plan_preview(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods, days=preview_days)

    if selected_section == "Overview":
        render_player_snapshot(player_state, st.session_state.goal_settings, manual_mods)
        render_support_status(st.session_state.goal_settings)
        render_today_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
        render_jump_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
        render_99k_optimizer_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
    elif selected_section == "Progress":
        render_progress_section(player_state, st.session_state.goal_settings, st.session_state.ratio_profile)
        render_forecast(player_state, st.session_state.goal_settings, manual_mods)
        with st.expander("Gain engine debug"):
            render_gain_debug_panel(player_state, st.session_state.goal_settings, st.session_state.ratio_profile, manual_mods)
    elif selected_section == "Gyms":
        render_next_gym_progress(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
        render_specialist_gyms_progress(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
        render_unlocked_gym_editor(player_state)
    elif selected_section == "Calendar":
        render_calendar_tab(plan, player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
        st.subheader("Today’s timed actions")
        render_daily_planner_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
        with st.expander("Jump sequence"):
            render_jump_sequence_panel(player_state, st.session_state.ratio_profile, st.session_state.goal_settings, manual_mods)
        with st.expander("Plan preview table"):
            render_plan_table(plan)

    save_persistent_state(api_key)


if __name__ == "__main__":
    main()
