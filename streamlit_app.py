# ============================================================
# REFACTORED: Session state / persistence layer
# Drop-in replacements for the originals in Untitled.txt
# ============================================================

# ---------------------------------------------------------------------------
# Single source of truth for all persisted session-state keys + defaults.
# reset_runtime_state, _current_persistence_payload, and
# _apply_persistent_payload all drive off this dict, so adding a new setting
# only requires one edit here.
# ---------------------------------------------------------------------------

PERSISTENCE_DEFAULTS: dict = {
    "goal_settings": None,
    "ratio_profile": None,
    "manual_mods": None,
    "manual_unlocked_gyms": [],
    "gym_multiselect": [],
    "highest_unlocked_gym_selector": "-- none --",
    "manual_99k_jump_entries": [],
    "player_state": None,
    "preview_days": 30,
    "display_timezone_name": DEFAULT_APP_TIMEZONE_NAME,
    "use_tct_times": False,
    "sleep_schedule_enabled": False,
    "sleep_start_time": dtime(hour=23, minute=0),
    "sleep_end_time": dtime(hour=7, minute=0),
    "notifications_enabled": True,
    "notification_toasts_enabled": True,
    "notification_browser_enabled": False,
    "notification_lead_minutes": 10,
    "notify_refill_ready": True,
    "notify_drug_clear": True,
    "notify_booster_clear": True,
    "notify_jump_prep": True,
    "notify_jump_execute": True,
    "notify_gym_unlock": True,
}

# Dataclass keys that need special construction on load
_DATACLASS_KEYS: dict = {
    "goal_settings": (GoalSettings, {"auto_schedule_jumps": "auto_schedule_happy_jumps"}),
    "ratio_profile": (RatioProfile, {}),
    "manual_mods": (TrainingModifiers, {}),
}


def _current_persistence_payload() -> Dict[str, Any]:
    """Snapshot current session state into a serialisable dict."""
    payload = {}
    for key, default in PERSISTENCE_DEFAULTS.items():
        payload[key] = st.session_state.get(key, default)
    return payload


def reset_runtime_state(keep_api_fields: bool = True) -> None:
    """Reset all persisted session-state keys to their defaults."""
    loaded_namespace = (
        st.session_state.get("_loaded_persistence_namespace")
        if keep_api_fields
        else None
    )

    for key, default in PERSISTENCE_DEFAULTS.items():
        # Lists/dicts are mutable — give each reset a fresh copy.
        st.session_state[key] = list(default) if isinstance(default, list) else default

    # Derived defaults that depend on other state
    st.session_state.goal_settings = GoalSettings()
    st.session_state.ratio_profile = RatioProfile()
    st.session_state.manual_mods = TrainingModifiers()
    st.session_state.manual_99k_jump_entries = manual_99k_schedule_datetimes(
        st.session_state.goal_settings
    )
    st.session_state.manual_99k_jump_date = local_today() + timedelta(days=7)

    # Internal / non-persisted runtime keys
    st.session_state.selected_calendar_date = None
    st.session_state._notified_events = []
    st.session_state._persistence_error = None

    if keep_api_fields:
        st.session_state._loaded_persistence_namespace = loaded_namespace


def _apply_persistent_payload(payload: Dict[str, Any]) -> None:
    """Restore session state from a persisted payload dict."""
    # Dataclass fields — use safe loader to tolerate schema drift
    for key, (cls, aliases) in _DATACLASS_KEYS.items():
        if isinstance(payload.get(key), dict):
            st.session_state[key] = _safe_dataclass_load(cls, payload[key], aliases)

    # PlayerState has a bespoke constructor
    if isinstance(payload.get("player_state"), dict):
        st.session_state.player_state = _player_state_from_dict(payload["player_state"])

    # Scalar / list keys — cast to expected types using PERSISTENCE_DEFAULTS as the guide
    _CASTS: dict = {
        "preview_days": int,
        "notification_lead_minutes": int,
        "use_tct_times": bool,
        "sleep_schedule_enabled": bool,
        "notifications_enabled": bool,
        "notification_toasts_enabled": bool,
        "notification_browser_enabled": bool,
        "notify_refill_ready": bool,
        "notify_drug_clear": bool,
        "notify_booster_clear": bool,
        "notify_jump_prep": bool,
        "notify_jump_execute": bool,
        "notify_gym_unlock": bool,
    }
    _LIST_KEYS = {
        "manual_unlocked_gyms",
        "gym_multiselect",
        "manual_99k_jump_entries",
    }

    for key, default in PERSISTENCE_DEFAULTS.items():
        if key in _DATACLASS_KEYS or key == "player_state":
            continue  # already handled above
        raw = payload.get(key, default)
        if key in _LIST_KEYS:
            st.session_state[key] = list(raw) if raw is not None else []
        elif key in _CASTS:
            st.session_state[key] = _CASTS[key](raw)
        else:
            st.session_state[key] = raw if raw is not None else default


# ---------------------------------------------------------------------------
# REFACTORED: _player_state_from_dict
# Uses _safe_dataclass_load for all nested dataclasses instead of direct **
# unpacking, so it gracefully handles schema changes without crashing.
# ---------------------------------------------------------------------------

def _player_state_from_dict(data: Dict[str, Any]) -> "PlayerState":
    return PlayerState(
        stats=_safe_dataclass_load(PlayerStats, data.get("stats", {})),
        recovery=_safe_dataclass_load(RecoveryState, data.get("recovery", {})),
        training_modifiers=_safe_dataclass_load(
            TrainingModifiers, data.get("training_modifiers", {})
        ),
        unlocked_gyms=list(data.get("unlocked_gyms", [])),
        faction_war_days=list(data.get("faction_war_days", [])),
        torn_name=data.get("torn_name", ""),
        torn_id=data.get("torn_id"),
        faction_id=data.get("faction_id"),
        faction_name=data.get("faction_name", ""),
        api_notes=list(data.get("api_notes", [])),
        last_sync=data.get("last_sync"),
    )


# ---------------------------------------------------------------------------
# REFACTORED: save_persistent_state
# Extracts the metadata block into a helper for clarity.
# ---------------------------------------------------------------------------

def _build_save_metadata() -> Dict[str, Any]:
    state = st.session_state.get("player_state")
    return {
        "torn_name": getattr(state, "torn_name", "") if state else "",
        "torn_id": getattr(state, "torn_id", None) if state else None,
        "saved_at": local_now(),
    }


def save_persistent_state(api_key: str = "") -> None:
    namespace = _api_namespace(api_key)
    if not namespace:
        return
    store = _read_persistence_store()
    payload = _current_persistence_payload()
    payload["_meta"] = _build_save_metadata()
    store.setdefault("profiles", {})[namespace] = payload
    _write_persistence_store(store)


# ---------------------------------------------------------------------------
# REFACTORED: set_app_timezone
# Removed the `global` mutation — APP_TIMEZONE is now set exclusively via
# st.session_state so it's scoped to each user session in multi-user deploys.
# Update all call sites: replace APP_TIMEZONE with
#   st.session_state.get("_app_timezone", ZoneInfo(DEFAULT_APP_TIMEZONE_NAME))
# ---------------------------------------------------------------------------

def set_app_timezone(timezone_name: Optional[str]) -> None:
    name = timezone_name or DEFAULT_APP_TIMEZONE_NAME
    try:
        tz = ZoneInfo(name)
    except Exception:
        tz = ZoneInfo(DEFAULT_APP_TIMEZONE_NAME)
    st.session_state["_app_timezone"] = tz


def get_app_timezone() -> "ZoneInfo":
    """Always use this instead of the module-level APP_TIMEZONE."""
    return st.session_state.get("_app_timezone", ZoneInfo(DEFAULT_APP_TIMEZONE_NAME))
