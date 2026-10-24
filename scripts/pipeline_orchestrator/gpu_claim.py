"""GPU right-of-way client: ask VICARIUS on port 5090 whether a card is free.

This file is the VICARIUS platform reference implementation for the GPU
right-of-way helper (spec docs/superpowers/specs/2026-09-17-gpu-right-of-way-
design.md, section "The helper (copied into modules)"): a module that starts
GPU work copies this file verbatim under the name gpu_claim.py and calls
check(), claim() and release() before touching a card. Planned copies: the
VOYAGER modules (3D_phase_1/src/gpu_claim.py, 3D_phase2/src/gpu_claim.py),
VEGA's orchestrator (reef_point_seg's pipeline_orchestrator/gpu_claim.py),
Vesper.llm (vicarius_llm/src/gpu_claim.py) and the Carousel driver
(driver/gpu_claim.py). This is the same copy-in pattern as lock_status.py in
reef_point_seg's pipeline_orchestrator.

The one claims registry is written only by vicarius_ui_os (the desktop UI on
port 5090); this helper never writes anything itself, it only asks over HTTP.
It never raises: an HTTP 409 or 428 from the server is a decision (refused,
wait, or needs a PIN), not a failure, and is handed back to the caller exactly as
the server sent it. Only a connection failure, a timeout, or a reply this
helper cannot parse counts as a failure, and that failure is reported as
decision "unregistered": the caller proceeds with its work, and VICARIUS's
watcher will show the caller as an unclaimed holder once port 5090 answers
again. VICARIUS_GPU_BYPASS=1 skips the request altogether and answers
"unregistered" at once, for a manual dev run with no desktop UI up.

Stdlib only, so any module can embed this with zero dependencies: json, os,
sys, urllib.request, urllib.error.
"""

import json
import os
import sys
import urllib.error
import urllib.request

# The desktop UI's own base address. VICARIUS_UI_URL overrides it so a test,
# or a box where 5090 is not the desktop UI's port, can point elsewhere.
BASE_URL_ENV = "VICARIUS_UI_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:5090"
BASE_URL = os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL

# A manual dev run with no desktop UI up sets this to skip the network call
# entirely and proceed as "unregistered" (spec: "logged as such").
BYPASS_ENV = "VICARIUS_GPU_BYPASS"

# How long a socket operation may take before this helper gives up and
# treats port 5090 as unreachable.
TIMEOUT_S = 3

CHECK_PATH = "/api/gpu/check"
CLAIM_PATH = "/api/gpu/claim"
RELEASE_PATH = "/api/gpu/release"

# One stderr line per distinct warning, so a long-running caller (a batch
# loop, a kept-open module) does not spam its own log every time it asks.
_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    """Print one stderr line for a distinct warning, matching lock_status.py.

    Parameters:
        msg: the warning text; a "gpu_claim: " prefix is added here so every
            call site keeps this consistent.

    Returns: None.
    """
    if msg not in _warned:
        _warned.add(msg)
        print(f"gpu_claim: {msg}", file=sys.stderr)


def _unregistered(context: str, reason: str) -> dict:
    """Build the "could not ask VICARIUS" fallback decision and warn once.

    Parameters:
        context: who was asking, for the warning: a module slug for check()
            and claim(), or a short label for release(), which carries no
            module of its own.
        reason: what went wrong, kept out of the canonical warning line and
            carried separately in the returned dict for the caller's own log.

    Returns: {"decision": "unregistered", "reason": reason}.
    """
    _warn_once(
        f"VICARIUS on 5090 did not answer; {context} is proceeding "
        f"unregistered and will show as unclaimed in the GPU monitor when "
        f"the UI returns ({reason})."
    )
    return {"decision": "unregistered", "reason": reason}


def _post(path: str, body: dict, context: str) -> dict:
    """POST one JSON body to the desktop UI and hand back its decision.

    An HTTP 409 or 428 (or any other status) whose body is a JSON object is
    a decision from the server, not a failure of this call, and is returned
    unchanged. A connection failure, a timeout, or a reply this function
    cannot parse as a JSON object falls back to decision "unregistered".

    Parameters:
        path: the route to POST to, one of CHECK_PATH, CLAIM_PATH,
            RELEASE_PATH.
        body: the JSON-serialisable request body.
        context: who is asking, used only in the "unregistered" warning.

    Returns: the parsed JSON object the server sent, or
    {"decision": "unregistered", "reason": "..."}. Never raises.
    """
    if os.environ.get(BYPASS_ENV) == "1":
        return _unregistered(context, f"{BYPASS_ENV}=1 is set")
    url = BASE_URL.rstrip("/") + path
    try:
        payload = json.dumps(body).encode("utf-8")
    except TypeError as exc:
        return _unregistered(context, f"could not encode the request body ({exc})")
    request = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError as decode_exc:
            return _unregistered(
                context, f"HTTP {exc.code} with an unreadable body ({decode_exc})")
        if isinstance(parsed, dict):
            return parsed
        return _unregistered(context, f"HTTP {exc.code} did not answer with a JSON object")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _unregistered(context, str(exc))
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        return _unregistered(context, f"an unreadable reply ({exc})")
    if not isinstance(parsed, dict):
        return _unregistered(context, "a reply that was not a JSON object")
    return parsed


def check(module: str, tier: "str | None" = None, by: str = "LO", waiting: bool = False,
          cards_wanted: "str | None" = None, pid: "int | None" = None) -> dict:
    """Ask VICARIUS whether `module` may claim a card right now.

    Advisory only: writes nothing to the claims registry, except that a
    non-proceeding answer with `waiting` True is remembered for the GPU
    traffic matrix's yellow "waiting" state (spec docs/superpowers/specs/
    2026-09-28-gpu-traffic-design.md section 9).

    Parameters:
        module: the calling module's slug, for example "3D_phase_1".
        tier: the right-of-way tier to ask about; omit to let VICARIUS
            derive it from the module's entry in the ladder.
        by: initials of the person or process this check is made for.
        waiting: True when the caller intends to keep asking every so
            often until it is allowed to start (for example the driver's
            own 60 second retry); False (the default) behaves exactly as
            before this parameter existed.
        cards_wanted: "one" or "all", overriding the GPU traffic row's own
            default for this one request (a stage that only ever uses a
            single card); omit to use the row's default.
        pid: the checking process's own id, so a claim this same run
            already holds for `module` (for example the runner's own
            post-spawn claim, made moments before this same run's startup
            check) is excluded from its own decision rather than counted
            as a competing holder (code review item 11); omit when the
            caller has no pid of its own to offer, which checks the whole
            registry unchanged, exactly as before this parameter existed.

    Returns: the decision dict VICARIUS sends, one of
    {"decision": "granted", "holders": [...], "message": "..."},
    {"decision": "refused", "holders": [...], "message": "..."},
    {"decision": "needs_pin", "holders": [...], "message": "..."},
    {"decision": "wait", "holders": [...], "message": "..."} (a manual edit
    window holds the memory; ask again later), or
    {"decision": "unregistered", "reason": "..."} when port 5090 could not
    be reached or its reply could not be read. Any other decision word is
    passed through unchanged too. Every answer but "unregistered" also
    carries the GPU traffic fields "row", "cards", "env", "device" and,
    when applicable, "notice" (see launch_env, card_uuids, device_string).
    Start only on "granted" or "unregistered"; every other answer means do
    not start. Never raises.
    """
    body = {"module": module, "tier": tier, "by": by}
    if waiting:
        body["waiting"] = True
    if cards_wanted is not None:
        body["cards_wanted"] = cards_wanted
    if pid is not None:
        body["pid"] = pid
    return _post(CHECK_PATH, body, context=str(module))


def claim(module: str, tier: "str | None" = None, run_id: str = "",
          pid: "int | None" = None, by: str = "LO", bypass: bool = False,
          reason: str = "", cards: "list[str] | None" = None,
          cards_wanted: "str | None" = None) -> dict:
    """Ask VICARIUS to record a claim on a card for `module`.

    Binding: on a "granted" decision, VICARIUS writes the claim to the
    registry (and sweeps any lower-tier holder) before answering.

    Parameters:
        module: the calling module's slug.
        tier: the right-of-way tier to claim; omit to let VICARIUS derive it
            from the module's entry in the ladder.
        run_id: the run's own identifier, shown to anyone this claim refuses.
        pid: the process id that will hold the card; VICARIUS requires it to
            be alive.
        by: initials of the person or process this claim is made for.
        bypass: True to use an admin PIN override on a "needs_pin" decision;
            VICARIUS requires a non-empty reason and an admin session when
            this is set.
        reason: the one-line reason shown and audited for a bypass.
        cards: the card UUIDs a launcher already spawned its process on
            (spec docs/superpowers/specs/2026-09-28-gpu-traffic-design.md
            section 6, "a claim sent after spawn"); typically
            card_uuids(answer) from the check() this claim follows. Omit
            to let VICARIUS choose the go cards itself.
        cards_wanted: "one" or "all", overriding the GPU traffic row's own
            default for this one request; ignored together with `cards`.

    Returns: the decision dict VICARIUS sends, one of
    {"decision": "granted", "claim_id": "...", "swept": [...]},
    {"decision": "refused", "holders": [...], "message": "..."},
    {"decision": "needs_pin", "holders": [...], "message": "..."},
    {"decision": "wait", "holders": [...], "message": "..."} (HTTP 409; no
    bypass changes it), or
    {"decision": "unregistered", "reason": "..."} when port 5090 could not
    be reached or its reply could not be read. Every answer but
    "unregistered" also carries the GPU traffic fields "row", "cards",
    "env", "device" and, when applicable, "notice". Never raises.
    """
    body = {"module": module, "tier": tier, "run_id": run_id, "pid": pid,
            "by": by, "bypass": bypass, "reason": reason}
    if cards is not None:
        body["cards"] = cards
    if cards_wanted is not None:
        body["cards_wanted"] = cards_wanted
    return _post(CLAIM_PATH, body, context=str(module))


_LAUNCH_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")


def launch_env(answer: dict) -> "dict[str, str]":
    """The environment variables a launcher should merge onto a spawned child.

    Parameters:
        answer: a check() or claim() answer.
    Returns:
        Only the keys CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER out of
        answer["env"], each kept only when its value is a string; any
        other key answer["env"] carries is dropped with one warning (code
        review item 9: a launcher merges this straight onto os.environ, so
        an answer that grew an unexpected key must never leak it onto a
        spawned child's environment). {} when "env" is absent, not a
        mapping, or carries neither known key (an "unregistered" answer,
        an answer from before GPU traffic shipped, or a hostile shape), so
        a caller never needs to guard this itself (plan docs/superpowers/
        plans/2026-09-28-gpu-traffic.md, "Shared contract"). Never raises.
    """
    env = answer.get("env") if isinstance(answer, dict) else None
    if not isinstance(env, dict):
        return {}
    extra = sorted(set(env) - set(_LAUNCH_ENV_KEYS))
    if extra:
        _warn_once(f"launch_env: ignoring unknown env key(s) {extra} in a check/claim answer")
    return {key: env[key] for key in _LAUNCH_ENV_KEYS if isinstance(env.get(key), str)}


def card_uuids(answer: dict) -> "list[str]":
    """The go cards' UUIDs from a check() or claim() answer, to send on a later claim.

    Parameters:
        answer: a check() or claim() answer.
    Returns:
        The "uuid" of each entry of answer["cards"], in order; [] when
        "cards" is absent, not a list, or holds an entry with no string
        "uuid". Never raises.
    """
    cards = answer.get("cards") if isinstance(answer, dict) else None
    if not isinstance(cards, list):
        return []
    return [card["uuid"] for card in cards if isinstance(card, dict) and isinstance(card.get("uuid"), str)]


def device_string(answer: dict) -> "str | None":
    """The go cards' device string from a check() or claim() answer.

    Parameters:
        answer: a check() or claim() answer.
    Returns:
        answer["device"] when it is a non-empty string; None otherwise (no
        card is go, or the answer predates GPU traffic). Never raises.
    """
    device = answer.get("device") if isinstance(answer, dict) else None
    return device if isinstance(device, str) and device else None


def release(claim_id: "str | None" = None, pid: "int | None" = None) -> dict:
    """Tell VICARIUS a claim is finished, freeing the card for the next start.

    Parameters:
        claim_id: the id claim() returned; the usual way to release.
        pid: the process id the claim was made for, when the claim id was
            lost. At least one of claim_id or pid should identify the claim.

    Returns: the JSON object VICARIUS sends back (its /api/gpu/release route
    answers 200 when the claim existed and 404 otherwise, both as a JSON
    body), or {"decision": "unregistered", "reason": "..."} when port 5090
    could not be reached or its reply could not be read. Never raises.
    """
    body = {}
    if claim_id is not None:
        body["claim_id"] = claim_id
    if pid is not None:
        body["pid"] = pid
    context = f"the release of {claim_id if claim_id is not None else pid}"
    return _post(RELEASE_PATH, body, context=context)


class GpuHeldError(Exception):
    """Signals that a module's own client is deliberately refusing GPU work.

    check(), claim() and release() above never raise this or anything else;
    it exists for a module's higher-level code (for example Vesper.llm's
    OllamaClient, which holds itself idle rather than dying when swept) to
    carry the held notice through its own call stack as one exception type,
    so every caller of that code has one thing to catch.

    Parameters:
        notice: the message a person waiting on the call should see.
    """

    def __init__(self, notice: str = ""):
        super().__init__(notice)
        self.notice = notice
