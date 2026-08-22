"""Fit a versioned Phase 3 knowledge seed from raw ``/matches`` replays.

Usage::

    python -m app.phase3.showdown.replay replay-a.json replay-b.json \
        --output app/phase3/showdown/knowledge.seed.json

The importer is deliberately tolerant of snake_case/camelCase wrappers.  Raw
files are never copied into the source tree; only aggregate evidence is written.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .learning import EventKnowledge


@dataclass(frozen=True, slots=True)
class ReplayFitReport:
    files_seen: int
    files_added: int
    hands_added: int
    duplicate_files: int

    def to_dict(self) -> dict[str, int]:
        return {
            "files_seen": self.files_seen,
            "files_added": self.files_added,
            "hands_added": self.hands_added,
            "duplicate_files": self.duplicate_files,
        }


@dataclass(frozen=True, slots=True)
class _ReplayRecord:
    file_hash: str
    observation_source: str
    hand_key: str
    rule: str
    hand: Mapping[str, Any]
    players_by_seat: Mapping[int, Mapping[str, Any] | str]
    full_numbers: Mapping[int, int]
    your_seat: int | None
    small_blind: int
    big_blind: int


def _record_signature(record: _ReplayRecord) -> str:
    """Return a stable semantic fingerprint, excluding file provenance."""

    payload = {
        "rule": record.rule,
        "hand": record.hand,
        "players": record.players_by_seat,
        "full_numbers": record.full_numbers,
        "your_seat": record.your_seat,
        "small_blind": record.small_blind,
        "big_blind": record.big_blind,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _channel_key(observation_key: str, channel: str) -> str:
    return f"{observation_key}:{channel}"


def _channel_done(knowledge: EventKnowledge, observation_key: str, channel: str) -> bool:
    # Seeds produced before channel-specific replay deduplication stored the base
    # key.  Treat those entries as complete for backwards-compatible idempotence.
    return observation_key in knowledge.observation_keys or _channel_key(
        observation_key, channel
    ) in knowledge.observation_keys


def _value(item: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in item:
            return item[name]
    return default


def _is_array(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _players(value: Any) -> dict[int, Mapping[str, Any] | str]:
    result: dict[int, Mapping[str, Any] | str] = {}
    if isinstance(value, Mapping):
        for raw_seat, player in value.items():
            try:
                if isinstance(raw_seat, bool):
                    continue
                result[int(raw_seat)] = player
            except (TypeError, ValueError):
                continue
        return result
    if not _is_array(value):
        return result
    for index, player in enumerate(value):
        if not isinstance(player, Mapping):
            continue
        try:
            raw_seat = _value(player, "seat", "seat_number", "seatNumber", default=index)
            if isinstance(raw_seat, bool):
                continue
            seat = int(raw_seat)
        except (TypeError, ValueError):
            continue
        result[seat] = player
    return result


def _number_map(value: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    if isinstance(value, Mapping):
        for raw_seat, raw_number in value.items():
            if isinstance(raw_seat, bool):
                continue
            if isinstance(raw_number, Mapping):
                raw_number = _value(
                    raw_number,
                    "number",
                    "private_number",
                    "privateNumber",
                    "card",
                )
            try:
                if isinstance(raw_number, bool):
                    continue
                seat, number = int(raw_seat), int(raw_number)
            except (TypeError, ValueError):
                continue
            if seat >= 0 and 1 <= number <= 13:
                result[seat] = number
        return result
    if _is_array(value):
        for index, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                continue
            try:
                raw_seat = _value(entry, "seat", "seat_number", "seatNumber", default=index)
                raw_number = _value(
                    entry,
                    "number",
                    "private_number",
                    "privateNumber",
                    "card",
                )
                if isinstance(raw_seat, bool) or isinstance(raw_number, bool):
                    continue
                seat = int(raw_seat)
                number = int(raw_number)
            except (TypeError, ValueError):
                continue
            if seat >= 0 and 1 <= number <= 13:
                result[seat] = number
    return result


def _full_numbers(hand: Mapping[str, Any]) -> dict[int, int]:
    for key in (
        "full_numbers",
        "fullNumbers",
        "dealt_numbers",
        "dealtNumbers",
        "private_numbers",
        "privateNumbers",
        "numbers",
        "cards",
    ):
        found = _number_map(hand.get(key))
        if found:
            return found
    return _number_map(hand.get("players"))


def _shown_numbers(hand: Mapping[str, Any], full: Mapping[int, int]) -> dict[int, int]:
    for key in ("shown_numbers", "shownNumbers", "revealed_numbers", "revealedNumbers"):
        found = _number_map(hand.get(key))
        if found:
            return found

    # Some raw exports mark each seat rather than providing a separate mapping.
    players = hand.get("players")
    if _is_array(players) or isinstance(players, Mapping):
        shown: dict[int, int] = {}
        entries = (
            list(players.items())
            if isinstance(players, Mapping)
            else list(enumerate(players))
        )
        for raw_index, player in entries:
            if not isinstance(player, Mapping):
                continue
            explicitly_shown = bool(
                _value(player, "shown", "revealed", "at_showdown", "atShowdown", default=False)
            )
            if not explicitly_shown:
                continue
            try:
                raw_seat = _value(player, "seat", default=raw_index)
                if isinstance(raw_seat, bool):
                    continue
                seat = int(raw_seat)
            except (TypeError, ValueError):
                continue
            if seat in full:
                shown[seat] = full[seat]
        if shown:
            return shown
    return {}


def _winners(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = [value]
    elif not _is_array(value):
        value = [value]
    result: list[int] = []
    for winner in value:
        if isinstance(winner, Mapping):
            winner = _value(
                winner,
                "seat",
                "player_seat",
                "playerSeat",
                "player_id",
                "playerId",
                "player",
                "id",
            )
        try:
            if isinstance(winner, bool):
                continue
            seat = int(winner)
        except (TypeError, ValueError):
            continue
        if seat >= 0 and seat not in result:
            result.append(seat)
    return result


def _actions(value: Any) -> list[dict[str, Any]]:
    if not _is_array(value):
        return []
    result: list[dict[str, Any]] = []
    for action in value:
        if not isinstance(action, Mapping):
            continue
        raw_seat = _value(action, "seat", "player_seat", "playerSeat")
        if isinstance(raw_seat, bool):
            continue
        normalized = {
            "round": _value(action, "round", "street"),
            "seat": raw_seat,
            "action": _value(action, "action", "type"),
        }
        for target, names in {
            "amount": ("amount", "raise_to", "raiseTo"),
            "pot_before": ("pot_before", "potBefore"),
            "to_call": ("to_call", "toCall"),
            "live_players": ("live_players", "livePlayers"),
            "position": ("position",),
        }.items():
            found = _value(action, *names)
            if found is not None:
                normalized[target] = found
        result.append(normalized)
    return result


def _iter_match_containers(
    node: Any,
    inherited_rule: str | None = None,
    inherited_players: dict[int, Mapping[str, Any] | str] | None = None,
    inherited_your_seat: int | None = None,
) -> Iterable[
    tuple[
        str,
        Mapping[str, Any],
        Sequence[Any],
        dict[int, Mapping[str, Any] | str],
        int | None,
    ]
]:
    """Yield containers that own a hands array without double-walking it."""

    if isinstance(node, Mapping):
        raw_rule = _value(node, "table_rule", "tableRule", default=inherited_rule)
        rule = str(raw_rule).strip() if raw_rule is not None else inherited_rule
        local_players = _players(node.get("players")) or inherited_players or {}
        raw_your_seat = _value(
            node, "your_seat", "yourSeat", default=inherited_your_seat
        )
        try:
            local_your_seat = (
                int(raw_your_seat)
                if raw_your_seat is not None and not isinstance(raw_your_seat, bool)
                else inherited_your_seat
            )
        except (TypeError, ValueError):
            local_your_seat = inherited_your_seat
        hands = _value(node, "hands", "hand_history", "handHistory")
        if rule and _is_array(hands):
            yield rule, node, hands, local_players, local_your_seat
            return
        for value in node.values():
            yield from _iter_match_containers(
                value, rule, local_players, local_your_seat
            )
    elif _is_array(node):
        for value in node:
            yield from _iter_match_containers(
                value, inherited_rule, inherited_players, inherited_your_seat
            )


def _normalise_hand(hand: Mapping[str, Any], index: int) -> tuple[dict[str, Any], dict[int, int]]:
    full = _full_numbers(hand)
    shown = _shown_numbers(hand, full)
    normalized = {
        "hand_number": _value(hand, "hand_number", "handNumber", default=index + 1),
        "community_number": _value(
            hand, "community_number", "communityNumber", "community"
        ),
        "winners": _winners(_value(hand, "winners", "winner_seats", "winnerSeats", default=[])),
        "pot": _value(hand, "pot", "pot_size", "potSize", default=0),
        "shown_numbers": shown,
        "actions": _actions(_value(hand, "actions", "action_history", "actionHistory", default=[])),
        "button_seat": _value(hand, "button_seat", "buttonSeat"),
    }
    if _value(hand, "side_pots", "sidePots", "multiple_pots", "multiplePots"):
        normalized["side_pots"] = True
    return normalized, full


def load_seed(path: str | Path | None) -> EventKnowledge:
    if path is None:
        return EventKnowledge()
    seed = Path(path)
    if not seed.is_file():
        raise FileNotFoundError(f"seed file does not exist: {seed}")
    try:
        data = json.loads(seed.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"seed file is not valid JSON: {seed}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("seed root must be a JSON object")
    version = data.get("version", 1)
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"unsupported knowledge seed version: {version!r}")
    return EventKnowledge.from_dict(data)


def fit_replays(
    paths: Iterable[str | Path],
    *,
    seed_path: str | Path | None = None,
) -> tuple[EventKnowledge, ReplayFitReport]:
    knowledge = load_seed(seed_path)
    files_seen = files_added = hands_added = duplicates = 0
    pending: dict[str, _ReplayRecord] = {}
    pending_hashes: set[str] = set()
    file_record_keys: dict[str, set[str]] = {}

    for raw_path in paths:
        files_seen += 1
        path = Path(raw_path)
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash in knowledge.source_hashes or source_hash in pending_hashes:
            duplicates += 1
            continue
        pending_hashes.add(source_hash)
        document = json.loads(raw)
        file_record_keys[source_hash] = set()
        for container_index, (
            rule,
            container,
            hands,
            players_by_seat,
            inherited_your_seat,
        ) in enumerate(
            _iter_match_containers(document)
        ):
            your_seat_raw = _value(container, "your_seat", "yourSeat")
            try:
                your_seat = (
                    int(your_seat_raw)
                    if your_seat_raw is not None and not isinstance(your_seat_raw, bool)
                    else inherited_your_seat
                )
            except (TypeError, ValueError):
                your_seat = None
            identity = _value(
                container,
                "match_id",
                "matchId",
                "run_id",
                "runId",
                "id",
                default=f"{path.resolve()}:{container_index}",
            )
            leg_identity = _value(container, "leg_number", "legNumber")
            if leg_identity is not None and not isinstance(leg_identity, bool):
                identity = f"{identity}:leg-{leg_identity}"
            observation_source = f"replay-v1:{rule}:{identity}"
            raw_small_blind = _value(container, "small_blind", "smallBlind", default=1)
            raw_big_blind = _value(container, "big_blind", "bigBlind", default=2)
            try:
                if isinstance(raw_small_blind, bool) or isinstance(raw_big_blind, bool):
                    raise ValueError
                small_blind = int(raw_small_blind)
                big_blind = int(raw_big_blind)
            except (TypeError, ValueError):
                small_blind, big_blind = 1, 2
            if small_blind < 0:
                small_blind = 1
            if big_blind < 0:
                big_blind = 2
            for hand_index, hand in enumerate(hands):
                if not isinstance(hand, Mapping):
                    continue
                normalized, full = _normalise_hand(hand, hand_index)
                hand_key = str(normalized["hand_number"])
                observation_key = f"{observation_source}:{hand_key}"
                file_record_keys[source_hash].add(observation_key)
                if _channel_done(knowledge, observation_key, "rule") and _channel_done(
                    knowledge, observation_key, "opponents"
                ):
                    continue
                record = _ReplayRecord(
                    file_hash=source_hash,
                    observation_source=observation_source,
                    hand_key=hand_key,
                    rule=rule,
                    hand=normalized,
                    players_by_seat=(players_by_seat or _players(hand.get("players"))),
                    full_numbers=full,
                    your_seat=your_seat,
                    small_blind=small_blind,
                    big_blind=big_blind,
                )
                existing = pending.get(observation_key)
                if existing is not None and _record_signature(existing) != _record_signature(
                    record
                ):
                    raise ValueError(
                        "conflicting replay records share identity "
                        f"{observation_key!r}; provide a distinct match/run id"
                    )
                pending.setdefault(observation_key, record)

    # Two deterministic passes make strength-bucket training independent of
    # replay input order: first learn the rule, then classify opponent actions
    # under the final posterior for this batch.
    rule_results: dict[str, bool] = {}
    for key, record in sorted(pending.items()):
        if _channel_done(knowledge, key, "rule"):
            rule_results[key] = False
            continue
        rule_results[key] = knowledge.observe_hand(
            record.rule,
            record.hand,
            record.players_by_seat,
            record.full_numbers,
            your_seat=record.your_seat,
            small_blind=record.small_blind,
            big_blind=record.big_blind,
            learn_rules=True,
            learn_opponents=False,
        )
        if rule_results[key]:
            knowledge.observation_keys.add(_channel_key(key, "rule"))

    learned_keys: set[str] = {
        key for key, learned in rule_results.items() if learned
    }
    for key, record in sorted(pending.items()):
        if _channel_done(knowledge, key, "opponents"):
            continue
        learned_profiles = knowledge.observe_hand(
            record.rule,
            record.hand,
            record.players_by_seat,
            record.full_numbers,
            your_seat=record.your_seat,
            small_blind=record.small_blind,
            big_blind=record.big_blind,
            learn_rules=False,
            learn_opponents=True,
        )
        if learned_profiles:
            knowledge.observation_keys.add(_channel_key(key, "opponents"))
            learned_keys.add(key)

    hands_added = len(learned_keys)
    for source_hash, record_keys in file_record_keys.items():
        if any(
            _channel_done(knowledge, key, "rule")
            or _channel_done(knowledge, key, "opponents")
            for key in record_keys
        ):
            knowledge.source_hashes.add(source_hash)
            files_added += 1

    return knowledge, ReplayFitReport(
        files_seen=files_seen,
        files_added=files_added,
        hands_added=hands_added,
        duplicate_files=duplicates,
    )


def write_seed(knowledge: EventKnowledge, output: str | Path) -> None:
    """Atomically write aggregate knowledge; runtime code never calls this."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(knowledge.to_dict(), indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", help="raw Phase 3 /matches JSON files")
    parser.add_argument("--seed", help="existing seed to extend")
    parser.add_argument("--output", required=True, help="destination knowledge seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    knowledge, report = fit_replays(args.replays, seed_path=args.seed)
    write_seed(knowledge, args.output)
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "ReplayFitReport",
    "fit_replays",
    "load_seed",
    "write_seed",
    "main",
]
