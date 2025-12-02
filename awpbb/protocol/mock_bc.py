"""Mock blockchain contract for offline timing experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict


def _token_id(pk_bytes: bytes, encN_bytes: bytes) -> bytes:
    return hashlib.sha256(pk_bytes + encN_bytes).digest()


@dataclass
class TxRecord:
    Xd: bytes
    TX: int
    pkX_RA: bytes
    encN: bytes
    sigRA: int
    sigCP: int
    buyer: str
    used: bool = False


class MockBC:
    def __init__(self) -> None:
        self.tx_by_token: Dict[bytes, TxRecord] = {}

    def register_by_cp(self, record: TxRecord) -> bytes:
        tid = _token_id(record.pkX_RA, record.encN)
        if tid in self.tx_by_token:
            raise ValueError("token already exists")
        self.tx_by_token[tid] = record
        return tid

    def confirm_by_b(self, tid: bytes, record: TxRecord) -> None:
        if tid not in self.tx_by_token:
            raise ValueError("unknown token")
        stored = self.tx_by_token[tid]
        if stored.used:
            raise ValueError("token already used")
        # minimal consistency checks
        if stored.Xd != record.Xd or stored.TX != record.TX:
            raise ValueError("record mismatch")
        if stored.pkX_RA != record.pkX_RA or stored.encN != record.encN:
            raise ValueError("token mismatch")
        stored.used = True


__all__ = ["MockBC", "TxRecord", "_token_id"]
