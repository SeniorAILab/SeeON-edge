"""Bounded scalar primitives for the PerceptionFrameV1 wire codec."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final, cast, final

MAGIC: Final = b"PFV1"
MAX_ITEMS: Final = 256
MAX_TEXT: Final = 128
_U16: Final = struct.Struct("<H")
_F64: Final = struct.Struct("<d")


@dataclass(frozen=True, slots=True)
class PerceptionWireError(Exception):
    code: str
    detail: str


@final
class Writer:
    """Bounded byte accumulator; mutation is its sole purpose."""

    def __init__(self) -> None:
        self.value = bytearray(MAGIC)

    def raw(self, value: bytes) -> None:
        self.value.extend(value)

    def u8(self, value: int) -> None:
        self.value.append(value)

    def u16(self, value: int) -> None:
        if not 0 <= value <= 65_535:
            raise PerceptionWireError("count_bounds", str(value))
        self.raw(_U16.pack(value))

    def text(self, value: str) -> None:
        encoded = value.encode()
        if len(encoded) > MAX_TEXT:
            raise PerceptionWireError("text_bounds", str(len(encoded)))
        self.u16(len(encoded))
        self.raw(encoded)


@final
class Reader:
    """Bounds-checking cursor; mutation advances one trusted parse."""

    def __init__(self, value: bytes) -> None:
        self.value = value
        self.offset = 0

    def raw(self, size: int) -> bytes:
        end = self.offset + size
        if size < 0 or end > len(self.value):
            raise PerceptionWireError("payload_truncated", str(self.offset))
        result = self.value[self.offset : end]
        self.offset = end
        return result

    def u8(self) -> int:
        return self.raw(1)[0]

    def u16(self, *, maximum: int = MAX_ITEMS) -> int:
        value = int.from_bytes(self.raw(2), "little")
        if value > maximum:
            raise PerceptionWireError("count_bounds", str(value))
        return value

    def u64(self) -> int:
        return int.from_bytes(self.raw(8), "little")

    def i64(self) -> int:
        return int.from_bytes(self.raw(8), "little", signed=True)

    def i32(self) -> int:
        return int.from_bytes(self.raw(4), "little", signed=True)

    def f64(self) -> float:
        return cast(float, _F64.unpack(self.raw(_F64.size))[0])

    def text(self) -> str:
        size = self.u16(maximum=MAX_TEXT)
        try:
            return self.raw(size).decode()
        except UnicodeDecodeError as error:
            raise PerceptionWireError("text_encoding", str(error)) from error


__all__ = ["MAGIC", "PerceptionWireError", "Reader", "Writer"]
