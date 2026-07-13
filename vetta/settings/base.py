"""Base settings model for the private refactor path."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict


class VettaSettings(BaseModel):
    """Strict Pydantic base for internal configuration objects."""

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    @classmethod
    def field_names(cls) -> set[str]:
        """Return the set of declared settings field names."""

        return set(cls.model_fields)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Self:
        """Validate a complete config dictionary into typed settings."""

        if not isinstance(config, Mapping):
            raise TypeError(
                "{}.from_dict() expected a mapping, got {}".format(
                    cls.__name__,
                    type(config).__name__,
                )
            )
        expected = cls.field_names()
        received = set(config.keys())
        unknown = sorted(received - expected)
        missing = sorted(expected - received)
        if unknown:
            raise ValueError("{} received unknown config keys: {}".format(cls.__name__, unknown))
        if missing:
            raise ValueError("{} missing required config keys: {}".format(cls.__name__, missing))
        values = {name: copy.deepcopy(config[name]) for name in expected}
        return cls.model_validate(values)

    @classmethod
    def from_defaults(cls, overrides: Mapping[str, Any] | None = None) -> Self:
        """Build settings from defaults plus an optional override mapping."""

        values = {
            name: copy.deepcopy(field.get_default(call_default_factory=True))
            for name, field in cls.model_fields.items()
            if not field.is_required()
        }
        if overrides:
            unknown = sorted(set(overrides.keys()) - cls.field_names())
            if unknown:
                raise ValueError("{} received unknown config keys: {}".format(cls.__name__, unknown))
            values.update(copy.deepcopy(dict(overrides)))
        return cls.from_dict(values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | Self) -> Self:
        """Validate a mapping or return an existing settings instance."""

        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise TypeError(
                "{}.from_mapping() expected a mapping or {} instance, got {}".format(
                    cls.__name__,
                    cls.__name__,
                    type(values).__name__,
                )
            )
        return cls.model_validate(values)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, mutable Python dict of the settings."""

        return copy.deepcopy(self.model_dump(mode="python"))

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict suitable for summaries and checkpoints."""

        return copy.deepcopy(self.model_dump(mode="json"))
