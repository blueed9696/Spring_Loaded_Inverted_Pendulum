from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import math


class Phase(Enum):
    FLIGHT = auto()
    STANCE = auto()


@dataclass
class SLIPState:
    x: float
    y: float
    xdot: float
    ydot: float
    phase: Phase
    t: float = 0.0
    step_idx: int = 0
    foot_x: float = 0.0

    def as_vector(self) -> List[float]:
        return [self.x, self.y, self.xdot, self.ydot]

    @classmethod
    def from_vector(
        cls,
        vec: List[float],
        phase: Phase,
        t: float = 0.0,
        step_idx: int = 0,
        foot_x: float = 0.0,
    ) -> "SLIPState":
        x, y, xdot, ydot = vec
        return cls(
            x=x,
            y=y,
            xdot=xdot,
            ydot=ydot,
            phase=phase,
            t=t,
            step_idx=step_idx,
            foot_x=foot_x,
        )


@dataclass
class ApexState:
    x: float
    y: float
    xdot: float
    step_idx: int = 0

    def to_full_state(
        self,
        phase: Phase = Phase.FLIGHT,
        t: float = 0.0,
        foot_x: float = 0.0,
    ) -> SLIPState:
        return SLIPState(
            x=self.x,
            y=self.y,
            xdot=self.xdot,
            ydot=0.0,
            phase=phase,
            t=t,
            step_idx=self.step_idx,
            foot_x=foot_x,
        )


@dataclass
class TouchDownState:
    x: float
    y: float
    xdot: float
    ydot: float
    theta: float
    l0: float
    t: float = 0.0
    step_idx: int = 0

    def as_vector(self) -> List[float]:
        return [self.x, self.y, self.xdot, self.ydot]

    @property
    def foot_x(self) -> float:
        return self.x + self.l0 * math.sin(self.theta)

    @classmethod
    def from_flight_state(
        cls,
        slip_state: SLIPState,
        theta: float,
        l0: float,
    ) -> "TouchDownState":
        y_td = l0 * math.cos(theta)
        return cls(
            x=slip_state.x,
            y=y_td,
            xdot=slip_state.xdot,
            ydot=slip_state.ydot,
            theta=theta,
            l0=l0,
            t=slip_state.t,
            step_idx=slip_state.step_idx,
        )
