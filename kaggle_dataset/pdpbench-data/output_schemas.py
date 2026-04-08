"""Structured output schemas for LLM responses across all PDPBench tasks."""

from dataclasses import dataclass


@dataclass
class MaskedNodePrediction:
    """Task 1: Predict the masked node in a route."""
    predicted_node: int
    reasoning: str


@dataclass
class RequestInsertionResult:
    """Task 2: Return full routes after inserting removed requests."""
    routes: list[list[int]]
    reasoning: str


@dataclass
class DistancePredictionResult:
    """Task 3: Predict the total distance of a given solution."""
    predicted_distance: float
    reasoning: str


@dataclass
class RouteCompletionResult:
    """Task 4: Return the completed route."""
    completed_route: list[int]
    reasoning: str


@dataclass
class FullSolutionResult:
    """Task 5: Return a complete PDPTW solution."""
    routes: list[list[int]]
    reasoning: str
