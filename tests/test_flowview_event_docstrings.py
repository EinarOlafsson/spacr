"""The public FlowView event payloads document every serialised field."""

from dataclasses import fields

from spacr.flowview import events


PUBLIC_EVENTS = (
    events.NodeAdded,
    events.EdgeAdded,
    events.StageStarted,
    events.StageProgress,
    events.StageMetric,
    events.StageThumbnail,
    events.StageCompleted,
    events.StageFailed,
)


def test_every_public_flow_event_field_is_documented():
    """A consumer can interpret each pickled value from its API prose."""
    missing = [
        f"{event.__name__}.{field.name}"
        for event in PUBLIC_EVENTS
        for field in fields(event)
        if f":param {field.name}:" not in (event.__doc__ or "")
    ]
    assert not missing, f"undocumented FlowView event fields: {missing}"
