import attrs

@attrs.define(slots=True, eq=False, order=False)
class Track:
    """
    A track object is associated with a set of animal/object instances
    across multiple frames of video. This allows tracking of unique
    entities in the video over time and space.

    Args:
        spawned_on: The video frame that this track was spawned on.
        name: A name given to this track for identifying purposes.
    """

    spawned_on: int = attrs.field(default=0, converter=int)
    name: str = attrs.field(default="", converter=str)

    def matches(self, other: "Track"):
        """
        Check if two tracks match by value.

        Args:
            other: The other track to check

        Returns:
            True if they match, False otherwise.
        """
        return attrs.asdict(self) == attrs.asdict(other)