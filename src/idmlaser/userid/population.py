"""This module contains the Population class, a collection of grouped communities."""

import heapq
from collections import namedtuple
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict

from .community import Community

ScheduledEvent = namedtuple("ScheduledEvent", ["func", "nodes", "args", "kwargs"])


@dataclass(order=True)
class ScheduledItem:
    tick: int
    eid: int
    event: ScheduledEvent = field(compare=False)


class Population:
    """A collection of grouped communities."""

    eid = 0

    def __init__(self, num_communities, community_props, agent_groups, agent_props):
        self.num_communities = num_communities
        self._communities = []
        self.community_props = community_props
        self.agent_groups = agent_groups
        self.agent_props = agent_props
        self.queue = []

        return

    def realize(self, callback):
        """Realize the set communities using the callback to initialize each community in turn."""
        for i in range(self.num_communities):
            community = Community()
            callback(self, community, i)
            self._communities.append(community)

        return

    def add_population_property(self, name: str, value: Any) -> None:
        """Add a property to the class."""
        setattr(self, name, value)
        return

    def allocate_community(self, community: Community, pops: Dict[str, int]) -> None:
        """Allocate memory for a community."""
        # for name in self.community_props:
        #     community.add_community_property(name, props[name] if name in props else None)
        for name in self.agent_groups:
            community.add_agent_group(name, pops[name] if name in pops else 0)
        for prop in self.agent_props:
            community.add_agent_property(*prop)
        community.allocate()
        return

    @property
    def communities(self):
        """Return the communities."""
        return self._communities

    def apply(self, callback, *args, **kwargs):
        """Apply the callback to each community in turn."""
        for index, community in enumerate(self._communities):
            callback(self, community, index, *args, **kwargs)
        return

    def add_event(self, event: ScheduledEvent, tick: int):
        """Add an event to each community."""
        heapq.heappush(self.queue, ScheduledItem(tick, self.eid, event))
        self.eid += 1
        return

    def do_events(self, tick):
        """Do scheduled events for this tick."""
        while self.queue and (self.queue[0].tick == tick):
            event = heapq.heappop(self.queue).event
            for index, community in enumerate(self._communities):
                if (not event.nodes) or (index in event.nodes):
                    event.func(self, community, index, *event.args, **event.kwargs)
        return
