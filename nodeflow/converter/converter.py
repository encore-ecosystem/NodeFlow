from nodeflow.adapter import Adapter
from typing import Iterable, Type, Optional
from collections import deque

from nodeflow.adapter.pipeline import Pipeline
from nodeflow.node.variable import Variable


class Converter:
    ROOT_CONVERTER: Optional['Converter'] = None

    def __init__(self, adapters: Optional[Iterable[Adapter]] = None, sub_converters: Optional[Iterable['Converter']] = None):
        self.graph          = {}
        self.sub_converters = set()

        (adapters is not None) and self.register_adapters(adapters)
        (sub_converters is not None) and self.register_converters(sub_converters)

    #
    # Adapters handlers
    #
    def register_adapter(self, adapter: Adapter):
        # Resolve source type
        source_type = adapter.get_type_of_source_variable().__name__
        if source_type not in self.graph:
            self.graph[source_type] = {}

        # Resolve target type
        target_type = adapter.get_type_of_target_variable().__name__
        self.graph[source_type][target_type] = adapter

    def register_adapters(self, adapters: Iterable[Adapter]):
        for adapter in adapters:
            self.register_adapter(adapter)

    #
    # Converters handlers
    #
    def register_converter(self, converter: 'Converter'):
        self.sub_converters.add(converter)

    def register_converters(self, converters: Iterable['Converter']):
        for converter in converters:
            self.register_converter(converter)

    def is_support_variable(self, variable_type: Type[Variable]) -> bool:
        return variable_type in self.graph

    def convert(self, variable: Variable, to_type: Type[Variable]) -> Optional[Variable]:
        pipeline, is_safe = self.get_converting_pipeline(source=variable.__class__, target=to_type)
        assert pipeline is not None, "Could not convert variable"
        return pipeline.compute(variable)

    def get_converting_pipeline(self, source: Type, target: Type) -> tuple[Optional[Pipeline], bool]:
        pipeline_with_loses_information : Optional[Pipeline] = None
        # ---------
        # BFS
        # ---------
        visited = set()
        queue   = deque()
        queue.append([source.__name__, [source.__name__]])

        while len(queue) > 0:
            root_type, type_road = queue.popleft()
            visited.add(root_type)
            for child in self.graph[root_type]:
                if child in visited:
                    continue
                if child == target.__name__:
                    pipeline = Pipeline()
                    road = type_road + [child]
                    for i in range(len(road) - 1):
                        pipeline.add_adapter(self.graph[road[i]][road[i + 1]])

                    if not pipeline_with_loses_information and pipeline.is_loses_information():
                        # the shortest pipeline with loosing an information
                        pipeline_with_loses_information = pipeline
                    else:
                        # the shortest pipeline without loosing an information
                        return pipeline, True
                queue.append([child, type_road + [child]])

        return pipeline_with_loses_information, False

    #
    # Context
    #
    def __enter__(self):
        Converter.ROOT_CONVERTER = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Converter.ROOT_CONVERTER = None


__all__ = [
    "Converter"
]