from nodeflow.adapters.abstract import Adapter
from nodeflow.node.abstract import Variable
from typing import Type


class Pipeline(Adapter):
    def __init__(self):
        self._loose_information        = False
        self._pipeline : list[Adapter] = []

    def add_adapter(self, adapter: Adapter):
        self._pipeline += [adapter]
        self._loose_information = self._loose_information or adapter.is_loose_information()
        if len(self._pipeline) > 1:
            s1 = self._pipeline[-2].get_type_of_target_variable()
            s2 = self._pipeline[-1].get_type_of_source_variable()
            assert s1 == s2, 'Adapter is invalid for current pipeline'

    def convert(self, variable: Variable):
        assert len(self._pipeline) > 0, "Pipeline is empty"
        for adapter in self._pipeline:
            variable = adapter.convert(variable)
        return variable

    def get_type_of_source_variable(self) -> Type[Variable]:
        assert len(self._pipeline) > 0, "Pipeline is empty"
        return self._pipeline[0].get_type_of_source_variable()

    def get_type_of_target_variable(self) -> Type[Variable]:
        assert len(self._pipeline) > 0, "Pipeline is empty"
        return self._pipeline[-1].get_type_of_target_variable()

    def is_loose_information(self) -> bool:
        return self._loose_information

__all__ = ['Pipeline']