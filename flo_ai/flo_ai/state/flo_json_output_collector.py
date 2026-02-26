import json
import re
from typing import Callable, Dict, List, Any, Optional

from flo_ai.error.flo_exception import FloException
from flo_ai.common.flo_logger import get_logger
from flo_ai.state.flo_output_collector import FloOutputCollector, CollectionStatus


class FloJsonOutputCollector(FloOutputCollector):
    def __init__(self, strict: bool = False):
        super().__init__()
        self.strict = strict
        self.status = CollectionStatus.success
        self.data: List[Dict[str, Any]] = []

    def append(self, agent_output: str) -> None:
        self.data.append(self.__extract_jsons(agent_output))

    def __strip_comments(self, json_str: str) -> str:
        cleaned = []
        i = 0
        length = len(json_str)

        while i < length:
            char = json_str[i]

            if char not in '"/*':
                cleaned.append(char)
                i += 1
                continue

            if char == '"':
                cleaned.append(char)
                i += 1
                while i < length:
                    char = json_str[i]
                    cleaned.append(char)
                    i += 1
                    if char == '"' and (i < 2 or json_str[i - 2] != '\\'):
                        break
                continue

            if char == '/' and i + 1 < length:
                next_char = json_str[i + 1]

                if next_char == '/':
                    i += 2
                    while i < length and json_str[i] != '\n':
                        i += 1
                    continue
                elif next_char == '*':
                    i += 2
                    while i + 1 < length:
                        if json_str[i] == '*' and json_str[i + 1] == '/':
                            i += 2
                            break
                        i += 1
                    continue

            cleaned.append(char)
            i += 1
        return ''.join(cleaned)

    def __find_balanced_braces(self, text: str) -> List[str]:
        """Find all top-level balanced { ... } blocks in text."""
        matches = []
        i = 0
        length = len(text)
        while i < length:
            if text[i] == '{':
                depth = 0
                start = i
                in_string = False
                while i < length:
                    ch = text[i]
                    if in_string:
                        if ch == '\\':
                            if i + 1 < length:
                                i += 1
                        elif ch == '"':
                            in_string = False
                    else:
                        if ch == '"':
                            in_string = True
                        elif ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                matches.append(text[start:i + 1])
                                i += 1
                                break
                    i += 1
            else:
                i += 1
        return matches

    def __extract_jsons(self, llm_response: str) -> Dict[str, Any]:
        json_matches = self.__find_balanced_braces(llm_response)
        json_object: Dict[str, Any] = {}
        for json_str in json_matches:
            try:
                json_obj = json.loads(self.__strip_comments(json_str))
                json_object.update(json_obj)
            except json.JSONDecodeError as e:
                self.status = CollectionStatus.partial
                get_logger().error(f'Invalid JSON in response: {json_str}, {e}')
        if self.strict and not json_matches:
            self.status = CollectionStatus.error
            get_logger().error(f'Error while finding json in -- {llm_response}')
            raise FloException(
                'JSON response expected in collector model: strict', error_code=1099
            )
        return json_object

    def pop(self) -> Dict[str, Any]:
        return self.data.pop()

    def peek(self) -> Optional[Dict[str, Any]]:
        return self.data[-1] if self.data else None

    def fetch(self) -> Dict[str, Any]:
        return self.__merge_data()

    def __merge_data(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for d in self.data:
            result.update(d)
        return result

    def rewind(
        self,
        then_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        depth: Optional[int] = None,
    ) -> None:
        """
        Replay memory entries newest→oldest, invoking `then_callback` per step.
        :param then_callback: function to handle each entry
        :param depth: max number of entries to process
        """
        if not self.data:
            get_logger().warning('No memory to rewind.')
            return

        entries = self.data[::-1]
        if depth is not None:
            entries = entries[:depth]

        def _recursive(idx: int) -> None:
            if idx >= len(entries):
                return
            if then_callback:
                then_callback(entries[idx])
            _recursive(idx + 1)

        _recursive(0)

    def iter_q(self, depth: Optional[int] = None) -> 'FloIterator':
        """
        Return a FloIterator for a while–for hybrid loop over memory steps.
        """
        return FloIterator(self, depth)


class FloIterator:
    """
    Hybrid while–for iterator over FloJsonOutputCollector data.
    Newest entries first, depth-limited.
    """

    def __init__(self, collector: FloJsonOutputCollector, depth: Optional[int] = None):
        self.entries = collector.data[::-1]
        self.limit = min(depth, len(self.entries)) if depth is not None else len(self.entries)
        self.index = 0

    def has_next(self) -> bool:
        return self.index < self.limit

    def next(self) -> List[Dict[str, Any]]:
        if not self.has_next():
            return []
        entry = self.entries[self.index]
        self.index += 1
        return [entry]
