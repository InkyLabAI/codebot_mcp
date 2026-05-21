from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Parameter:
    name: str
    type: Optional[str] = None
    value: Optional[str] = None
    
    def to_dict(self):
        return {
            'name': self.name,
            'type': self.type,
            'value': self.value
        }

@dataclass
class ClassChunk:
    id: str
    name: str
    code: str
    docstring: Optional[str]
    start_line: int
    end_line: int
    decorators: List[str]
    superclasses: List[str]
    file_path: str
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'docstring': self.docstring,
            'decorators': self.decorators,
            'superclasses': self.superclasses,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'code': self.code,
        }
    
@dataclass
class FunctionChunk:
    id: str
    name: str
    code: str
    docstring: Optional[str]
    start_line: int
    end_line: int
    parameters: List[Parameter]
    decorators: List[str]
    file_path: str
    class_name: Optional[str] = None
    nested: Optional[str] = None
    return_type: Optional[str] = None
    calls: List[str] = None
    var_types: Dict[str, List[str]] = None  # Maps variable names to possible class names (supports conditional assignments)
    var_sources: Dict[str, str] = None  # Maps var names to source expressions for return-type resolution

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'class_name': self.class_name,
            'nested': self.nested,
            'parameters': [param.to_dict() for param in self.parameters],
            'docstring': self.docstring,
            'decorators': self.decorators,
            'return_type': self.return_type,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'code': self.code,
            'calls': self.calls,
            'var_types': self.var_types,
            'var_sources': self.var_sources
        }
    