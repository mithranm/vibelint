"""
Runtime Module Tracer for Vibelint Self-Improvement

Runs Python modules with mock values and traces all method calls with full stack traces.
Perfect for understanding execution flow and identifying optimization opportunities.
"""

import ast
import importlib
import importlib.util
import inspect
import json
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class CallInfo:
    """Information about a function/method call."""

    function_name: str
    module_name: str
    file_path: str
    line_number: int
    call_time: float
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    locals_snapshot: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    execution_time_ms: float = 0.0
    call_depth: int = 0
    caller_info: Optional[str] = None


@dataclass
class DependencyNode:
    """A node in the dependency knowledge graph."""

    function_signature: str
    module_path: str
    dependencies: List[str] = field(default_factory=list)
    code_embedding: Optional[List[float]] = None
    semantic_embedding: Optional[List[float]] = None
    execution_context: Dict[str, Any] = field(default_factory=dict)
    performance_profile: Dict[str, float] = field(default_factory=dict)


@dataclass
class TraceSession:
    """Complete tracing session data with embedding integration."""

    module_name: str
    start_time: float
    end_time: float
    total_calls: int
    call_stack: List[CallInfo] = field(default_factory=list)
    performance_hotspots: List[CallInfo] = field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    dependency_knowledge_graph: Dict[str, DependencyNode] = field(default_factory=dict)
    execution_embeddings: Dict[str, List[float]] = field(default_factory=dict)


class MockProvider:
    """Provides mock values for different types and situations."""

    @staticmethod
    def create_mock_for_type(type_hint: Any, context: str = "") -> Any:
        """Create appropriate mock value based on type hint."""
        if type_hint == str:
            return f"mock_string_{context}"
        elif type_hint == int:
            return 42
        elif type_hint == float:
            return 3.14
        elif type_hint == bool:
            return True
        elif type_hint == list:
            return ["mock_item1", "mock_item2"]
        elif type_hint == dict:
            return {"mock_key": "mock_value"}
        elif type_hint == Path:
            return Path("/mock/path/file.py")
        elif hasattr(type_hint, "__origin__"):
            # Handle generic types like List[str], Dict[str, int]
            origin = type_hint.__origin__
            if origin == list:
                return ["mock_list_item"]
            elif origin == dict:
                return {"mock_dict_key": "mock_dict_value"}

        # Default mock object
        return MockProvider._create_mock_object(type_hint, context)

    @staticmethod
    def _create_mock_object(cls: type, context: str = ""):
        """Create a mock object with realistic methods."""

        class MockObject:
            def __init__(self):
                self._mock_context = context
                self._mock_class = cls.__name__ if hasattr(cls, "__name__") else str(cls)

            def __getattr__(self, name):
                return lambda *args, **kwargs: f"mock_result_from_{name}"

            def __str__(self):
                return f"Mock{self._mock_class}({self._mock_context})"

            def __repr__(self):
                return self.__str__()

        return MockObject()

    @staticmethod
    def mock_common_dependencies() -> Dict[str, Any]:
        """Create mocks for common dependencies."""
        return {
            "os": MockProvider._create_mock_module("os"),
            "sys": MockProvider._create_mock_module("sys"),
            "pathlib.Path": Path("/mock/path"),
            "requests": MockProvider._create_mock_module("requests"),
            "json": MockProvider._create_mock_module("json"),
            "time": MockProvider._create_mock_module("time"),
            "datetime": MockProvider._create_mock_module("datetime"),
            "asyncio": MockProvider._create_mock_module("asyncio"),
        }

    @staticmethod
    def _create_mock_module(module_name: str):
        """Create a mock module with common methods."""

        class MockModule:
            def __getattr__(self, name):
                if name in ["get", "post", "put", "delete"]:  # requests
                    return lambda *args, **kwargs: MockProvider._mock_response()
                elif name in ["loads", "dumps"]:  # json
                    return lambda *args, **kwargs: {"mock": "json_data"}
                elif name in ["sleep"]:  # time/asyncio
                    return lambda *args, **kwargs: None
                elif name in ["now"]:  # datetime
                    return lambda *args, **kwargs: "2024-01-01T00:00:00"
                else:
                    return lambda *args, **kwargs: f"mock_{name}_result"

        return MockModule()

    @staticmethod
    def _mock_response():
        """Mock HTTP response object."""

        class MockResponse:
            status_code = 200

            def json(self):
                return {"mock": "response"}

            def text(self):
                return "mock response text"

        return MockResponse()


class VanguardEmbeddingIntegration:
    """
    Integrates with VanguardOne (code) and VanguardTwo (semantic) embeddings
    to create a rich dependency knowledge graph.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vanguard_one_url = self.config.get("code_api_url")
        self.vanguard_two_url = self.config.get("natural_api_url")

    async def create_dependency_embeddings(
        self, call_stack: List[CallInfo], source_code_cache: Dict[str, str]
    ) -> Dict[str, DependencyNode]:
        """
        Create embedding-enhanced dependency nodes from call stack.
        """

        dependency_nodes = {}

        for call in call_stack:
            node_id = f"{call.module_name}.{call.function_name}"

            if node_id in dependency_nodes:
                continue

            # Extract function source code
            function_source = await self._extract_function_source(call, source_code_cache)

            # Get function signature and docstring for semantic embedding
            semantic_text = await self._create_semantic_description(call, function_source)

            # Create embeddings
            code_embedding = await self._get_code_embedding(function_source)
            semantic_embedding = await self._get_semantic_embedding(semantic_text)

            # Build dependency node
            dependencies = self._extract_dependencies_from_call(call, call_stack)

            node = DependencyNode(
                function_signature=f"{call.function_name}({', '.join(map(str, call.args))})",
                module_path=call.file_path,
                dependencies=dependencies,
                code_embedding=code_embedding,
                semantic_embedding=semantic_embedding,
                execution_context={
                    "call_frequency": self._calculate_call_frequency(node_id, call_stack),
                    "average_execution_time": call.execution_time_ms,
                    "typical_args": call.args,
                    "error_patterns": [],  # Would track exceptions
                },
                performance_profile={
                    "avg_time_ms": call.execution_time_ms,
                    "memory_impact": "unknown",  # Could be measured
                    "io_operations": self._detect_io_operations(function_source),
                },
            )

            dependency_nodes[node_id] = node

        return dependency_nodes

    async def _extract_function_source(self, call: CallInfo, source_cache: Dict[str, str]) -> str:
        """Extract the source code of a specific function."""
        try:
            if call.file_path not in source_cache:
                with open(call.file_path, "r") as f:
                    source_cache[call.file_path] = f.read()

            source = source_cache[call.file_path]

            # Parse AST to find the function
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == call.function_name
                    and node.lineno <= call.line_number
                ):

                    # Extract function source
                    lines = source.split("\n")
                    start_line = node.lineno - 1

                    # Find end of function (next function or class at same indent level)
                    end_line = len(lines)
                    base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())

                    for i in range(start_line + 1, len(lines)):
                        line = lines[i]
                        if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                            if any(
                                line.strip().startswith(kw)
                                for kw in ["def ", "class ", "async def "]
                            ):
                                end_line = i
                                break

                    return "\n".join(lines[start_line:end_line])

            return f"# Function {call.function_name} not found in source"

        except Exception as e:
            return f"# Error extracting source: {e}"

    async def _create_semantic_description(self, call: CallInfo, function_source: str) -> str:
        """Create semantic description for natural language embedding."""
        # Extract docstring and comments
        docstring = ""
        comments = []

        try:
            tree = ast.parse(function_source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if ast.get_docstring(node):
                        docstring = ast.get_docstring(node)
                    break
        except:
            pass

        # Create rich semantic description
        description = f"""
Function: {call.function_name}
Module: {call.module_name}
Purpose: {docstring or "No docstring available"}
Typical arguments: {call.args}
Return type: {type(call.return_value).__name__ if call.return_value else "unknown"}
Execution context: Called with depth {call.call_depth}
Performance: Executes in {call.execution_time_ms:.2f}ms on average
Dependencies: Interacts with other functions in execution flow
Usage pattern: Runtime analysis shows this function is used for {self._infer_usage_pattern(call)}
        """.strip()

        return description

    def _infer_usage_pattern(self, call: CallInfo) -> str:
        """Infer usage pattern from call context."""
        if "config" in call.function_name.lower():
            return "configuration management"
        elif "validate" in call.function_name.lower():
            return "code validation and analysis"
        elif "analyze" in call.function_name.lower():
            return "static or dynamic code analysis"
        elif "trace" in call.function_name.lower():
            return "execution tracing and monitoring"
        elif any(kw in call.function_name.lower() for kw in ["load", "read", "parse"]):
            return "data loading and parsing"
        elif any(kw in call.function_name.lower() for kw in ["save", "write", "store"]):
            return "data persistence and storage"
        else:
            return "general purpose computation"

    async def _get_code_embedding(self, source_code: str) -> Optional[List[float]]:
        """Get code embedding from VanguardOne."""
        if not self.vanguard_one_url:
            return None

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.vanguard_one_url,
                    json={
                        "model": "text-embedding-ada-002",  # API expects this
                        "input": source_code[:8000],  # Limit input size
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("data", [{}])[0].get("embedding")

        except Exception as e:
            print(f"Failed to get code embedding: {e}")

        return None

    async def _get_semantic_embedding(self, semantic_text: str) -> Optional[List[float]]:
        """Get semantic embedding from VanguardTwo."""
        if not self.vanguard_two_url:
            return None

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.vanguard_two_url,
                    json={
                        "model": "text-embedding-ada-002",  # API expects this
                        "input": semantic_text[:8000],  # Limit input size
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("data", [{}])[0].get("embedding")

        except Exception as e:
            print(f"Failed to get semantic embedding: {e}")

        return None

    def _extract_dependencies_from_call(
        self, call: CallInfo, all_calls: List[CallInfo]
    ) -> List[str]:
        """Extract what this function depends on based on call stack."""
        dependencies = []

        # Find calls that happen within this function's execution
        call_start_time = call.call_time
        call_end_time = call_start_time + (call.execution_time_ms / 1000)

        for other_call in all_calls:
            if (
                other_call != call
                and call_start_time <= other_call.call_time <= call_end_time
                and other_call.call_depth > call.call_depth
            ):

                dependency = f"{other_call.module_name}.{other_call.function_name}"
                if dependency not in dependencies:
                    dependencies.append(dependency)

        return dependencies

    def _calculate_call_frequency(self, node_id: str, call_stack: List[CallInfo]) -> int:
        """Calculate how often this function is called."""
        return sum(
            1 for call in call_stack if f"{call.module_name}.{call.function_name}" == node_id
        )

    def _detect_io_operations(self, source_code: str) -> int:
        """Detect I/O operations in source code."""
        io_indicators = [
            "open(",
            "read(",
            "write(",
            "requests.",
            "http",
            "json.load",
            "json.dump",
            "pickle.",
            "os.path",
            "pathlib",
            "sqlite",
            "database",
        ]

        return sum(1 for indicator in io_indicators if indicator in source_code)


class RuntimeTracer:
    """
    Advanced runtime tracer that can run modules with mocks and trace execution.
    """

    def __init__(
        self,
        include_modules: Optional[Set[str]] = None,
        exclude_modules: Optional[Set[str]] = None,
        trace_external: bool = False,
        max_call_depth: int = 50,
    ):
        self.include_modules = include_modules or set()
        self.exclude_modules = exclude_modules or {
            "sys",
            "os",
            "traceback",
            "inspect",
            "importlib",
            "__main__",
            "typing",
            "dataclasses",
        }
        self.trace_external = trace_external
        self.max_call_depth = max_call_depth

        self.call_stack: List[CallInfo] = []
        self.current_depth = 0
        self.session_start = 0.0
        self.call_times = {}

    def should_trace_frame(self, frame) -> bool:
        """Determine if we should trace this frame."""
        filename = frame.f_code.co_filename
        module_name = frame.f_globals.get("__name__", "")

        # Skip if too deep
        if self.current_depth > self.max_call_depth:
            return False

        # Skip built-in modules
        if filename.startswith("<") or "site-packages" in filename:
            return not self.trace_external

        # Include specific modules
        if self.include_modules:
            return any(
                inc_mod in module_name or inc_mod in filename for inc_mod in self.include_modules
            )

        # Exclude specific modules
        return not any(
            exc_mod in module_name or exc_mod in filename for exc_mod in self.exclude_modules
        )

    def trace_function(self, frame, event, arg):
        """Main tracing function."""
        if not self.should_trace_frame(frame):
            return self.trace_function

        filename = frame.f_code.co_filename
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        module_name = frame.f_globals.get("__name__", "")

        if event == "call":
            self.current_depth += 1

            # Get function arguments
            args, kwargs = self._extract_call_args(frame)

            # Get caller information
            caller_info = self._get_caller_info()

            call_info = CallInfo(
                function_name=function_name,
                module_name=module_name,
                file_path=filename,
                line_number=line_number,
                call_time=time.time(),
                args=args,
                kwargs=kwargs,
                locals_snapshot=dict(frame.f_locals),
                call_depth=self.current_depth,
                caller_info=caller_info,
            )

            self.call_stack.append(call_info)

        elif event == "return":
            self.current_depth = max(0, self.current_depth - 1)

            # Update the most recent call with return value and timing
            if self.call_stack:
                recent_call = None
                for call in reversed(self.call_stack):
                    if (
                        call.function_name == function_name
                        and call.file_path == filename
                        and call.return_value is None
                    ):
                        recent_call = call
                        break

                if recent_call:
                    recent_call.return_value = arg
                    recent_call.execution_time_ms = (time.time() - recent_call.call_time) * 1000

        return self.trace_function

    def _extract_call_args(self, frame) -> tuple[List[Any], Dict[str, Any]]:
        """Extract function arguments from frame."""
        try:
            arginfo = inspect.getargvalues(frame)
            args = []
            kwargs = {}

            for arg_name in arginfo.args:
                value = arginfo.locals.get(arg_name)
                args.append(self._sanitize_value(value))

            # Handle keyword arguments
            if arginfo.keywords:
                keyword_args = arginfo.locals.get(arginfo.keywords, {})
                kwargs.update({k: self._sanitize_value(v) for k, v in keyword_args.items()})

            return args, kwargs

        except Exception:
            return [], {}

    def _get_caller_info(self) -> str:
        """Get information about who called this function."""
        try:
            stack = inspect.stack()
            # Skip trace frames
            for frame_info in stack[3:]:  # Skip trace_function, _extract_call_args, etc.
                if not any(skip in frame_info.filename for skip in ["trace", "inspect"]):
                    return f"{frame_info.filename}:{frame_info.lineno} in {frame_info.function}"
            return "unknown_caller"
        except:
            return "unknown_caller"

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize value for JSON serialization."""
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            elif isinstance(value, (list, tuple)):
                return [self._sanitize_value(v) for v in value[:3]]  # Limit to first 3 items
            elif isinstance(value, dict):
                return {k: self._sanitize_value(v) for k, v in list(value.items())[:3]}
            elif hasattr(value, "__dict__"):
                return f"<{type(value).__name__} object>"
            else:
                return str(value)[:100]  # Truncate long strings
        except:
            return "<unserializable>"

    @contextmanager
    def trace_execution(self):
        """Context manager for tracing execution."""
        self.session_start = time.time()
        old_trace = sys.gettrace()

        try:
            sys.settrace(self.trace_function)
            yield self
        finally:
            sys.settrace(old_trace)

    def run_module_with_mocks(
        self,
        module_path: Path,
        function_name: str = None,
        mock_args: List[Any] = None,
        mock_kwargs: Dict[str, Any] = None,
    ) -> TraceSession:
        """
        Run a module or specific function with mocks and trace execution.
        """
        print(f"🔍 Tracing execution of {module_path}")

        # Prepare mocks
        mocks = MockProvider.mock_common_dependencies()

        # Add module to include set
        self.include_modules.add(module_path.stem)

        session = TraceSession(
            module_name=str(module_path), start_time=time.time(), end_time=0.0, total_calls=0
        )

        try:
            with self.trace_execution():
                # Import the module
                spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
                module = importlib.util.module_from_spec(spec)

                # Inject mocks into module namespace
                for mock_name, mock_obj in mocks.items():
                    if "." not in mock_name:
                        setattr(module, mock_name, mock_obj)

                # Execute the module
                spec.loader.exec_module(module)

                # Run specific function if specified
                if function_name:
                    if hasattr(module, function_name):
                        func = getattr(module, function_name)

                        # Prepare arguments
                        if mock_args is None:
                            mock_args = self._generate_mock_args_for_function(func)
                        if mock_kwargs is None:
                            mock_kwargs = {}

                        print(f"📞 Calling {function_name}({mock_args}, {mock_kwargs})")
                        result = func(*mock_args, **mock_kwargs)
                        print(f"📊 Function returned: {result}")

        except Exception as e:
            print(f"❌ Execution failed: {e}")
            traceback.print_exc()

        session.end_time = time.time()
        session.total_calls = len(self.call_stack)
        session.call_stack = self.call_stack.copy()

        # Analyze performance hotspots
        session.performance_hotspots = sorted(
            [call for call in self.call_stack if call.execution_time_ms > 1.0],
            key=lambda x: x.execution_time_ms,
            reverse=True,
        )[:10]

        # Build dependency graph
        session.dependency_graph = self._build_dependency_graph()

        return session

    def _generate_mock_args_for_function(self, func: Callable) -> List[Any]:
        """Generate mock arguments for a function based on its signature."""
        try:
            sig = inspect.signature(func)
            mock_args = []

            for param_name, param in sig.parameters.items():
                if param.kind == param.VAR_POSITIONAL:
                    break  # Don't mock *args
                elif param.kind == param.VAR_KEYWORD:
                    break  # Don't mock **kwargs

                # Create mock based on annotation or default
                if param.annotation != param.empty:
                    mock_value = MockProvider.create_mock_for_type(param.annotation, param_name)
                elif param.default != param.empty:
                    mock_value = param.default
                else:
                    mock_value = f"mock_{param_name}"

                mock_args.append(mock_value)

            return mock_args

        except Exception:
            return ["mock_arg"]

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build a dependency graph from call stack."""
        graph = {}

        for call in self.call_stack:
            caller = call.caller_info or "root"
            callee = f"{call.module_name}.{call.function_name}"

            if caller not in graph:
                graph[caller] = []

            if callee not in graph[caller]:
                graph[caller].append(callee)

        return graph

    def export_trace_report(self, session: TraceSession, output_path: Path):
        """Export detailed trace report."""
        report = {
            "session_info": {
                "module": session.module_name,
                "duration_ms": (session.end_time - session.start_time) * 1000,
                "total_calls": session.total_calls,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "call_stack": [
                {
                    "function": f"{call.module_name}.{call.function_name}",
                    "file": call.file_path,
                    "line": call.line_number,
                    "execution_time_ms": call.execution_time_ms,
                    "call_depth": call.call_depth,
                    "args": call.args,
                    "kwargs": call.kwargs,
                    "return_value": call.return_value,
                    "caller": call.caller_info,
                }
                for call in session.call_stack
            ],
            "performance_hotspots": [
                {
                    "function": f"{call.module_name}.{call.function_name}",
                    "execution_time_ms": call.execution_time_ms,
                    "file": call.file_path,
                    "line": call.line_number,
                }
                for call in session.performance_hotspots
            ],
            "dependency_graph": session.dependency_graph,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Trace report exported to: {output_path}")


# Convenience functions


def trace_vibelint_module(module_name: str, function_name: str = None) -> TraceSession:
    """Trace a vibelint module with automatic setup."""
    vibelint_src = Path(__file__).parent
    module_path = vibelint_src / f"{module_name}.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    tracer = RuntimeTracer(include_modules={"vibelint"}, trace_external=False, max_call_depth=20)

    return tracer.run_module_with_mocks(module_path, function_name)


def analyze_self_improvement_execution():
    """Trace the self-improvement system to understand its execution flow."""
    print("🧠 Analyzing vibelint self-improvement execution...")

    session = trace_vibelint_module("self_improvement", "run_vibelint_self_improvement")

    # Export trace report
    report_path = (
        Path(__file__).parent.parent.parent / ".vibelint-self-improvement" / "execution_trace.json"
    )
    report_path.parent.mkdir(exist_ok=True)

    tracer = RuntimeTracer()
    tracer.export_trace_report(session, report_path)

    print("🎯 Performance hotspots found:")
    for hotspot in session.performance_hotspots[:5]:
        print(f"  - {hotspot.function_name}: {hotspot.execution_time_ms:.2f}ms")

    return session


if __name__ == "__main__":
    # Example usage
    analyze_self_improvement_execution()
