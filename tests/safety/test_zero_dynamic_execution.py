"""
tests.safety.test_zero_dynamic_execution
========================================

Static AST-based safety verification ensuring zero dynamic model execution,
deserialization, class instantiation, or forbidden runtime imports in the
production source tree.

Phase 12 Safety Hardening.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Prohibited function call targets: (module_name, func_name) or builtins
PROHIBITED_CALLS = {
    ("pickle", "load"),
    ("pickle", "loads"),
    ("_pickle", "load"),
    ("_pickle", "loads"),
    ("torch", "load"),
    ("torch.jit", "load"),
    ("joblib", "load"),
    ("joblib", "loads"),
    ("cloudpickle", "load"),
    ("cloudpickle", "loads"),
    ("dill", "load"),
    ("dill", "loads"),
}

PROHIBITED_BUILTINS = {"eval", "exec"}

PROHIBITED_IMPORTS = {
    "torch",
    "transformers",
    "llama_cpp",
    "joblib",
    "cloudpickle",
    "dill",
}


class SafetyASTVisitor(ast.NodeVisitor):
    """AST visitor collecting dangerous invocation nodes."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.violations: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct builtin calls (eval, exec)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in PROHIBITED_BUILTINS:
                self.violations.append(
                    f"{self.filepath}:{node.lineno} prohibited builtin call {func_name}()"
                )

        # Check attribute calls (e.g. pickle.load, torch.load)
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            # Case 1: module.function (e.g. pickle.loads)
            if isinstance(node.func.value, ast.Name):
                mod_name = node.func.value.id
                if (mod_name, attr_name) in PROHIBITED_CALLS:
                    self.violations.append(
                        f"{self.filepath}:{node.lineno} prohibited call {mod_name}.{attr_name}()"
                    )
            # Case 2: module.submodule.function (e.g. torch.jit.load)
            elif isinstance(node.func.value, ast.Attribute):
                sub_attr = node.func.value.attr
                if isinstance(node.func.value.value, ast.Name):
                    root_mod = node.func.value.value.id
                    full_mod = f"{root_mod}.{sub_attr}"
                    if (full_mod, attr_name) in PROHIBITED_CALLS:
                        self.violations.append(
                            f"{self.filepath}:{node.lineno} prohibited call "
                            f"{full_mod}.{attr_name}()"
                        )

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_pkg = alias.name.split(".")[0]
            if root_pkg in PROHIBITED_IMPORTS:
                self.violations.append(
                    f"{self.filepath}:{node.lineno} prohibited import '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root_pkg = node.module.split(".")[0]
            if root_pkg in PROHIBITED_IMPORTS:
                self.violations.append(
                    f"{self.filepath}:{node.lineno} prohibited import from '{node.module}'"
                )
        self.generic_visit(node)


def get_src_python_files() -> list[Path]:
    """Find all python files under src/model_scanner/."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "model_scanner"
    return sorted(src_dir.rglob("*.py"))


class TestZeroDynamicExecution:
    """Verify zero dynamic execution primitives across production codebase."""

    def test_no_prohibited_calls_or_imports(self) -> None:
        """Scan all source files with AST visitor to assert 0 safety violations."""
        py_files = get_src_python_files()
        assert len(py_files) >= 20, f"Expected >20 source files, found {len(py_files)}"

        all_violations: list[str] = []
        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
            visitor = SafetyASTVisitor(py_file)
            visitor.visit(tree)
            all_violations.extend(visitor.violations)

        assert not all_violations, (
            f"Found {len(all_violations)} safety violations in production code:\n"
            + "\n".join(all_violations)
        )
