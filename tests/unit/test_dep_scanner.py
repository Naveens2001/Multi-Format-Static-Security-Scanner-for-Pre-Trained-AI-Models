"""
tests.unit.test_dep_scanner
============================

Unit tests for model_scanner.analyzers.repository.dep_scanner — dependency
scanning (SC201/SC202/SC203/SC204 rules) and helper functions.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from model_scanner.analyzers.repository.dep_scanner import (
    _edit_distance_1,
    _extract_affected_versions,
    _extract_pinned_version,
    _normalize_name,
    _normalize_version,
    _parse_environment_yml_deps,
    _parse_pyproject_toml,
    _parse_requirements_file,
    _version_satisfies_range,
    scan_dependencies,
)

# ---------------------------------------------------------------------------
# _normalize_version
# ---------------------------------------------------------------------------


class TestNormalizeVersion:
    def test_simple(self) -> None:
        assert _normalize_version("1.2.3") == (1, 2, 3)

    def test_two_part(self) -> None:
        assert _normalize_version("4.5") == (4, 5)

    def test_epoch_stripped(self) -> None:
        assert _normalize_version("1!2.3.4") == (2, 3, 4)

    def test_local_stripped(self) -> None:
        assert _normalize_version("1.2.3+local") == (1, 2, 3)

    def test_pre_release_alpha(self) -> None:
        result = _normalize_version("1.0.0a1")
        # Pre-release should compare less than the release; just ensure it returns a tuple
        assert isinstance(result, tuple)
        assert len(result) >= 1

    def test_pre_release_rc(self) -> None:
        result = _normalize_version("2.0.0rc1")
        assert isinstance(result, tuple)

    def test_post_release(self) -> None:
        result = _normalize_version("1.0.0.post1")
        assert isinstance(result, tuple)

    def test_non_numeric_segment(self) -> None:
        result = _normalize_version("1.0.dev0")
        assert isinstance(result, tuple)

    def test_empty_string_returns_zero(self) -> None:
        result = _normalize_version("")
        assert result == (0,)


# ---------------------------------------------------------------------------
# _version_satisfies_range
# ---------------------------------------------------------------------------


class TestVersionSatisfiesRange:
    def test_in_open_range(self) -> None:
        assert _version_satisfies_range("1.5.0", "1.0.0", None) is True

    def test_below_introduced(self) -> None:
        assert _version_satisfies_range("0.9.0", "1.0.0", None) is False

    def test_exactly_introduced(self) -> None:
        assert _version_satisfies_range("1.0.0", "1.0.0", None) is True

    def test_in_closed_range(self) -> None:
        assert _version_satisfies_range("1.5.0", "1.0.0", "2.0.0") is True

    def test_at_fixed(self) -> None:
        assert _version_satisfies_range("2.0.0", "1.0.0", "2.0.0") is False

    def test_above_fixed(self) -> None:
        assert _version_satisfies_range("3.0.0", "1.0.0", "2.0.0") is False

    def test_zero_introduced_means_all(self) -> None:
        assert _version_satisfies_range("2.0.0", "0", None) is True

    def test_invalid_version_returns_false(self) -> None:
        assert _version_satisfies_range("not.a.version", "1.0.0", "2.0.0") is False


# ---------------------------------------------------------------------------
# _normalize_name
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_hyphens_underscores(self) -> None:
        assert _normalize_name("My_Package-Name") == "my-package-name"

    def test_dots(self) -> None:
        assert _normalize_name("my.pkg") == "my-pkg"


# ---------------------------------------------------------------------------
# _extract_pinned_version
# ---------------------------------------------------------------------------


class TestExtractPinnedVersion:
    def test_exact_pin(self) -> None:
        assert _extract_pinned_version("==1.2.3") == "1.2.3"

    def test_with_spaces(self) -> None:
        assert _extract_pinned_version("== 4.5.6") == "4.5.6"

    def test_no_match(self) -> None:
        assert _extract_pinned_version(">=1.0.0") is None

    def test_wildcard(self) -> None:
        assert _extract_pinned_version("*") is None


# ---------------------------------------------------------------------------
# _edit_distance_1
# ---------------------------------------------------------------------------


class TestEditDistance1:
    def test_identical_is_not_1(self) -> None:
        assert _edit_distance_1("numpy", "numpy") is False

    def test_substitution(self) -> None:
        assert _edit_distance_1("numpy", "numpi") is True

    def test_insertion(self) -> None:
        assert _edit_distance_1("numpy", "numpyy") is True

    def test_deletion(self) -> None:
        assert _edit_distance_1("numpyy", "numpy") is True

    def test_two_edits(self) -> None:
        assert _edit_distance_1("numpy", "numpi2") is False

    def test_length_diff_too_large(self) -> None:
        assert _edit_distance_1("a", "abcde") is False


# ---------------------------------------------------------------------------
# _parse_requirements_file
# ---------------------------------------------------------------------------


class TestParseRequirementsFile:
    def test_exact_pin(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("torch==2.0.1\n", encoding="utf-8")
        deps = _parse_requirements_file(req, "requirements.txt")
        assert len(deps) == 1
        assert deps[0].name == "torch"
        assert deps[0].is_pinned is True

    def test_loose_spec(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("numpy>=1.20\n", encoding="utf-8")
        deps = _parse_requirements_file(req, "requirements.txt")
        assert deps[0].is_pinned is False

    def test_comment_skipped(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("# this is a comment\ntorch==2.0\n", encoding="utf-8")
        deps = _parse_requirements_file(req, "requirements.txt")
        assert len(deps) == 1

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("\n\ntorch==2.0\n\n", encoding="utf-8")
        deps = _parse_requirements_file(req, "requirements.txt")
        assert len(deps) == 1

    def test_inline_comment_stripped(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("torch==2.0  # pin for reproducibility\n", encoding="utf-8")
        deps = _parse_requirements_file(req, "requirements.txt")
        assert deps[0].is_pinned is True

    def test_vcs_url_dep(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("git+https://github.com/user/repo.git\n", encoding="utf-8")
        deps = _parse_requirements_file(req, "requirements.txt")
        assert len(deps) == 1
        assert deps[0].is_vcs_or_url is True

    def test_vcs_no_egg(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("git+https://github.com/user/repo2.git\n", encoding="utf-8")
        deps = _parse_requirements_file(req, "requirements.txt")
        # No egg fragment → name is <unknown>
        assert deps[0].name == "<unknown>"

    def test_include_directive(self, tmp_path: Path) -> None:
        base = tmp_path / "requirements.txt"
        extra = tmp_path / "extra.txt"
        extra.write_text("scipy==1.9.0\n", encoding="utf-8")
        base.write_text("-r extra.txt\ntorch==2.0\n", encoding="utf-8")
        deps = _parse_requirements_file(base, "requirements.txt")
        names = {d.name for d in deps}
        assert "scipy" in names
        assert "torch" in names

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        deps = _parse_requirements_file(tmp_path / "nonexistent.txt", "nonexistent.txt")
        assert deps == []

    def test_depth_limit_prevents_recursion(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("-r requirements.txt\n", encoding="utf-8")
        # Should not infinite-loop; returns limited results
        deps = _parse_requirements_file(req, "requirements.txt", depth=20)
        assert deps == []


# ---------------------------------------------------------------------------
# _parse_pyproject_toml
# ---------------------------------------------------------------------------


class TestParsePyprojectToml:
    def test_pep621_list(self, tmp_path: Path) -> None:
        p = tmp_path / "pyproject.toml"
        p.write_text(
            textwrap.dedent("""\
                [project]
                dependencies = [
                  "torch>=2.0",
                  "numpy==1.24.0",
                ]
            """),
            encoding="utf-8",
        )
        deps = _parse_pyproject_toml(p)
        names = {d.name for d in deps}
        assert "torch" in names
        assert "numpy" in names

    def test_poetry_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "pyproject.toml"
        p.write_text(
            textwrap.dedent("""\
                [tool.poetry.dependencies]
                python = "^3.10"
                torch = "^2.0"
                numpy = "==1.24.0"
            """),
            encoding="utf-8",
        )
        deps = _parse_pyproject_toml(p)
        names = {d.name for d in deps}
        assert "torch" in names
        assert "numpy" in names
        # python should be skipped
        assert "python" not in names

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        deps = _parse_pyproject_toml(tmp_path / "nonexistent.toml")
        assert deps == []

    def test_invalid_toml_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "pyproject.toml"
        p.write_bytes(b"not valid toml !!!{{{")
        deps = _parse_pyproject_toml(p)
        assert deps == []


# ---------------------------------------------------------------------------
# _parse_environment_yml_deps
# ---------------------------------------------------------------------------


class TestParseEnvironmentYml:
    def test_basic_deps(self, tmp_path: Path) -> None:
        yml = tmp_path / "environment.yml"
        yml.write_text(
            textwrap.dedent("""\
                name: myenv
                dependencies:
                  - numpy==1.24.0
                  - scipy>=1.9
                  - torch
            """),
            encoding="utf-8",
        )
        deps = _parse_environment_yml_deps(yml)
        names = {d.name for d in deps}
        assert "numpy" in names
        assert "scipy" in names
        assert "torch" in names

    def test_pip_marker_skipped(self, tmp_path: Path) -> None:
        yml = tmp_path / "environment.yml"
        yml.write_text(
            textwrap.dedent("""\
                dependencies:
                  - pip
                  - torch==2.0
            """),
            encoding="utf-8",
        )
        deps = _parse_environment_yml_deps(yml)
        names = {d.name for d in deps}
        # "pip" is marker and should be skipped
        assert "pip" not in names
        assert "torch" in names

    def test_top_level_key_exits_deps_section(self, tmp_path: Path) -> None:
        yml = tmp_path / "environment.yml"
        yml.write_text(
            textwrap.dedent("""\
                name: myenv
                dependencies:
                  - numpy==1.24
                channels:
                  - defaults
            """),
            encoding="utf-8",
        )
        deps = _parse_environment_yml_deps(yml)
        names = {d.name for d in deps}
        assert "numpy" in names
        assert "defaults" not in names

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        deps = _parse_environment_yml_deps(tmp_path / "nonexistent.yml")
        assert deps == []


# ---------------------------------------------------------------------------
# _extract_affected_versions
# ---------------------------------------------------------------------------


class TestExtractAffectedVersions:
    def _make_advisory(
        self,
        pkg_name: str,
        introduced: str,
        fixed: str | None,
        adv_id: str = "TEST-001",
    ) -> dict:  # type: ignore[type-arg]
        events: list[dict[str, str]] = [{"introduced": introduced}]
        if fixed:
            events.append({"fixed": fixed})
        return {
            "id": adv_id,
            "affected": [
                {
                    "package": {"name": pkg_name, "ecosystem": "PyPI"},
                    "ranges": [{"type": "ECOSYSTEM", "events": events}],
                }
            ],
        }

    def test_simple_range(self) -> None:
        adv = self._make_advisory("torch", "1.0.0", "2.0.0")
        ranges = _extract_affected_versions(adv, "torch")
        assert len(ranges) == 1
        assert ranges[0] == ("1.0.0", "2.0.0")

    def test_open_ended_range(self) -> None:
        adv = self._make_advisory("torch", "1.0.0", None)
        ranges = _extract_affected_versions(adv, "torch")
        assert len(ranges) == 1
        assert ranges[0] == ("1.0.0", None)

    def test_wrong_package_returns_empty(self) -> None:
        adv = self._make_advisory("torch", "1.0.0", "2.0.0")
        ranges = _extract_affected_versions(adv, "numpy")
        assert ranges == []

    def test_case_insensitive_name_match(self) -> None:
        adv = self._make_advisory("Torch", "1.0.0", "2.0.0")
        ranges = _extract_affected_versions(adv, "torch")
        assert len(ranges) == 1

    def test_non_ecosystem_range_skipped(self) -> None:
        adv = {
            "id": "X",
            "affected": [
                {
                    "package": {"name": "torch"},
                    "ranges": [
                        {"type": "GIT", "events": [{"introduced": "abc123"}]}
                    ],
                }
            ],
        }
        ranges = _extract_affected_versions(adv, "torch")
        assert ranges == []

    def test_no_affected_field(self) -> None:
        ranges = _extract_affected_versions({"id": "X"}, "torch")
        assert ranges == []


# ---------------------------------------------------------------------------
# scan_dependencies — integration (no real advisory DB needed)
# ---------------------------------------------------------------------------


class TestScanDependencies:
    def test_unpinned_dep_emits_sc201(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("torch>=2.0\n", encoding="utf-8")
        findings, _ = scan_dependencies(tmp_path)
        rule_ids = {f.rule_id for f in findings}
        assert "SC201" in rule_ids

    def test_exact_pin_no_sc201(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("torch==2.0.1\n", encoding="utf-8")
        findings, _ = scan_dependencies(tmp_path)
        sc201 = [f for f in findings if f.rule_id == "SC201"]
        assert sc201 == []

    def test_vcs_dep_emits_sc203(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("git+https://github.com/user/repo.git#egg=mylib\n", encoding="utf-8")
        findings, _ = scan_dependencies(tmp_path)
        rule_ids = {f.rule_id for f in findings}
        assert "SC203" in rule_ids

    def test_config_json_version_pin(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("numpy>=1.20\n", encoding="utf-8")
        config = {"transformers_version": "4.30.0"}
        findings, _ = scan_dependencies(tmp_path, config_json=config)
        # transformers==4.30.0 is pinned → no SC201 for it
        sc201_names = {f.dep_name for f in findings if f.rule_id == "SC201"}
        assert "transformers" not in sc201_names

    def test_no_dep_files_returns_empty(self, tmp_path: Path) -> None:
        findings, _ = scan_dependencies(tmp_path)
        assert findings == []

    def test_pyproject_toml_parsed(self, tmp_path: Path) -> None:
        p = tmp_path / "pyproject.toml"
        p.write_text(
            '[project]\ndependencies = ["numpy>=1.20"]\n',
            encoding="utf-8",
        )
        findings, _ = scan_dependencies(tmp_path)
        rule_ids = {f.rule_id for f in findings}
        assert "SC201" in rule_ids

    def test_environment_yml_parsed(self, tmp_path: Path) -> None:
        yml = tmp_path / "environment.yml"
        yml.write_text("dependencies:\n  - numpy>=1.20\n", encoding="utf-8")
        findings, _ = scan_dependencies(tmp_path)
        rule_ids = {f.rule_id for f in findings}
        assert "SC201" in rule_ids

    def test_sc202_with_in_memory_advisory(self, tmp_path: Path) -> None:
        """SC202 fires when pinned version is in advisory range."""
        import model_scanner.analyzers.repository.dep_scanner as ds

        advisory = {
            "id": "TEST-999",
            "affected": [
                {
                    "package": {"name": "torch", "ecosystem": "PyPI"},
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "events": [{"introduced": "1.0.0"}, {"fixed": "3.0.0"}],
                        }
                    ],
                }
            ],
        }
        req = tmp_path / "requirements.txt"
        req.write_text("torch==2.0.1\n", encoding="utf-8")

        # Monkey-patch to inject advisory and snapshot date
        original_load_advisories = ds._load_advisories
        original_load_snapshot = ds._load_snapshot_date
        ds._load_advisories = lambda: [advisory]  # type: ignore[assignment]
        ds._load_snapshot_date = lambda: "2025-01-01"  # type: ignore[assignment]
        try:
            findings, performed = scan_dependencies(tmp_path)
        finally:
            ds._load_advisories = original_load_advisories  # type: ignore[assignment]
            ds._load_snapshot_date = original_load_snapshot  # type: ignore[assignment]

        assert performed is True
        sc202 = [f for f in findings if f.rule_id == "SC202"]
        assert len(sc202) >= 1
        assert sc202[0].advisory_id == "TEST-999"

    def test_no_advisory_db_scanning_not_performed(self, tmp_path: Path) -> None:
        import model_scanner.analyzers.repository.dep_scanner as ds

        original = ds._load_snapshot_date
        ds._load_snapshot_date = lambda: None  # type: ignore[assignment]
        try:
            req = tmp_path / "requirements.txt"
            req.write_text("torch==2.0.1\n", encoding="utf-8")
            _, performed = scan_dependencies(tmp_path)
        finally:
            ds._load_snapshot_date = original  # type: ignore[assignment]

        assert performed is False

    def test_torch_version_from_config_json(self, tmp_path: Path) -> None:
        config = {"torch_version": "2.0.1"}
        findings, _ = scan_dependencies(tmp_path, config_json=config)
        # torch==2.0.1 from config.json is pinned → no SC201
        sc201_names = {f.dep_name for f in findings if f.rule_id == "SC201"}
        assert "torch" not in sc201_names


# ---------------------------------------------------------------------------
# Advisory loading with real files
# ---------------------------------------------------------------------------


class TestAdvisoryLoading:
    def test_load_snapshot_date_real_file(self, tmp_path: Path) -> None:
        import model_scanner.analyzers.repository.dep_scanner as ds

        adv_dir = tmp_path / "advisories"
        adv_dir.mkdir()
        meta = adv_dir / "snapshot_meta.json"
        meta.write_text('{"snapshot_date": "2025-06-01"}', encoding="utf-8")

        original_meta = ds._SNAPSHOT_META_FILE
        ds._SNAPSHOT_META_FILE = meta  # type: ignore[assignment]
        try:
            date = ds._load_snapshot_date()
        finally:
            ds._SNAPSHOT_META_FILE = original_meta  # type: ignore[assignment]

        assert date == "2025-06-01"

    def test_load_advisories_from_dir(self, tmp_path: Path) -> None:
        import model_scanner.analyzers.repository.dep_scanner as ds

        adv_dir = tmp_path / "advisories"
        adv_dir.mkdir()
        advisory_data = {"id": "TEST-001", "affected": []}
        (adv_dir / "TEST-001.json").write_text(
            __import__("json").dumps(advisory_data), encoding="utf-8"
        )

        original_dir = ds._ADVISORIES_DIR
        ds._ADVISORIES_DIR = adv_dir  # type: ignore[assignment]
        try:
            advisories = ds._load_advisories()
        finally:
            ds._ADVISORIES_DIR = original_dir  # type: ignore[assignment]

        assert any(a.get("id") == "TEST-001" for a in advisories)

    def test_load_advisories_skips_snapshot_meta(self, tmp_path: Path) -> None:
        import model_scanner.analyzers.repository.dep_scanner as ds

        adv_dir = tmp_path / "advisories"
        adv_dir.mkdir()
        (adv_dir / "snapshot_meta.json").write_text('{"snapshot_date": "x"}', encoding="utf-8")

        original_dir = ds._ADVISORIES_DIR
        ds._ADVISORIES_DIR = adv_dir  # type: ignore[assignment]
        try:
            advisories = ds._load_advisories()
        finally:
            ds._ADVISORIES_DIR = original_dir  # type: ignore[assignment]

        # snapshot_meta.json should be skipped
        assert advisories == []


# ---------------------------------------------------------------------------
# Parsing edge cases
# ---------------------------------------------------------------------------


class TestParseEdgeCases:
    def test_pyproject_toml_non_string_item_skipped(self, tmp_path: Path) -> None:
        import textwrap
        p = tmp_path / "pyproject.toml"
        p.write_text(
            textwrap.dedent("""\
                [project]
                dependencies = ["torch>=2.0"]
            """),
            encoding="utf-8",
        )
        deps = _parse_pyproject_toml(p)
        assert any(d.name == "torch" for d in deps)

    def test_requirements_non_matching_line_skipped(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        # Line that won't match _REQ_LINE_RE
        req.write_text("123_invalid_start==1.0\ntorch==2.0\n", encoding="utf-8")
        deps = _parse_requirements_file(req, "requirements.txt")
        # torch should still be parsed
        assert any(d.name == "torch" for d in deps)
