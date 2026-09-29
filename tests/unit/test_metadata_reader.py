"""
tests.unit.test_metadata_reader
=================================

Unit tests for model_scanner.analyzers.repository.metadata_reader.
"""

from __future__ import annotations

import json
from pathlib import Path

from model_scanner.analyzers.repository.metadata_reader import (
    _list_of_str,
    _parse_json_bounded,
    _parse_yaml_frontmatter,
    _read_bounded,
    _str_or_none,
    read_repo_metadata,
)

# ---------------------------------------------------------------------------
# _read_bounded
# ---------------------------------------------------------------------------


class TestReadBounded:
    def test_reads_normal_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        result = _read_bounded(f)
        assert result == "hello"

    def test_returns_none_on_missing(self, tmp_path: Path) -> None:
        result = _read_bounded(tmp_path / "nonexistent.txt")
        assert result is None


# ---------------------------------------------------------------------------
# _parse_json_bounded
# ---------------------------------------------------------------------------


class TestParseJsonBounded:
    def test_valid_object(self) -> None:
        obj, err = _parse_json_bounded('{"a": 1}', "test.json")
        assert obj == {"a": 1}
        assert err is None

    def test_invalid_json(self) -> None:
        obj, err = _parse_json_bounded("not json", "test.json")
        assert obj == {}
        assert err is not None
        assert "test.json" in err

    def test_array_root_is_error(self) -> None:
        obj, err = _parse_json_bounded("[1, 2, 3]", "test.json")
        assert obj == {}
        assert err is not None


# ---------------------------------------------------------------------------
# _parse_yaml_frontmatter
# ---------------------------------------------------------------------------


class TestParseYamlFrontmatter:
    def test_basic_frontmatter(self) -> None:
        text = "---\nlicense: apache-2.0\nlibrary_name: transformers\n---\nBody\n"
        result = _parse_yaml_frontmatter(text)
        assert result.get("license") == "apache-2.0"
        assert result.get("library_name") == "transformers"

    def test_no_frontmatter_returns_empty(self) -> None:
        result = _parse_yaml_frontmatter("Just plain text.\n")
        assert result == {}

    def test_unclosed_frontmatter_returns_empty(self) -> None:
        result = _parse_yaml_frontmatter("---\nlicense: mit\n")
        assert result == {}

    def test_anchor_alias_rejected(self) -> None:
        text = "---\nlicense: &anchor mit\n---\nBody\n"
        result = _parse_yaml_frontmatter(text)
        assert result == {}

    def test_inline_list_tags(self) -> None:
        text = "---\ntags: [nlp, text-generation]\n---\n"
        result = _parse_yaml_frontmatter(text)
        assert isinstance(result.get("tags"), list)
        assert "nlp" in result["tags"]  # type: ignore[operator]

    def test_block_list_tags(self) -> None:
        text = "---\ntags:\n  - nlp\n  - vision\n---\n"
        result = _parse_yaml_frontmatter(text)
        assert isinstance(result.get("tags"), list)
        assert "nlp" in result["tags"]  # type: ignore[operator]

    def test_null_value_treated_as_list(self) -> None:
        text = "---\ntags:\n  - one\n  - two\n---\n"
        result = _parse_yaml_frontmatter(text)
        assert result.get("tags") == ["one", "two"]

    def test_unknown_key_ignored(self) -> None:
        text = "---\nunknown_field: value\nlicense: mit\n---\n"
        result = _parse_yaml_frontmatter(text)
        assert "unknown_field" not in result
        assert result.get("license") == "mit"

    def test_indented_line_skipped(self) -> None:
        text = "---\n  nested_key: value\nlicense: apache-2.0\n---\n"
        result = _parse_yaml_frontmatter(text)
        assert "nested_key" not in result

    def test_comment_line_skipped(self) -> None:
        text = "---\n# this is a comment\nlicense: mit\n---\n"
        result = _parse_yaml_frontmatter(text)
        assert result.get("license") == "mit"


# ---------------------------------------------------------------------------
# _str_or_none and _list_of_str
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_str_or_none_with_string(self) -> None:
        assert _str_or_none("hello") == "hello"

    def test_str_or_none_with_empty(self) -> None:
        assert _str_or_none("") is None

    def test_str_or_none_with_none(self) -> None:
        assert _str_or_none(None) is None

    def test_list_of_str_with_list(self) -> None:
        assert _list_of_str(["a", "b", 3]) == ["a", "b"]

    def test_list_of_str_with_str(self) -> None:
        assert _list_of_str("hello") == ["hello"]

    def test_list_of_str_with_int(self) -> None:
        assert _list_of_str(42) == []


# ---------------------------------------------------------------------------
# read_repo_metadata — integration
# ---------------------------------------------------------------------------


class TestReadRepoMetadata:
    def test_empty_dir_returns_defaults(self, tmp_path: Path) -> None:
        meta = read_repo_metadata(tmp_path)
        assert meta.model_type is None
        assert meta.architectures == []
        assert meta.license is None

    def test_config_json_parsed(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "transformers_version": "4.30.0",
                "torch_dtype": "float16",
            }),
            encoding="utf-8",
        )
        meta = read_repo_metadata(tmp_path)
        assert meta.model_type == "llama"
        assert "LlamaForCausalLM" in meta.architectures
        assert meta.transformers_version == "4.30.0"
        assert meta.torch_dtype == "float16"

    def test_invalid_config_json_records_error(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_bytes(b"not json {{{{")
        meta = read_repo_metadata(tmp_path)
        assert len(meta.parse_errors) > 0

    def test_tokenizer_config_parsed(self, tmp_path: Path) -> None:
        tok = tmp_path / "tokenizer_config.json"
        tok.write_text(
            json.dumps({"tokenizer_class": "LlamaTokenizer"}),
            encoding="utf-8",
        )
        meta = read_repo_metadata(tmp_path)
        assert meta.tokenizer_class == "LlamaTokenizer"

    def test_readme_frontmatter_parsed(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        frontmatter = (
            "---\n"
            "license: apache-2.0\n"
            "base_model: meta-llama/Llama-2-7b\n"
            "library_name: transformers\n"
            "tags:\n"
            "  - text-generation\n"
            "---\n"
            "# Model\n"
        )
        readme.write_text(frontmatter, encoding="utf-8")
        meta = read_repo_metadata(tmp_path)
        assert meta.license == "apache-2.0"
        assert meta.base_model == "meta-llama/Llama-2-7b"
        assert meta.library_name == "transformers"
        assert "text-generation" in meta.tags

    def test_readme_tags_inline(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(
            "---\ntags: [nlp, generation]\n---\n",
            encoding="utf-8",
        )
        meta = read_repo_metadata(tmp_path)
        assert "nlp" in meta.tags

    def test_readme_tags_as_string(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("---\ntags: nlp\n---\n", encoding="utf-8")
        meta = read_repo_metadata(tmp_path)
        # Single-string tags should be wrapped in a list
        assert meta.tags == ["nlp"]

    def test_config_json_non_object_records_error(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text("[1, 2, 3]", encoding="utf-8")
        meta = read_repo_metadata(tmp_path)
        assert len(meta.parse_errors) > 0
