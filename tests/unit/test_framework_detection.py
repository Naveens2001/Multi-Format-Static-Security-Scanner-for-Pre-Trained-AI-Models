"""
tests.unit.test_framework_detection
=====================================

Unit tests for model_scanner.detection.framework — static ML framework inference.
"""

from __future__ import annotations

from model_scanner.detection.framework import (
    Framework,
    FrameworkConfidence,
    FrameworkIdentification,
    infer_from_gguf_metadata,
    infer_from_pickle_globals,
    infer_from_pytorch_zip,
    infer_from_safetensors_metadata,
)


class TestFrameworkIdentification:
    def test_unknown_factory(self) -> None:
        fi = FrameworkIdentification.unknown()
        assert fi.framework == Framework.UNKNOWN
        assert fi.confidence == FrameworkConfidence.LOW
        assert fi.evidence == ()

    def test_enum_values(self) -> None:
        assert Framework.PYTORCH == "PyTorch"
        assert FrameworkConfidence.HIGH == "HIGH"


class TestInferFromPickleGlobals:
    def test_empty_globals_returns_unknown(self) -> None:
        fi = infer_from_pickle_globals([])
        assert fi.framework == Framework.UNKNOWN

    def test_pytorch_globals_detected(self) -> None:
        fi = infer_from_pickle_globals(["torch._utils", "torch.storage"])
        assert fi.framework == Framework.PYTORCH
        assert fi.confidence == FrameworkConfidence.HIGH

    def test_single_pytorch_module_medium_confidence(self) -> None:
        fi = infer_from_pickle_globals(["torch._utils"])
        assert fi.framework == Framework.PYTORCH
        assert fi.confidence == FrameworkConfidence.MEDIUM

    def test_sklearn_globals_detected(self) -> None:
        fi = infer_from_pickle_globals(["sklearn.linear_model", "sklearn.pipeline"])
        assert fi.framework == Framework.SKLEARN

    def test_xgboost_globals_detected(self) -> None:
        fi = infer_from_pickle_globals(["xgboost.core"])
        assert fi.framework == Framework.XGBOOST

    def test_lightgbm_globals_detected(self) -> None:
        fi = infer_from_pickle_globals(["lightgbm.basic"])
        assert fi.framework == Framework.LIGHTGBM

    def test_numpy_globals_detected(self) -> None:
        fi = infer_from_pickle_globals(["numpy.core.multiarray"])
        assert fi.framework == Framework.NUMPY

    def test_unknown_globals_return_unknown(self) -> None:
        fi = infer_from_pickle_globals(["builtins", "collections", "os"])
        assert fi.framework == Framework.UNKNOWN

    def test_submodule_detection(self) -> None:
        fi = infer_from_pickle_globals(["torch.nn.modules.linear"])
        assert fi.framework == Framework.PYTORCH


class TestInferFromPyTorchZip:
    def test_canonical_pytorch_structure(self) -> None:
        fi = infer_from_pytorch_zip(
            member_names=["archive/data.pkl", "archive/version"],
            pickle_global_modules=[],
        )
        assert fi.framework == Framework.PYTORCH
        assert fi.confidence == FrameworkConfidence.HIGH

    def test_data_pkl_with_rebuild_hits(self) -> None:
        fi = infer_from_pytorch_zip(
            member_names=["archive/data.pkl"],
            pickle_global_modules=["torch._utils._rebuild_tensor_v2"],
        )
        assert fi.framework == Framework.PYTORCH
        assert fi.confidence == FrameworkConfidence.HIGH

    def test_partial_evidence_medium_confidence(self) -> None:
        fi = infer_from_pytorch_zip(
            member_names=["archive/data.pkl"],
            pickle_global_modules=[],
        )
        assert fi.framework == Framework.PYTORCH
        assert fi.confidence == FrameworkConfidence.MEDIUM

    def test_unknown_when_no_evidence(self) -> None:
        fi = infer_from_pytorch_zip(
            member_names=["weights.bin"],
            pickle_global_modules=[],
        )
        assert fi.framework == Framework.UNKNOWN

    def test_fallback_to_pickle_globals(self) -> None:
        fi = infer_from_pytorch_zip(
            member_names=["data.bin"],
            pickle_global_modules=["sklearn.linear_model"],
        )
        assert fi.framework == Framework.SKLEARN


class TestInferFromSafetensorsMetadata:
    def test_format_pt_returns_pytorch(self) -> None:
        fi = infer_from_safetensors_metadata({"format": "pt"}, set())
        assert fi.framework == Framework.PYTORCH
        assert fi.confidence == FrameworkConfidence.HIGH

    def test_format_tf_returns_tensorflow(self) -> None:
        fi = infer_from_safetensors_metadata({"format": "tf"}, set())
        assert fi.framework == Framework.TENSORFLOW

    def test_format_flax_returns_jax(self) -> None:
        fi = infer_from_safetensors_metadata({"format": "flax"}, set())
        assert fi.framework == Framework.JAX_FLAX

    def test_format_np_returns_numpy(self) -> None:
        fi = infer_from_safetensors_metadata({"format": "np"}, set())
        assert fi.framework == Framework.NUMPY

    def test_bf16_dtype_signals_pytorch_medium(self) -> None:
        fi = infer_from_safetensors_metadata({}, {"BF16", "F32"})
        assert fi.framework == Framework.PYTORCH
        assert fi.confidence == FrameworkConfidence.MEDIUM

    def test_unknown_when_no_signals(self) -> None:
        fi = infer_from_safetensors_metadata({}, {"F32"})
        assert fi.framework == Framework.UNKNOWN


class TestInferFromGGUFMetadata:
    def test_architecture_key_detected(self) -> None:
        fi = infer_from_gguf_metadata({"general.architecture": "llama"})
        assert fi.framework == Framework.PYTORCH
        assert "llama" in " ".join(fi.evidence)

    def test_file_type_key_detected(self) -> None:
        fi = infer_from_gguf_metadata({"general.file_type": 15})
        assert fi.framework == Framework.PYTORCH

    def test_quantization_key_detected(self) -> None:
        fi = infer_from_gguf_metadata({"llm.quantization_version": 2})
        assert fi.framework == Framework.PYTORCH

    def test_empty_metadata_returns_unknown(self) -> None:
        fi = infer_from_gguf_metadata({})
        assert fi.framework == Framework.UNKNOWN
