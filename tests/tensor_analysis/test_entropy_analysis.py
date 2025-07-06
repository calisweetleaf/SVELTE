import pytest
import numpy as np
import logging
import os
from pathlib import Path
from src.tensor_analysis.entropy_analysis import EntropyAnalysisModule

# Helper to create dummy tensor field
def create_dummy_tensor_field(num_tensors=1, shape=(10, 10), add_nan=False, add_inf=False):
    field = {}
    for i in range(num_tensors):
        tensor = np.random.rand(*shape).astype('float32')
        if add_nan:
            tensor[0,0] = np.nan
        if add_inf:
            tensor[0,1] = np.inf
        field[f"tensor_{i}"] = tensor
    return field

@pytest.fixture
def simple_1d_tensor_data():
    # Data for which entropy can be easily estimated or bounded
    # For Shannon: -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0 for two equiprobable states
    # For this data, with 2 bins, hist might be [~0.5, ~0.5] if bins split at 0.5
    return {"tensor_1d": np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.1, 0.9])} # 8 samples

@pytest.fixture
def simple_2d_tensor_data():
    return {"tensor_2d": np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [0.4, 0.5, 0.6]])} # 3x3

@pytest.fixture
def nan_inf_tensor_data():
    return {"tensor_nan_inf": np.array([0.1, np.nan, 0.5, np.inf, 0.9])}


class TestEntropyAnalysisModule:
    def test_initialization_valid(self):
        field = create_dummy_tensor_field()
        module = EntropyAnalysisModule(field)
        assert module is not None
        assert module.tensor_field == field
        assert module.logger.level == logging.INFO # Default

    def test_initialization_custom_config(self):
        field = create_dummy_tensor_field()
        config = {'log_level': logging.DEBUG, 'default_bins': 128}
        module = EntropyAnalysisModule(field, config=config)
        assert module.config['log_level'] == logging.DEBUG
        assert module.config['default_bins'] == 128
        assert module.logger.level == logging.DEBUG

    def test_initialization_invalid_tensor_field_type(self):
        with pytest.raises(TypeError, match="tensor_field must be a dictionary"):
            EntropyAnalysisModule("not_a_dict")

    def test_initialization_empty_tensor_field_warning(self, caplog):
        EntropyAnalysisModule({})
        assert "Empty tensor field provided" in caplog.text

    def test_initialization_invalid_tensor_key_type(self):
        field = {123: np.random.rand(2,2)}
        with pytest.raises(TypeError, match="Tensor field key '123' is not a string"):
            EntropyAnalysisModule(field)

    def test_initialization_invalid_tensor_value_type(self):
        field = {"tensor1": [1,2,3]}
        with pytest.raises(TypeError, match="Tensor 'tensor1' is not a numpy.ndarray"):
            EntropyAnalysisModule(field)

    def test_initialization_empty_tensor_value(self):
        field = {"tensor1": np.array([])}
        with pytest.raises(ValueError, match="Tensor 'tensor1' is empty"):
            EntropyAnalysisModule(field)

    def test_initialization_tensor_with_nan_warning(self, caplog):
        field = create_dummy_tensor_field(add_nan=True)
        EntropyAnalysisModule(field)
        assert f"Tensor '{list(field.keys())[0]}' contains NaN values" in caplog.text

    def test_initialization_tensor_with_inf_warning(self, caplog):
        field = create_dummy_tensor_field(add_inf=True)
        EntropyAnalysisModule(field)
        assert f"Tensor '{list(field.keys())[0]}' contains infinite values" in caplog.text

    # --- Entropy Calculation Tests ---
    @pytest.mark.parametrize("method, extra_params", [
        ("shannon", {}),
        ("renyi", {"alpha": 2.0}),
        ("tsallis", {"alpha": 2.0}),
        ("scipy", {}),
        ("differential", {})
    ])
    def test_calculate_entropy_value_methods(self, simple_1d_tensor_data, method, extra_params):
        module = EntropyAnalysisModule(simple_1d_tensor_data)
        data = simple_1d_tensor_data["tensor_1d"]
        # Note: _calculate_entropy_value filters nan/inf, but simple_1d_tensor_data is clean
        entropy_val = module._calculate_entropy_value(data, bins=2, method=method, **extra_params)
        assert isinstance(entropy_val, float)
        if method == "shannon": # With 2 bins for data [0.1,0.2,0.8,0.9,0.15,0.85,0.1,0.9]
                                # Bins could be e.g. [0.1, 0.5], (0.5, 0.9]
                                # Counts: 4 in first, 4 in second. P=[0.5, 0.5]. Entropy = 1.0
            # This depends HEAVILY on np.histogram binning strategy with just 2 bins.
            # Let's test a more predictable case for shannon
            predictable_data = np.array([1,1,1,1,0,0,0,0]) # P=[0.5,0.5] -> H=1
            entropy_predictable = module._calculate_entropy_value(predictable_data, bins=2, method="shannon")
            assert np.isclose(entropy_predictable, 1.0)

            all_same_data = np.ones(8) # P=[1.0] -> H=0
            entropy_zeros = module._calculate_entropy_value(all_same_data, bins=2, method="shannon")
            assert np.isclose(entropy_zeros, 0.0)

    def test_calculate_entropy_value_empty_after_filter(self, nan_inf_tensor_data):
        module = EntropyAnalysisModule(nan_inf_tensor_data)
        data = np.array([np.nan, np.inf]) # Will be empty after filtering
        entropy_val = module._calculate_entropy_value(data, bins=2, method="shannon", alpha=2.0)
        assert np.isclose(entropy_val, 0.0)

    def test_calculate_entropy_invalid_method(self, simple_1d_tensor_data):
        module = EntropyAnalysisModule(simple_1d_tensor_data)
        data = simple_1d_tensor_data["tensor_1d"]
        with pytest.raises(ValueError, match="Unknown entropy method: invalid_method"):
            module._calculate_entropy_value(data, bins=2, method="invalid_method", alpha=2.0)

    def test_compute_entropy_1d(self, simple_1d_tensor_data):
        module = EntropyAnalysisModule(simple_1d_tensor_data)
        entropy_maps = module.compute_entropy(bins=2, method="shannon")
        assert "tensor_1d" in entropy_maps
        assert isinstance(entropy_maps["tensor_1d"], (float, np.number)) # Single value for 1D tensor

    def test_compute_entropy_2d(self, simple_2d_tensor_data):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        tensor_shape = simple_2d_tensor_data["tensor_2d"].shape
        entropy_maps = module.compute_entropy(bins=2, method="shannon")
        assert "tensor_2d" in entropy_maps
        # Entropy computed along last axis, so output shape is tensor.shape[:-1]
        assert entropy_maps["tensor_2d"].shape == tensor_shape[:-1]
        assert entropy_maps["tensor_2d"].size == tensor_shape[0]


    # --- Gradient Calculation Tests ---
    @pytest.mark.parametrize("method", ["gaussian", "central", "sobel", "prewitt"])
    def test_compute_entropy_gradient_methods_2d(self, simple_2d_tensor_data, method):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        tensor = simple_2d_tensor_data["tensor_2d"]
        gradients = module.compute_entropy_gradient(sigma=1.0, method=method)
        assert "tensor_2d" in gradients
        assert "magnitude" in gradients["tensor_2d"]
        assert "vectors" in gradients["tensor_2d"]
        assert gradients["tensor_2d"]["magnitude"].shape == tensor.shape
        assert isinstance(gradients["tensor_2d"]["vectors"], list)
        assert len(gradients["tensor_2d"]["vectors"]) == tensor.ndim
        for grad_component in gradients["tensor_2d"]["vectors"]:
            assert grad_component.shape == tensor.shape

    def test_compute_entropy_gradient_invalid_method(self, simple_1d_tensor_data):
        module = EntropyAnalysisModule(simple_1d_tensor_data)
        with pytest.raises(ValueError, match="Unknown gradient method: invalid_grad_method"):
            module.compute_entropy_gradient(method="invalid_grad_method")

    # --- Semantic Density Tests ---
    @pytest.mark.parametrize("method", ["sliding", "global", "adaptive"])
    def test_semantic_density_map_methods_2d(self, simple_2d_tensor_data, method):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        tensor = simple_2d_tensor_data["tensor_2d"]
        density_maps = module.semantic_density_map(window_size=3, method=method)
        assert "tensor_2d" in density_maps
        assert density_maps["tensor_2d"].shape == tensor.shape

    def test_semantic_density_invalid_method(self, simple_1d_tensor_data):
        module = EntropyAnalysisModule(simple_1d_tensor_data)
        with pytest.raises(ValueError, match="Unknown density method: invalid_density_method"):
            module.semantic_density_map(method="invalid_density_method")

    # --- Orchestration and Other Methods ---
    def test_analyze_method(self, simple_2d_tensor_data):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        results = module.analyze(parallel=False) # Test sequential first
        assert "tensor_2d" in results
        for key in ['entropy', 'gradient', 'density', 'anomalies']:
            assert key in results["tensor_2d"]
        # Check that underlying dicts are populated
        assert "tensor_2d" in module.entropy_maps
        assert "tensor_2d" in module.gradients
        assert "tensor_2d" in module.semantic_density
        assert "tensor_2d" in module.anomaly_scores

    # Minimal test for parallel execution to ensure it runs
    # Detailed testing of parallelism is complex
    def test_analyze_method_parallel(self, simple_2d_tensor_data):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        results = module.analyze(parallel=True)
        assert "tensor_2d" in results
        for key in ['entropy', 'gradient', 'density', 'anomalies']:
            assert key in results["tensor_2d"]

    def test_detect_anomalies(self, simple_2d_tensor_data):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        # Need to compute some base metrics first
        module.compute_entropy()
        module.compute_entropy_gradient()
        module.semantic_density_map()

        anomaly_scores = module.detect_anomalies()
        assert "tensor_2d" in anomaly_scores
        assert anomaly_scores["tensor_2d"].shape == simple_2d_tensor_data["tensor_2d"].shape

    def test_cluster_anomalies(self, simple_2d_tensor_data):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        module.analyze() # Populates anomaly_scores

        # This test is basic, mainly checking execution. DBSCAN results are sensitive.
        # Ensure anomaly_scores is not empty for this test to be meaningful
        if np.sum(module.anomaly_scores["tensor_2d"] > 2.0) > 0 :
             cluster_results = module.cluster_anomalies(eps=0.5, min_samples=1) # min_samples=1 to ensure clusters
             assert "tensor_2d" in cluster_results
             assert "count" in cluster_results["tensor_2d"]
        else:
            pytest.skip("No anomalies detected with default settings, skipping cluster test or adjust data/params.")


    def test_visualize_entropy_run(self, simple_2d_tensor_data, tmp_path, caplog):
        # This test mainly checks if the function runs without error and respects display=False
        # It does not check the content of the plot.
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        module.compute_entropy()

        # Mock matplotlib.pyplot.show if necessary for non-interactive environments
        # For now, rely on display=False

        output_dir = tmp_path / "viz_output"
        file_paths = module.visualize_entropy(output_dir=str(output_dir), display=False)

        assert "tensor_2d" in file_paths
        assert Path(file_paths["tensor_2d"]).exists()
        assert "No entropy maps available" not in caplog.text

    def test_visualize_entropy_no_maps(self, simple_2d_tensor_data, caplog):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        # No compute_entropy called
        module.visualize_entropy(display=False)
        assert "No entropy maps available. Run compute_entropy first." in caplog.text


    @pytest.mark.parametrize("fmt", ["hdf5", "npz", "json"])
    def test_save_results_formats(self, simple_2d_tensor_data, tmp_path, fmt):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        module.analyze(parallel=False) # Populate all result dicts

        output_file = tmp_path / f"results.{fmt}"
        saved_path = module.save_results(str(output_file), format=fmt)

        assert Path(saved_path).exists()
        assert Path(saved_path).name == f"results.{fmt}"

        # Basic check for content (can be expanded)
        if fmt == "hdf5":
            import h5py
            with h5py.File(output_file, 'r') as f:
                assert 'entropy_maps' in f
                assert 'tensor_2d' in f['entropy_maps']
        elif fmt == "npz":
            data = np.load(output_file)
            assert 'entropy_tensor_2d' in data

    def test_save_results_invalid_format(self, simple_2d_tensor_data, tmp_path):
        module = EntropyAnalysisModule(simple_2d_tensor_data)
        module.analyze(parallel=False)
        with pytest.raises(ValueError, match="Unsupported file format: invalid_fmt"):
            module.save_results(str(tmp_path / "results.invalid_fmt"), format="invalid_fmt")

    def test_batch_process_run(self, simple_1d_tensor_data, simple_2d_tensor_data):
        batch = {
            "batch1": simple_1d_tensor_data,
            "batch2": simple_2d_tensor_data
        }
        # Use a fresh module for batch processing to avoid state interference
        # Or ensure batch_process correctly uses temporary/isolated modules
        module = EntropyAnalysisModule({}) # Main module not used for its tensor_field here

        results = module.batch_process(batch, parallel=False) # Test sequential first
        assert "batch1" in results
        assert "batch2" in results
        assert "tensor_1d" in results["batch1"]
        assert "tensor_2d" in results["batch2"]

    def test_batch_process_parallel_run(self, simple_1d_tensor_data, simple_2d_tensor_data):
        batch = {
            "batch1": simple_1d_tensor_data,
            "batch2": simple_2d_tensor_data
        }
        module = EntropyAnalysisModule({})

        results = module.batch_process(batch, parallel=True)
        assert "batch1" in results
        assert "batch2" in results
        assert "tensor_1d" in results["batch1"]
        assert "tensor_2d" in results["batch2"]

print("Created test file: tests/tensor_analysis/test_entropy_analysis.py")
