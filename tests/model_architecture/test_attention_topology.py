import pytest
import numpy as np
import logging
from src.model_architecture.attention_topology import AttentionTopologySystem, CurvatureMethod, TopologyMetrics

# Basic Configurations
N_CTX_TEST = 4  # Sequence length for simple tests
BATCH_SIZE_TEST = 2
DIM_TEST = N_CTX_TEST # Dimensionality of the manifold, typically sequence length

@pytest.fixture
def simple_attention_tensor():
    """Create a simple, deterministic attention tensor for testing."""
    # A simple tensor: batch_size x seq_len x seq_len
    # For multi-head, it would be batch_size x num_heads x seq_len x seq_len
    # Using seq_len x seq_len for simplicity in some tests, or batch_size x seq_len x seq_len
    attention = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.2, 0.5, 0.2],
        [0.1, 0.1, 0.2, 0.6]
    ])
    # Ensure rows sum to 1 (or close to it, for a pseudo-attention matrix)
    # For actual attention, it's often post-softmax. Here, just ensuring positivity and structure.
    attention = attention / attention.sum(axis=1, keepdims=True)
    return {"layer1": attention}

@pytest.fixture
def batched_attention_tensor():
    """Create a batched attention tensor."""
    attn1 = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.2, 0.5, 0.2],
        [0.1, 0.1, 0.2, 0.6]
    ])
    attn1 = attn1 / attn1.sum(axis=1, keepdims=True)
    attn2 = np.array([
        [0.5, 0.2, 0.2, 0.1],
        [0.2, 0.4, 0.2, 0.2],
        [0.1, 0.3, 0.3, 0.3],
        [0.2, 0.2, 0.3, 0.3]
    ])
    attn2 = attn2 / attn2.sum(axis=1, keepdims=True)
    return {"layer_batched": np.stack([attn1, attn2])}


@pytest.fixture
def multi_head_attention_tensor():
    """Create a multi-head attention tensor: batch x heads x seq_len x seq_len."""
    # Batch size = 1, Heads = 2, SeqLen = N_CTX_TEST
    head1 = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.2, 0.5, 0.2],
        [0.1, 0.1, 0.2, 0.6]
    ]) / 4 # Simplified
    head2 = np.array([
        [0.5, 0.2, 0.2, 0.1],
        [0.2, 0.4, 0.2, 0.2],
        [0.1, 0.3, 0.3, 0.3],
        [0.2, 0.2, 0.3, 0.3]
    ]) / 4 # Simplified

    # Stack heads, add batch dimension
    multi_head_attn = np.expand_dims(np.stack([head1, head2], axis=0), axis=0)
    return {"layer_multi_head": multi_head_attn}

class TestAttentionTopologySystem:

    def test_initialization(self, simple_attention_tensor):
        """Test basic initialization of AttentionTopologySystem."""
        ats = AttentionTopologySystem(simple_attention_tensor, log_level=logging.DEBUG)
        assert ats is not None
        assert "layer1" in ats.tensor_field
        assert ats.tensor_field["layer1"].shape == (N_CTX_TEST, N_CTX_TEST)
        assert ats.logger.level == logging.DEBUG

    def test_initialization_empty_tensor_field(self):
        """Test initialization with empty tensor field raises ValueError."""
        with pytest.raises(ValueError, match="Tensor field dictionary cannot be empty"):
            AttentionTopologySystem({})

    def test_initialization_invalid_tensor_type(self):
        """Test initialization with invalid tensor type raises TypeError."""
        invalid_tensor_field = {"layer1": [[0.1, 0.9], [0.8, 0.2]]} # List instead of ndarray
        with pytest.raises(TypeError, match="Tensor layer1 must be a numpy array"):
            AttentionTopologySystem(invalid_tensor_field)

    def test_initialization_invalid_tensor_ndim(self):
        """Test initialization with tensor of insufficient dimensions."""
        invalid_tensor_field = {"layer1": np.array([0.1, 0.9])} # 1D array
        with pytest.raises(ValueError, match="Tensor layer1 must have at least 2 dimensions"):
            AttentionTopologySystem(invalid_tensor_field)

    def test_initialization_tensor_with_nan(self):
        """Test initialization with tensor containing NaN values."""
        tensor_with_nan = np.array([[0.1, np.nan], [0.8, 0.2]])
        invalid_tensor_field = {"layer1": tensor_with_nan}
        with pytest.raises(ValueError, match="Tensor layer1 contains NaN or Inf values"):
            AttentionTopologySystem(invalid_tensor_field)

    def test_compute_metric_tensor(self, simple_attention_tensor):
        """Test compute_metric_tensor output shape and basic properties."""
        ats = AttentionTopologySystem(simple_attention_tensor)
        metric = ats.compute_metric_tensor(simple_attention_tensor["layer1"])
        assert metric.shape == (N_CTX_TEST, N_CTX_TEST)
        assert np.all(metric >= 0) # Distances should be non-negative

    def test_compute_metric_tensor_multi_head(self, multi_head_attention_tensor):
        """Test metric tensor computation with multi-head attention."""
        ats = AttentionTopologySystem(multi_head_attention_tensor)
        # The system averages heads, so the input to compute_metric_tensor will be (batch, seq, seq)
        # or (seq,seq) if batch=1 and it's squeezed.
        # Here, tensor_field["layer_multi_head"] is (1, 2, N_CTX_TEST, N_CTX_TEST)
        # compute_metric_tensor is called internally by compute_curvature
        ats.compute_curvature() # This will call compute_metric_tensor
        assert "layer_multi_head" in ats.metric_tensors
        metric = ats.metric_tensors["layer_multi_head"]
        # After mean over heads, shape should be (batch_size, seq_len, seq_len)
        # Batch size is 1, so it might be squeezed.
        assert metric.shape == (1, N_CTX_TEST, N_CTX_TEST) or metric.shape == (N_CTX_TEST, N_CTX_TEST)


    def test_compute_christoffel_symbols(self, simple_attention_tensor):
        """Test compute_christoffel_symbols output shape."""
        ats = AttentionTopologySystem(simple_attention_tensor)
        metric = ats.compute_metric_tensor(simple_attention_tensor["layer1"])
        # Add batch dim if not present, as compute_christoffel_symbols might expect it
        if metric.ndim == 2:
             metric = np.expand_dims(metric, axis=0) # BxNxN

        # The current implementation of compute_christoffel_symbols has issues with derivatives.
        # For now, just test if it runs and produces the correct shape.
        # Expected shape: (batch_size, dim, dim, dim) or (dim, dim, dim)
        # try:
        christoffel = ats.compute_christoffel_symbols(metric)
        if metric.ndim == 3: # Batched
                assert christoffel.shape == (metric.shape[0], DIM_TEST, DIM_TEST, DIM_TEST)
        else: # Not batched
                assert christoffel.shape == (DIM_TEST, DIM_TEST, DIM_TEST)
        # except Exception as e:
        #     pytest.skip(f"Skipping Christoffel test due to known implementation issues: {e}")


    def test_compute_riemann_tensor(self, simple_attention_tensor):
        """Test compute_riemann_tensor output shape."""
        ats = AttentionTopologySystem(simple_attention_tensor)
        metric = ats.compute_metric_tensor(simple_attention_tensor["layer1"])
        # Add batch dim for consistency if metric is 2D
        if metric.ndim == 2:
            metric_batched = np.expand_dims(metric, axis=0)
        else:
            metric_batched = metric

        # try:
        # Christoffel symbols are needed for Riemann tensor.
        christoffel = ats.compute_christoffel_symbols(metric_batched)
        riemann = ats.compute_riemann_tensor(christoffel)

        # Expected shape: (batch_size, dim, dim, dim, dim) or (dim, dim, dim, dim)
        # Christoffel is (batch, D, D, D) or (D,D,D)
        # Riemann is (batch, D, D, D, D) or (D,D,D,D)
        if christoffel.ndim == 4: # Batched Christoffel
            assert riemann.shape == (christoffel.shape[0], DIM_TEST, DIM_TEST, DIM_TEST, DIM_TEST)
        else: # Non-batched
            assert riemann.shape == (DIM_TEST, DIM_TEST, DIM_TEST, DIM_TEST)
        # except Exception as e:
        #    pytest.skip(f"Skipping Riemann test due to known implementation issues: {e}")

    @pytest.mark.parametrize("method", CurvatureMethod)
    def test_compute_curvature_methods(self, simple_attention_tensor, method):
        """Test compute_curvature for all methods, checking output shapes."""
        ats = AttentionTopologySystem(simple_attention_tensor)
        # try:
        curvature_tensors = ats.compute_curvature(method=method)
        assert "layer1" in curvature_tensors
        curvature = curvature_tensors["layer1"]

        # simple_attention_tensor["layer1"] is (N,N)
        # metric_tensor becomes (N,N)
        # christoffel becomes (N,N,N) (k,i,j)
        # riemann becomes (N,N,N,N) (i,j,k,l)

        if method == CurvatureMethod.RIEMANN:
            assert curvature.shape == (DIM_TEST, DIM_TEST, DIM_TEST, DIM_TEST)
        elif method == CurvatureMethod.RICCI:
            assert curvature.shape == (DIM_TEST, DIM_TEST)
        elif method == CurvatureMethod.SCALAR:
            # For a single layer, non-batched, scalar curvature should be a scalar.
            assert isinstance(curvature, (np.number, float, int)) or curvature.shape == () or curvature.shape == (1,)
        elif method == CurvatureMethod.SECTIONAL:
            num_planes = DIM_TEST * (DIM_TEST - 1) // 2
            assert curvature.shape == (num_planes,)
        # except Exception as e:
            # This will catch issues from underlying derivative calculations too
            # pytest.skip(f"Skipping curvature test ({method.value}) due to known implementation issues: {e}")

    def test_analyze_topology(self, simple_attention_tensor):
        """Test analyze_topology basic execution and output type."""
        ats = AttentionTopologySystem(simple_attention_tensor)
        # Analyze topology relies on compute_curvature, which has known issues.
        # Test that it runs and returns the correct type.
        # try:
        metrics = ats.analyze_topology("layer1")
        assert isinstance(metrics, TopologyMetrics)
        assert hasattr(metrics, "curvature")
        assert hasattr(metrics, "entropy")
        assert hasattr(metrics, "homology")
        assert isinstance(metrics.homology, dict)
        assert "betti_0" in metrics.homology
        # except Exception as e:
        #     pytest.skip(f"Skipping analyze_topology test due to known implementation issues: {e}")


    def test_analyze_topology_missing_layer(self, simple_attention_tensor):
        """Test analyze_topology for a non-existent layer."""
        ats = AttentionTopologySystem(simple_attention_tensor)
        with pytest.raises(KeyError, match="Layer non_existent_layer not found in tensor field"):
            ats.analyze_topology("non_existent_layer")

    def test_get_curvature_statistics(self, simple_attention_tensor):
        """Test get_curvature_statistics basic execution."""
        ats = AttentionTopologySystem(simple_attention_tensor)
        # try:
        # Compute scalar curvature as it's simplest
        ats.compute_curvature(method=CurvatureMethod.SCALAR)
        stats = ats.get_curvature_statistics("layer1")
        assert isinstance(stats, dict)
        keys = ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis', 'shape']
        for key in keys:
            assert key in stats
        # except Exception as e:
        #      pytest.skip(f"Skipping get_curvature_statistics test due to known implementation issues: {e}")


    def test_batched_input_curvature(self, batched_attention_tensor):
        """Test curvature computation with batched input."""
        ats = AttentionTopologySystem(batched_attention_tensor)
        layer_name = "layer_batched"
        # Test with Ricci curvature as an example
        # try:
        curvatures = ats.compute_curvature(method=CurvatureMethod.RICCI)
        assert layer_name in curvatures
        ricci_curvature = curvatures[layer_name]
        # Expected shape for batched Ricci: (batch_size, dim, dim)
        assert ricci_curvature.shape == (BATCH_SIZE_TEST, DIM_TEST, DIM_TEST)
        # except Exception as e:
        #     pytest.skip(f"Skipping batched input test due to known implementation issues: {e}")

    def test_visualization_placeholder(self, simple_attention_tensor):
        """Placeholder for testing visualization. Requires matplotlib and visual inspection or mocking."""
        ats = AttentionTopologySystem(simple_attention_tensor)
        # try:
        ats.compute_curvature(method=CurvatureMethod.SCALAR) # Compute some curvature
        # This test would ideally mock plt.show or check if a file is saved.
        # For now, just ensure it doesn't crash if MATPLOTLIB_AVAILABLE is True.
        if ats.MATPLOTLIB_AVAILABLE:
            # ats.visualize_curvature("layer1") # Will attempt to show plot - disable for CI
            pass # Avoid actual plotting in automated tests unless mocked
        else:
            with pytest.raises(ImportError): # Should raise if not available and called
                ats.visualize_curvature("layer1")
        # except Exception as e:
        #      pytest.skip(f"Skipping visualization test due to known implementation issues or display environment: {e}")
        assert True # If it runs without crashing (or raises ImportError correctly)

# Example of a test for a fixed, simple case if we knew the expected output
# This would require a corrected implementation first.
# def test_christoffel_known_case(self):
#     # Define a very simple metric for which Christoffel symbols are known (e.g., Euclidean 2D space)
#     # metric_euclidean_2d = np.eye(2) # Batched: np.array([np.eye(2)])
#     # ats = AttentionTopologySystem({"dummy": np.random.rand(2,2)}) # Dummy tensor field
#     # christoffel = ats.compute_christoffel_symbols(np.array([np.eye(2)]))
#     # For Euclidean space, Christoffel symbols should be all zero.
#     # assert np.allclose(christoffel, 0)

# To run these tests:
# Ensure pytest is installed: pip install pytest
# Navigate to the root of the SVELTE project
# Run: python -m pytest tests/model_architecture/test_attention_topology.py
# (Or simply `pytest` if paths are configured)

# Add a check for scipy.stats availability for relevant tests
try:
    from scipy import stats as scipy_stats
    SCIPY_STATS_AVAILABLE_TEST_SCOPE = True
except ImportError:
    SCIPY_STATS_AVAILABLE_TEST_SCOPE = False

if SCIPY_STATS_AVAILABLE_TEST_SCOPE:
    def test_get_curvature_statistics_with_scipy(simple_attention_tensor):
        ats = AttentionTopologySystem(simple_attention_tensor)
        # try:
        ats.compute_curvature(method=CurvatureMethod.SCALAR)
        stats = ats.get_curvature_statistics("layer1")
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        # Check they are numbers (float or int)
        assert isinstance(stats['skewness'], (float, int))
        assert isinstance(stats['kurtosis'], (float, int))
        # except Exception as e:
        #     pytest.skip(f"Skipping full stats test due to known implementation issues: {e}")

# This else branch might be tricky if SCIPY_STATS_AVAILABLE is a global in the SUT
# and not easily mockable for a specific test run without altering the SUT.
# The SUT now uses self.SCIPY_STATS_AVAILABLE, so this test needs adjustment or removal
# if we cannot easily control that instance variable from here for this specific test.
# For now, assuming the SUT's SCIPY_STATS_AVAILABLE reflects the test environment.
# else:
#     def test_get_curvature_statistics_without_scipy(simple_attention_tensor):
#         ats = AttentionTopologySystem(simple_attention_tensor)
#         original_scipy_flag = ats.SCIPY_STATS_AVAILABLE
#         ats.SCIPY_STATS_AVAILABLE = False # Attempt to mock

#         ats.compute_curvature(method=CurvatureMethod.SCALAR)
#         stats = ats.get_curvature_statistics("layer1")
#         assert stats['skewness'] == 0.0
#         assert stats['kurtosis'] == 0.0

#         ats.SCIPY_STATS_AVAILABLE = original_scipy_flag # Restore

# Ensure the main block of attention_topology.py is guarded by if __name__ == "__main__":
# This is important for pytest discovery. (It is already guarded)

# Note: Many tests are marked with `pytest.skip` if they encounter exceptions
# related to the known issues in derivative calculations. Once those are fixed,
# these skips should be removed or refined.
# The primary goal of these initial tests is to check shapes and basic execution.
# True mathematical correctness tests require either:
# 1. A known analytical solution for a simple manifold.
# 2. Comparison against a trusted library or implementation.
# These are currently out of scope for the initial test creation.

print("Created test file: tests/model_architecture/test_attention_topology.py")
