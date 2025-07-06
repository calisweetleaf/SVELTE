# src/cognitive_cartography/visualization_engine.py
"""
Visualization Engine for SVELTE Framework.
Generates multi-dimensional visualizations of tensor fields, entropy maps, and symbolic patterns.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

# Visualization dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.manifold import TSNE, UMAP
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class VisualizationError(Exception):
    """Exception raised for visualization errors."""
    pass

class ChartType(Enum):
    """Types of supported visualizations."""
    HEATMAP = "heatmap"
    SCATTER_3D = "scatter_3d"
    SURFACE = "surface"
    NETWORK = "network"
    STREAMLINES = "streamlines"
    CONTOUR = "contour"
    VOLUME = "volume"
    PARALLEL_COORDINATES = "parallel_coordinates"
    DENDOGRAM = "dendogram"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    RADAR = "radar"

class ColorScheme(Enum):
    """Color schemes for visualizations."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    TURBO = "turbo"
    RAINBOW = "rainbow"
    COOLWARM = "coolwarm"
    SPECTRAL = "spectral"
    CUSTOM = "custom"

class InteractionMode(Enum):
    """Interaction modes for visualizations."""
    STATIC = "static"
    HOVER = "hover"
    CLICK = "click"
    BRUSH = "brush"
    ZOOM = "zoom"
    PAN = "pan"
    ROTATE = "rotate"
    ANIMATE = "animate"

@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    chart_type: ChartType
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    interaction_mode: InteractionMode = InteractionMode.HOVER
    width: int = 800
    height: int = 600
    title: str = ""
    show_colorbar: bool = True
    show_legend: bool = True
    background_color: str = "white"
    font_size: int = 12
    opacity: float = 1.0
    animation_duration: int = 500
    custom_colorscale: Optional[List[Tuple[float, str]]] = None
    dimension_reduction: Optional[str] = None
    clustering: bool = False
    annotations: List[Dict[str, Any]] = field(default_factory=list)

class VisualizationRenderer:
    """Base class for visualization renderers."""
    
    def __init__(self, engine: str = "plotly"):
        """Initialize renderer with specified engine."""
        self.engine = engine
        self._validate_engine()
        
    def _validate_engine(self):
        """Validate that the rendering engine is available."""
        if self.engine == "plotly" and not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly not available. Install with: pip install plotly")
        elif self.engine == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise VisualizationError("Matplotlib not available. Install with: pip install matplotlib")
    
    def render(self, data: Dict[str, Any], config: VisualizationConfig) -> Any:
        """Render visualization with given data and configuration."""
        raise NotImplementedError("Subclasses must implement render method")

class PlotlyRenderer(VisualizationRenderer):
    """Plotly-based visualization renderer."""
    
    def __init__(self):
        super().__init__("plotly")
    
    def render(self, data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Render visualization using Plotly."""
        if config.chart_type == ChartType.HEATMAP:
            return self._render_heatmap(data, config)
        elif config.chart_type == ChartType.SCATTER_3D:
            return self._render_scatter_3d(data, config)
        elif config.chart_type == ChartType.SURFACE:
            return self._render_surface(data, config)
        elif config.chart_type == ChartType.NETWORK:
            return self._render_network(data, config)
        elif config.chart_type == ChartType.CONTOUR:
            return self._render_contour(data, config)
        elif config.chart_type == ChartType.PARALLEL_COORDINATES:
            return self._render_parallel_coordinates(data, config)
        elif config.chart_type == ChartType.SANKEY:
            return self._render_sankey(data, config)
        else:
            raise VisualizationError(f"Unsupported chart type: {config.chart_type}")
    
    def _render_heatmap(self, data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Render heatmap visualization."""
        z_data = data.get("z", np.random.rand(10, 10))
        x_labels = data.get("x", None)
        y_labels = data.get("y", None)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale=config.color_scheme.value,
            showscale=config.show_colorbar,
            hoverongaps=False
        ))
        
        self._apply_layout(fig, config)
        return fig
    
    def _render_scatter_3d(self, data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Render 3D scatter plot."""
        x = data.get("x", np.random.rand(100))
        y = data.get("y", np.random.rand(100))
        z = data.get("z", np.random.rand(100))
        colors = data.get("colors", x)
        text = data.get("text", None)
        
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale=config.color_scheme.value,
                showscale=config.show_colorbar,
                opacity=config.opacity
            ),
            text=text,
            hovertemplate="<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>"
        ))
        
        self._apply_layout_3d(fig, config)
        return fig
    
    def _render_surface(self, data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Render 3D surface plot."""
        z_data = data.get("z", np.random.rand(20, 20))
        x_data = data.get("x", None)
        y_data = data.get("y", None)
        
        fig = go.Figure(data=go.Surface(
            z=z_data,
            x=x_data,
            y=y_data,
            colorscale=config.color_scheme.value,
            showscale=config.show_colorbar,
            opacity=config.opacity
        ))
        
        self._apply_layout_3d(fig, config)
        return fig
    
    def _render_network(self, data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Render network graph visualization."""
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        # Extract node positions
        node_x = [node.get("x", 0) for node in nodes]
        node_y = [node.get("y", 0) for node in nodes]
        node_text = [node.get("text", f"Node {i}") for i, node in enumerate(nodes)]
        node_colors = [node.get("color", 0) for node in nodes]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = nodes[edge["source"]]["x"], nodes[edge["source"]]["y"]
            x1, y1 = nodes[edge["target"]]["x"], nodes[edge["target"]]["y"]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=2, color='gray'),
                               hoverinfo='none',
                               mode='lines')
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=node_text,
                               textposition="middle center",
                               marker=dict(
                                   size=15,
                                   color=node_colors,
                                   colorscale=config.color_scheme.value,
                                   showscale=config.show_colorbar,
                                   line=dict(width=2, color='black')
                               ))
        
        fig = go.Figure(data=[edge_trace, node_trace])
        self._apply_layout(fig, config)
        fig.update_layout(showlegend=False)
        return fig
    
    def _render_contour(self, data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Render contour plot."""
        z_data = data.get("z", np.random.rand(20, 20))
        x_data = data.get("x", None)
        y_data = data.get("y", None)
        
        fig = go.Figure(data=go.Contour(
            z=z_data,
            x=x_data,
            y=y_data,
            colorscale=config.color_scheme.value,
            showscale=config.show_colorbar,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))
        
        self._apply_layout(fig, config)
        return fig
    
    def _render_parallel_coordinates(self, data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Render parallel coordinates plot."""
        dimensions = data.get("dimensions", [])
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=data.get("colors", [0] * len(dimensions[0]["values"])),
                     colorscale=config.color_scheme.value,
                     showscale=config.show_colorbar),
            dimensions=dimensions
        ))
        
        self._apply_layout(fig, config)
        return fig
    
    def _render_sankey(self, data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Render Sankey diagram."""
        nodes = data.get("nodes", [])
        links = data.get("links", [])
        
        fig = go.Figure(data=go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color="blue"
            ),
            link=dict(
                source=links.get("source", []),
                target=links.get("target", []),
                value=links.get("value", []),
                color=links.get("color", ["rgba(0,0,255,0.4)"] * len(links.get("source", [])))
            )
        ))
        
        self._apply_layout(fig, config)
        return fig
    
    def _apply_layout(self, fig: go.Figure, config: VisualizationConfig):
        """Apply layout configuration to figure."""
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            paper_bgcolor=config.background_color,
            plot_bgcolor=config.background_color,
            font=dict(size=config.font_size),
            showlegend=config.show_legend
        )
        
        # Add annotations
        for annotation in config.annotations:
            fig.add_annotation(**annotation)
    
    def _apply_layout_3d(self, fig: go.Figure, config: VisualizationConfig):
        """Apply 3D layout configuration to figure."""
        self._apply_layout(fig, config)
        fig.update_layout(
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
                bgcolor=config.background_color
            )
        )

class MatplotlibRenderer(VisualizationRenderer):
    """Matplotlib-based visualization renderer."""
    
    def __init__(self):
        super().__init__("matplotlib")
    
    def render(self, data: Dict[str, Any], config: VisualizationConfig):
        """Render visualization using Matplotlib."""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
        
        if config.chart_type == ChartType.HEATMAP:
            return self._render_heatmap(data, config, fig, ax)
        elif config.chart_type == ChartType.SCATTER_3D:
            return self._render_scatter_3d(data, config, fig)
        else:
            raise VisualizationError(f"Matplotlib renderer does not support {config.chart_type}")
    
    def _render_heatmap(self, data: Dict[str, Any], config: VisualizationConfig, fig, ax):
        """Render heatmap using matplotlib."""
        z_data = data.get("z", np.random.rand(10, 10))
        
        im = ax.imshow(z_data, cmap=config.color_scheme.value, aspect='auto')
        
        if config.show_colorbar:
            plt.colorbar(im, ax=ax)
        
        ax.set_title(config.title, fontsize=config.font_size)
        return fig
    
    def _render_scatter_3d(self, data: Dict[str, Any], config: VisualizationConfig, fig):
        """Render 3D scatter plot using matplotlib."""
        ax = fig.add_subplot(111, projection='3d')
        
        x = data.get("x", np.random.rand(100))
        y = data.get("y", np.random.rand(100))
        z = data.get("z", np.random.rand(100))
        colors = data.get("colors", x)
        
        scatter = ax.scatter(x, y, z, c=colors, cmap=config.color_scheme.value, 
                           alpha=config.opacity)
        
        if config.show_colorbar:
            plt.colorbar(scatter, ax=ax)
        
        ax.set_title(config.title, fontsize=config.font_size)
        return fig

class VisualizationEngine:
    """
    Main visualization engine for SVELTE framework.
    
    Coordinates multiple rendering backends and provides high-level
    visualization generation for tensor fields, entropy maps, and symbolic patterns.
    """
    
    def __init__(self, default_renderer: str = "plotly", cache_dir: Optional[str] = None):
        """Initialize visualization engine."""
        self.default_renderer = default_renderer
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.renderers = {}
        self.cache = {}
        self._setup_renderers()
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VisualizationEngine initialized with {default_renderer} renderer")
    
    def _setup_renderers(self):
        """Setup available renderers."""
        if PLOTLY_AVAILABLE:
            self.renderers["plotly"] = PlotlyRenderer()
        
        if MATPLOTLIB_AVAILABLE:
            self.renderers["matplotlib"] = MatplotlibRenderer()
        
        if not self.renderers:
            raise VisualizationError("No visualization backends available. Install plotly or matplotlib.")
    
    def create_visualization(self, data: Dict[str, Any], config: VisualizationConfig,
                           renderer: Optional[str] = None) -> Any:
        """Create a visualization with given data and configuration."""
        renderer_name = renderer or self.default_renderer
        
        if renderer_name not in self.renderers:
            raise VisualizationError(f"Renderer {renderer_name} not available")
        
        # Check cache
        cache_key = self._generate_cache_key(data, config, renderer_name)
        if cache_key in self.cache:
            logger.debug(f"Returning cached visualization: {cache_key}")
            return self.cache[cache_key]
        
        # Apply dimension reduction if requested
        if config.dimension_reduction and SKLEARN_AVAILABLE:
            data = self._apply_dimension_reduction(data, config.dimension_reduction)
        
        # Generate visualization
        start_time = time.time()
        viz = self.renderers[renderer_name].render(data, config)
        
        # Cache result
        self.cache[cache_key] = viz
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {config.chart_type.value} visualization in {generation_time:.2f}s")
        
        return viz
    
    def visualize_tensor_field(self, tensor_field: Dict[str, np.ndarray], 
                              config: Optional[VisualizationConfig] = None) -> Any:
        """Visualize tensor field data."""
        if not config:
            config = VisualizationConfig(chart_type=ChartType.HEATMAP)
        
        # Prepare tensor data for visualization
        data = self._prepare_tensor_data(tensor_field)
        
        return self.create_visualization(data, config)
    
    def visualize_entropy_map(self, entropy_maps: Dict[str, np.ndarray],
                             config: Optional[VisualizationConfig] = None) -> Any:
        """Visualize entropy map data."""
        if not config:
            config = VisualizationConfig(chart_type=ChartType.CONTOUR, color_scheme=ColorScheme.PLASMA)
        
        # Prepare entropy data for visualization
        data = self._prepare_entropy_data(entropy_maps)
        
        return self.create_visualization(data, config)
    
    def visualize_attention_topology(self, curvature_data: Dict[str, np.ndarray],
                                   config: Optional[VisualizationConfig] = None) -> Any:
        """Visualize attention topology and curvature."""
        if not config:
            config = VisualizationConfig(chart_type=ChartType.SURFACE, color_scheme=ColorScheme.VIRIDIS)
        
        # Prepare curvature data for visualization
        data = self._prepare_curvature_data(curvature_data)
        
        return self.create_visualization(data, config)
    
    def visualize_symbolic_patterns(self, symbolic_data: Dict[str, Any],
                                   config: Optional[VisualizationConfig] = None) -> Any:
        """Visualize symbolic patterns and relationships."""
        if not config:
            config = VisualizationConfig(chart_type=ChartType.NETWORK, color_scheme=ColorScheme.SPECTRAL)
        
        # Prepare symbolic data for visualization
        data = self._prepare_symbolic_data(symbolic_data)
        
        return self.create_visualization(data, config)
    
    def visualize_architecture_graph(self, graph_data: Dict[str, Any],
                                   config: Optional[VisualizationConfig] = None) -> Any:
        """Visualize model architecture graph."""
        if not config:
            config = VisualizationConfig(chart_type=ChartType.NETWORK, color_scheme=ColorScheme.TURBO)
        
        # Prepare graph data for visualization
        data = self._prepare_graph_data(graph_data)
        
        return self.create_visualization(data, config)
    
    def create_dashboard(self, datasets: Dict[str, Dict[str, Any]], 
                        layout: str = "grid") -> Any:
        """Create multi-panel dashboard visualization."""
        if self.default_renderer != "plotly":
            raise VisualizationError("Dashboard creation requires Plotly renderer")
        
        # Create subplot structure
        n_plots = len(datasets)
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
        
        subplot_titles = list(datasets.keys())
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Add individual visualizations
        for i, (name, dataset) in enumerate(datasets.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            data = dataset["data"]
            chart_type = dataset.get("chart_type", ChartType.HEATMAP)
            
            if chart_type == ChartType.HEATMAP:
                trace = go.Heatmap(z=data.get("z"), showscale=False)
            elif chart_type == ChartType.SCATTER_3D:
                trace = go.Scatter(x=data.get("x"), y=data.get("y"), mode='markers')
            else:
                trace = go.Scatter(x=range(len(data.get("y", []))), y=data.get("y", []))
            
            fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            height=600 * rows,
            title_text="SVELTE Analysis Dashboard"
        )
        
        return fig
    
    def _prepare_tensor_data(self, tensor_field: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Prepare tensor data for visualization."""
        # For multiple tensors, create a combined visualization
        if len(tensor_field) == 1:
            tensor = next(iter(tensor_field.values()))
            if tensor.ndim == 2:
                return {"z": tensor}
            elif tensor.ndim == 1:
                return {"y": tensor, "x": range(len(tensor))}
        
        # Multiple tensors - create comparison data
        tensor_names = list(tensor_field.keys())
        tensor_values = []
        
        for name, tensor in tensor_field.items():
            if tensor.ndim == 1:
                tensor_values.append(tensor)
            else:
                tensor_values.append(tensor.flatten()[:100])  # Sample for visualization
        
        return {
            "dimensions": [
                {"label": name, "values": values.tolist()}
                for name, values in zip(tensor_names, tensor_values)
            ]
        }
    
    def _prepare_entropy_data(self, entropy_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Prepare entropy data for visualization."""
        if len(entropy_maps) == 1:
            entropy = next(iter(entropy_maps.values()))
            if entropy.ndim == 2:
                return {"z": entropy}
            else:
                # Create 2D representation
                size = int(np.sqrt(len(entropy)))
                if size * size == len(entropy):
                    return {"z": entropy.reshape(size, size)}
                else:
                    return {"y": entropy, "x": range(len(entropy))}
        
        # Multiple entropy maps
        combined_data = np.stack([emap for emap in entropy_maps.values()])
        return {"z": np.mean(combined_data, axis=0)}
    
    def _prepare_curvature_data(self, curvature_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Prepare curvature data for visualization."""
        # Extract a representative curvature tensor
        curvature = next(iter(curvature_data.values()))
        
        if curvature.ndim >= 2:
            # Take 2D slice for surface plot
            return {"z": curvature[:20, :20] if curvature.shape[0] > 20 else curvature}
        else:
            # 1D curvature - create artificial 2D surface
            x = np.linspace(0, 1, len(curvature))
            y = np.linspace(0, 1, 10)
            X, Y = np.meshgrid(x, y)
            Z = np.tile(curvature, (10, 1))
            return {"x": X, "y": Y, "z": Z}
    
    def _prepare_symbolic_data(self, symbolic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare symbolic data for network visualization."""
        # Extract pattern relationships for network visualization
        patterns = symbolic_data.get("patterns", {})
        
        nodes = []
        edges = []
        
        # Create nodes from patterns
        for i, (pattern_id, pattern_info) in enumerate(patterns.items()):
            nodes.append({
                "x": np.random.uniform(-1, 1),
                "y": np.random.uniform(-1, 1),
                "text": pattern_id,
                "color": i
            })
        
        # Create edges from relationships
        for i in range(len(nodes)):
            for j in range(i + 1, min(i + 3, len(nodes))):  # Connect to nearby nodes
                edges.append({"source": i, "target": j})
        
        return {"nodes": nodes, "edges": edges}
    
    def _prepare_graph_data(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare architecture graph data for visualization."""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Position nodes using force-directed layout
        n_nodes = len(nodes)
        if n_nodes == 0:
            return {"nodes": [], "edges": []}
        
        # Simple circular layout
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = max(1, n_nodes / 10)
        
        positioned_nodes = []
        for i, node in enumerate(nodes):
            positioned_nodes.append({
                "x": radius * np.cos(angles[i]),
                "y": radius * np.sin(angles[i]),
                "text": node.get("name", f"Node {i}"),
                "color": i
            })
        
        return {"nodes": positioned_nodes, "edges": edges}
    
    def _apply_dimension_reduction(self, data: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Apply dimension reduction to high-dimensional data."""
        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "umap":
            reducer = UMAP(n_components=2, random_state=42)
        else:
            logger.warning(f"Unknown dimension reduction method: {method}")
            return data
        
        # Apply reduction to appropriate data
        if "z" in data and data["z"].ndim == 2:
            flattened = data["z"].reshape(-1, data["z"].shape[1])
            reduced = reducer.fit_transform(flattened)
            data["x"] = reduced[:, 0]
            data["y"] = reduced[:, 1]
            data.pop("z")  # Remove original high-dimensional data
        
        return data
    
    def _generate_cache_key(self, data: Dict[str, Any], config: VisualizationConfig, 
                           renderer: str) -> str:
        """Generate cache key for visualization."""
        # Create hash from data and config
        data_str = json.dumps(data, sort_keys=True, default=str)
        config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
        combined = f"{data_str}_{config_str}_{renderer}"
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    def save_visualization(self, viz: Any, filepath: str, format: str = "html"):
        """Save visualization to file."""
        if self.default_renderer == "plotly" and hasattr(viz, 'write_html'):
            if format == "html":
                viz.write_html(filepath)
            elif format == "png":
                viz.write_image(filepath)
            elif format == "pdf":
                viz.write_image(filepath)
            else:
                raise VisualizationError(f"Unsupported format: {format}")
        
        elif self.default_renderer == "matplotlib":
            viz.savefig(filepath, format=format, dpi=300, bbox_inches='tight')
        
        logger.info(f"Visualization saved to {filepath}")
    
    def get_available_renderers(self) -> List[str]:
        """Get list of available renderers."""
        return list(self.renderers.keys())
    
    def clear_cache(self):
        """Clear visualization cache."""
        self.cache.clear()
        logger.info("Visualization cache cleared")

def main():
    """CLI entry point for visualization engine testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SVELTE Visualization Engine CLI")
    parser.add_argument('--test', action='store_true', help='Run test visualizations')
    parser.add_argument('--renderer', default='plotly', help='Renderer to use')
    parser.add_argument('--output', help='Output file path')
    args = parser.parse_args()
    
    if args.test:
        engine = VisualizationEngine(default_renderer=args.renderer)
        
        # Test heatmap
        test_data = {"z": np.random.rand(20, 20)}
        config = VisualizationConfig(chart_type=ChartType.HEATMAP, title="Test Heatmap")
        viz = engine.create_visualization(test_data, config)
        
        if args.output:
            engine.save_visualization(viz, args.output)
        
        print(f"Test visualization created with {args.renderer} renderer")

if __name__ == "__main__":
    main()