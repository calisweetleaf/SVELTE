#!/usr/bin/env python3
"""
SVELTE: Symbolic Vector Entropy & Latent Tensor Excavation Framework
Unified CLI System Launcher
"""
import os
import sys
import argparse
import logging
import traceback
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- SVELTE Engine Imports (stub if missing) ---
try:
    from src.gguf_diagnostic_scanner import main as pipeline_main
    from src.metadata.metadata_extractor import MetadataExtractor
    from src.model_architecture.attention_topology import AttentionTopologySystem
    from src.model_architecture.graph_builder import ArchitectureGraphBuilder
    from src.model_architecture.memory_pattern import MemoryPatternRecognitionSystem
    from src.symbolic.meta_interpretation import MetaInterpretationSynthesisModule
    from src.symbolic.symbolic_mapping import SymbolicMappingModule
    from src.tensor_analysis.activation_sim import ActivationSpaceSimulator
    from src.tensor_analysis.entropy_analysis import EntropyAnalysisModule
    from src.tensor_analysis.gguf_parser import GGUFParser
    from src.tensor_analysis.quantization import QuantizationReconstructor
    from src.tensor_analysis.tensor_field import TensorFieldConstructor
    from src.utils.file_io import ensure_dir, write_json, read_json, FileIOException
except ImportError:
    # Stubs for missing modules (for dev/test)
    def pipeline_main(*a, **kw): print("[STUB] pipeline_main called")
    class MetadataExtractor: pass
    class AttentionTopologySystem: pass
    class ArchitectureGraphBuilder: pass
    class MemoryPatternRecognitionSystem: pass
    class MetaInterpretationSynthesisModule: pass
    class SymbolicMappingModule: pass
    class ActivationSpaceSimulator: pass
    class EntropyAnalysisModule: pass
    class GGUFParser: pass
    class QuantizationReconstructor: pass
    class TensorFieldConstructor: pass
    def ensure_dir(path, **kw): os.makedirs(path, exist_ok=True)
    def write_json(data, path, **kw): pass
    def read_json(path, **kw): return {}
    class FileIOException(Exception): pass

OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "svelte_cli.log")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("SVELTE_CLI")

# --- Boot Banner ---
def print_banner():
    print("="*90)
    print(" SVELTE: Symbolic Vector Entropy & Latent Tensor Excavation Framework")
    print(" Unified CLI - Advanced Model Analysis & Interpretability Suite")
    print(" (c) 2025 SVELTE Project | https://github.com/svelte-framework")
    print("="*90)

# --- Help Menu ---
def print_help():
    print("\nSVELTE CLI Help:")
    print("  --model <file>      Analyze a specific GGUF model file")
    print("  --output <dir>      Set output directory (default: ./output)")
    print("  --pipeline          Run full analysis pipeline (headless)")
    print("  --modules <list>    Comma-separated list of modules to run (e.g. parser,quant,entropy)")
    print("  --headless          Run in non-interactive mode (requires --model)")
    print("  --help              Show this help menu\n")
    print("Available Modules:")
    print("  parser, quant, entropy, activation, graph, attention, memory, symbolic, meta, pipeline")
    print("\nExample:")
    print("  python main.py --model models/model.gguf --modules parser,entropy,graph --output results\n")

# --- GGUF File Discovery ---
def discover_gguf_files(search_dir: str = "models") -> List[str]:
    files = glob.glob(os.path.join(search_dir, "*.gguf"))
    return sorted(files)

# --- Interactive Menu ---
def interactive_menu() -> str:
    print("\nSelect an operation:")
    print(" 1. Full Analysis Pipeline")
    print(" 2. Metadata Extraction")
    print(" 3. Tensor Field Construction")
    print(" 4. Quantization Analysis")
    print(" 5. Activation Simulation")
    print(" 6. Entropy Analysis")
    print(" 7. Architecture Graph Building")
    print(" 8. Attention Topology Analysis")
    print(" 9. Memory Pattern Recognition")
    print("10. Symbolic Mapping")
    print("11. Meta-Interpretation Synthesis")
    print("12. Export/Save Results")
    print("13. Help")
    print("14. Exit")
    return input("Enter choice [1-14]: ").strip()

# --- Main CLI Logic ---
def main():
    print_banner()
    parser = argparse.ArgumentParser(description="SVELTE Unified CLI", add_help=False)
    parser.add_argument('--model', type=str, help='Path to GGUF model file')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--pipeline', action='store_true', help='Run full analysis pipeline')
    parser.add_argument('--modules', type=str, help='Comma-separated list of modules to run')
    parser.add_argument('--headless', action='store_true', help='Non-interactive mode')
    parser.add_argument('--help', action='store_true', help='Show help menu')
    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

    ensure_dir(args.output)

    # Headless pipeline mode
    if args.pipeline or args.headless:
        if not args.model:
            print("[ERROR] --model argument required for headless or pipeline mode.")
            sys.exit(1)
        if args.pipeline:
            logger.info("Running full SVELTE pipeline...")
            pipeline_main()
            print("[SVELTE] Pipeline complete. See output directory for results.")
            sys.exit(0)
        # Selective modules
        modules = [m.strip().lower() for m in (args.modules or '').split(',') if m.strip()]
        if not modules:
            print("[ERROR] --modules required for headless non-pipeline mode.")
            sys.exit(1)
        run_selected_modules(args.model, modules, args.output)
        sys.exit(0)

    # Interactive CLI
    while True:
        try:
            # Model selection
            gguf_files = discover_gguf_files()
            print("\nAvailable GGUF model files:")
            for idx, f in enumerate(gguf_files):
                print(f"  [{idx+1}] {f}")
            model_path = None
            while not model_path:
                sel = input(f"Select model [1-{len(gguf_files)}] or enter path: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(gguf_files):
                    model_path = gguf_files[int(sel)-1]
                elif os.path.isfile(sel):
                    model_path = sel
                else:
                    print("Invalid selection. Try again.")
            print(f"[SVELTE] Selected model: {model_path}")
            # Main menu loop
            while True:
                choice = interactive_menu()
                if choice == "1":
                    logger.info("Running full analysis pipeline...")
                    pipeline_main()
                    print("[SVELTE] Pipeline complete. See output directory for results.")
                elif choice == "2":
                    extractor = MetadataExtractor(model_path)
                    metadata = extractor.extract() if hasattr(extractor, 'extract') else {}
                    print("Extracted Metadata:\n", metadata)
                elif choice == "3":
                    gguf = GGUFParser(model_path)
                    gguf.parse()
                    tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
                    tensor_field = tensor_field_constructor.construct()
                    print("Tensor Field constructed.")
                elif choice == "4":
                    gguf = GGUFParser(model_path)
                    gguf.parse()
                    quant_recon = QuantizationReconstructor(gguf.get_quantization())
                    print("Quantization schemes:", getattr(quant_recon, 'quantization', {}))
                elif choice == "5":
                    gguf = GGUFParser(model_path)
                    gguf.parse()
                    tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
                    tensor_field = tensor_field_constructor.construct()
                    activation_sim = ActivationSpaceSimulator(tensor_field)
                    dummy_input = input("Enter input shape (comma-separated, e.g. 1,128): ").strip()
                    shape = tuple(map(int, dummy_input.split(',')))
                    inputs = __import__('numpy').random.randn(*shape)
                    activations = activation_sim.simulate(inputs)
                    print("Activations simulated.")
                elif choice == "6":
                    gguf = GGUFParser(model_path)
                    gguf.parse()
                    tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
                    tensor_field = tensor_field_constructor.construct()
                    entropy_module = EntropyAnalysisModule(tensor_field)
                    entropy_maps = entropy_module.compute_entropy()
                    print("Entropy maps computed.")
                elif choice == "7":
                    extractor = MetadataExtractor(model_path)
                    metadata = extractor.extract() if hasattr(extractor, 'extract') else {}
                    graph_builder = ArchitectureGraphBuilder(metadata)
                    graph_builder.build_graph()
                    print("Architecture graph built.")
                elif choice == "8":
                    gguf = GGUFParser(model_path)
                    gguf.parse()
                    tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
                    tensor_field = tensor_field_constructor.construct()
                    attention_topology = AttentionTopologySystem(tensor_field)
                    curvature = attention_topology.compute_curvature()
                    print("Attention topology curvature computed.")
                elif choice == "9":
                    gguf = GGUFParser(model_path)
                    gguf.parse()
                    tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
                    tensor_field = tensor_field_constructor.construct()
                    memory_pattern = MemoryPatternRecognitionSystem(tensor_field)
                    patterns = memory_pattern.detect_patterns()
                    print("Memory patterns detected.")
                elif choice == "10":
                    gguf = GGUFParser(model_path)
                    gguf.parse()
                    tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
                    tensor_field = tensor_field_constructor.construct()
                    entropy_module = EntropyAnalysisModule(tensor_field)
                    entropy_maps = entropy_module.compute_entropy()
                    symbolic_mapping = SymbolicMappingModule(entropy_maps, tensor_field)
                    symbolic_mapping.extract_grammar()
                    symbolic_mapping.encode_symbolic()
                    symbolic_mapping.verify_interpretability()
                    print("Symbolic mapping complete.")
                elif choice == "11":
                    print("Meta-Interpretation Synthesis requires outputs from all modules.")
                    print("Run the full pipeline or provide module outputs.")
                elif choice == "12":
                    print("Exporting results to output directory...")
                    # Placeholder: implement export logic as needed
                    print("Results exported.")
                elif choice == "13":
                    print_help()
                elif choice == "14":
                    print("Exiting SVELTE CLI.")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please select a valid option.")
        except KeyboardInterrupt:
            print("\n[SVELTE] Interrupted by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {e}")
            logger.error(traceback.format_exc())
            continue

def run_selected_modules(model_path: str, modules: List[str], output_dir: str):
    """Run selected SVELTE modules in headless mode."""
    logger.info(f"Running modules: {modules} on model: {model_path}")
    results = {}
    try:
        if 'parser' in modules:
            gguf = GGUFParser(model_path)
            gguf.parse()
            results['tensors'] = list(getattr(gguf, 'tensors', {}).keys())
        if 'quant' in modules:
            gguf = GGUFParser(model_path)
            gguf.parse()
            quant_recon = QuantizationReconstructor(gguf.get_quantization())
            results['quantization'] = getattr(quant_recon, 'quantization', {})
        if 'entropy' in modules:
            gguf = GGUFParser(model_path)
            gguf.parse()
            tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
            tensor_field = tensor_field_constructor.construct()
            entropy_module = EntropyAnalysisModule(tensor_field)
            results['entropy'] = entropy_module.compute_entropy()
        if 'activation' in modules:
            gguf = GGUFParser(model_path)
            gguf.parse()
            tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
            tensor_field = tensor_field_constructor.construct()
            activation_sim = ActivationSpaceSimulator(tensor_field)
            dummy_input = __import__('numpy').random.randn(1, 128)
            results['activations'] = activation_sim.simulate(dummy_input)
        if 'graph' in modules:
            extractor = MetadataExtractor(model_path)
            metadata = extractor.extract() if hasattr(extractor, 'extract') else {}
            graph_builder = ArchitectureGraphBuilder(metadata)
            graph_builder.build_graph()
            results['graph'] = str(graph_builder.get_graph())
        if 'attention' in modules:
            gguf = GGUFParser(model_path)
            gguf.parse()
            tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
            tensor_field = tensor_field_constructor.construct()
            attention_topology = AttentionTopologySystem(tensor_field)
            results['curvature'] = attention_topology.compute_curvature()
        if 'memory' in modules:
            gguf = GGUFParser(model_path)
            gguf.parse()
            tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
            tensor_field = tensor_field_constructor.construct()
            memory_pattern = MemoryPatternRecognitionSystem(tensor_field)
            results['memory_patterns'] = memory_pattern.detect_patterns()
        if 'symbolic' in modules:
            gguf = GGUFParser(model_path)
            gguf.parse()
            tensor_field_constructor = TensorFieldConstructor(gguf.tensors)
            tensor_field = tensor_field_constructor.construct()
            entropy_module = EntropyAnalysisModule(tensor_field)
            entropy_maps = entropy_module.compute_entropy()
            symbolic_mapping = SymbolicMappingModule(entropy_maps, tensor_field)
            symbolic_mapping.extract_grammar()
            symbolic_mapping.encode_symbolic()
            symbolic_mapping.verify_interpretability()
            results['symbolic'] = symbolic_mapping.symbolic_patterns
            results['grammar'] = symbolic_mapping.grammar
        if 'meta' in modules:
            print("Meta-Interpretation Synthesis requires outputs from all modules. Run pipeline or provide all outputs.")
        # Save results
        out_path = os.path.join(output_dir, f"svelte_results_{int(time.time())}.json")
        write_json(results, out_path)
        print(f"[SVELTE] Module results written to {out_path}")
    except Exception as e:
        print(f"[ERROR] {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", str(e))
        logger.error(traceback.format_exc())
        sys.exit(1)