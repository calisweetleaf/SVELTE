SVELTE Framework File Structure
==========================

┣ docs
┃ ┗ SVELTE_Framework.md
┣ models
┣ output
┣ src
┃ ┣ metadata
┃ ┃ ┗ metadata_extractor.py
┃ ┣ model_architecture
┃ ┃ ┣ attention_topology.py
┃ ┃ ┣ graph_builder.py
┃ ┃ ┗ memory_pattern.py
┃ ┣ symbolic
┃ ┃ ┣ meta_interpretation.py
┃ ┃ ┗ symbolic_mapping.py
┃ ┣ tensor_analysis
┃ ┃ ┣ activation_sim.py
┃ ┃ ┣ entropy_analysis.py
┃ ┃ ┣ gguf_parser.py
┃ ┃ ┣ quantization.py
┃ ┃ ┗ tensor_field.py
┃ ┣ utils
┃ ┃ ┗ file_io.py
┃ ┗ gguf_diagnostic_scanner.py
┣ tests
┃ ┣ test_activation_sim.py
┃ ┣ test_attention_topology.py
┃ ┣ test_entropy_analysis.py
┃ ┣ test_gguf_parser.py
┃ ┣ test_graph_builder.py
┃ ┣ test_memory_pattern.py
┃ ┣ test_meta_interpretation.py
┃ ┣ test_metadata_extractor.py
┃ ┣ test_quantization.py
┃ ┣ test_symbolic_mapping.py
┃ ┗ test_tensor_field.py
┗ main.py
