# src/symbolic/symbolic_mapping.py
"""
Symbolic Mapping Module for SVELTE Framework.
Transforms tensor patterns into symbolic representations and extracts computational grammars.
author: Morpheus
date: 2025-05-01
description: This module provides a system for transforming tensor patterns into symbolic representations and extracting computational grammars.
version: 0.1.3 # Updated version
ID: 002
SHA-256: abcdef1234567890abcdef1234567890abcdef123456 # Placeholder
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable # Consolidated typing imports
from collections import Counter, defaultdict
import logging
# argparse will be imported locally in main()

class SymbolicMappingModule:
    def __init__(self, entropy_maps: Dict[str, float], tensor_field: Dict[str, np.ndarray]):
        if not isinstance(entropy_maps, dict) or not all(isinstance(v, (float, int, np.number)) for v in entropy_maps.values()):
            raise ValueError("entropy_maps must be a dictionary mapping strings to numbers (float/int)")
        if not isinstance(tensor_field, dict) or not all(isinstance(v, np.ndarray) for v in tensor_field.values()):
            raise ValueError("tensor_field must be a dictionary mapping strings to numpy arrays")

        self.logger = logging.getLogger("SymbolicMappingModule")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"Initializing with {len(entropy_maps)} entropy maps and {len(tensor_field)} tensor fields")
        self.entropy_maps = entropy_maps
        self.tensor_field = tensor_field
        self.symbolic_patterns: Dict[str, List[str]] = {}
        self.grammar: Dict[str, Any] = {}
        self.abstraction_hierarchy: Dict[str, Dict[str, Any]] = {
            "level_0": {"symbols": [], "relations": {}}, "level_1": {"symbols": [], "relations": {}}, "level_2": {"symbols": [], "relations": {}}}
        self.interpretability_scores: Dict[str, Any] = {}
        self.processed_tensors: set[str] = set()
        self.processing_metadata: Dict[str, Any] = {
            "timestamp": np.datetime64('now'),
            "entropy_range": (min(entropy_maps.values()) if entropy_maps else 0.0, max(entropy_maps.values()) if entropy_maps else 0.0),
            "tensor_dimensions": {k: v.shape for k, v in tensor_field.items()}}
        self.logger.debug(f"Initialization complete. Processing metadata: {self.processing_metadata}")

    def extract_grammar(self) -> Dict[str, Any]:
        if not self.symbolic_patterns:
            self.logger.error("Cannot extract grammar: Symbolic patterns are empty. Run encode_symbolic first.")
            raise ValueError("Symbolic patterns have not been generated. Run encode_symbolic first.")
        try:
            all_sequences: List[Tuple[str, ...]] = []
            for key, pattern_data in self.symbolic_patterns.items():
                if isinstance(pattern_data, (list, tuple)): all_sequences.append(tuple(str(s) for s in pattern_data))
                elif isinstance(pattern_data, str): all_sequences.append(tuple(pattern_data.split()))
                else: self.logger.warning(f"Unsupported pattern_data type for key {key}: {type(pattern_data)}")

            if not all_sequences:
                self.logger.warning("No valid sequences for grammar extraction.")
                self.grammar = {"terminals": [], "non_terminals": ["S"], "production_rules": {"S": [[]]}}
                return self.grammar

            ngram_counts: Counter[Tuple[str, ...]] = Counter()
            terminals: set[str] = set()
            for seq in all_sequences:
                terminals.update(seq)
                for n in [3, 2]:
                    for i in range(len(seq) - n + 1): ngram_counts[tuple(seq[i:i+n])] += 1

            min_ngram_freq = max(2, int(0.05 * len(all_sequences))) if all_sequences else 2
            frequent_ngrams = {ngram for ngram, count in ngram_counts.items() if count >= min_ngram_freq and len(ngram) > 1}

            non_terminal_map: Dict[Tuple[str,...], str] = {}
            non_terminals_set: set[str] = set()
            nt_counter = 1
            for ngram in sorted(frequent_ngrams, key=lambda x: (-len(x), -ngram_counts[x])):
                is_sub_nt = False # Basic overlap check placeholder
                nt_symbol = f"N{nt_counter}"
                non_terminal_map[ngram] = nt_symbol
                non_terminals_set.add(nt_symbol)
                nt_counter += 1

            def replace_ngrams(seq_tuple: Tuple[str, ...]) -> Tuple[str, ...]:
                seq_list = list(seq_tuple)
                result_list = []
                processed_indices = [False] * len(seq_list)
                sorted_ngrams_to_replace = sorted(non_terminal_map.keys(), key=len, reverse=True)
                for k_ngram in range(len(seq_list)):
                    if processed_indices[k_ngram]: continue
                    replaced_this_step = False
                    for ngram_key in sorted_ngrams_to_replace:
                        n_len = len(ngram_key)
                        if k_ngram + n_len <= len(seq_list) and not any(processed_indices[k_ngram : k_ngram + n_len]):
                            current_slice = tuple(seq_list[k_ngram : k_ngram + n_len])
                            if current_slice == ngram_key:
                                result_list.append(non_terminal_map[ngram_key])
                                for l_idx in range(n_len): processed_indices[k_ngram + l_idx] = True
                                replaced_this_step = True; break
                    if not replaced_this_step:
                        result_list.append(seq_list[k_ngram])
                        processed_indices[k_ngram] = True
                return tuple(result_list)

            transformed_sequences = [replace_ngrams(seq) for seq in all_sequences]
            production_rules: Dict[str, List[List[str]]] = defaultdict(list)
            for ngram, nt_symbol in non_terminal_map.items():
                if not production_rules[nt_symbol]: production_rules[nt_symbol].append(list(ngram))

            unique_transformed_sequences = sorted(list(set(s for s in transformed_sequences if s)))
            if unique_transformed_sequences:
                production_rules["S"] = [list(seq) for seq in unique_transformed_sequences]
            else:
                 production_rules["S"] = [[]] if not all_sequences else [[str(s) for s in all_sequences[0]]]
            final_non_terminals = sorted(list(non_terminals_set | ({"S"} if production_rules.get("S") else set())))
            final_terminals = sorted(list(terminals))
            self.grammar = {"terminals": final_terminals, "non_terminals": final_non_terminals, "production_rules": dict(production_rules)}
            self.logger.info(f"Grammar extraction complete. Rules: {sum(len(r_list) for r_list in self.grammar['production_rules'].values())}, NTs: {len(final_non_terminals)}, Ts: {len(final_terminals)}")
            return self.grammar
        except Exception as e:
            self.logger.exception("Unexpected error during grammar extraction.")
            raise RuntimeError(f"Grammar extraction failed: {e}")

    def encode_symbolic(self) -> Dict[str, List[str]]:
        self.logger.info("Beginning symbolic encoding process")
        self.symbolic_patterns.clear(); self.abstraction_hierarchy = {"level_0": {"symbols": [], "relations": {}}, "level_1": {"symbols": [], "relations": {}}, "level_2": {"symbols": [], "relations": {}}}
        symbol_vocabulary = {"periodic": ["α","β","γ","δ","ε"], "chaotic": ["Ω","Ψ","Φ","Θ","Λ"], "stable": ["A","B","C","D","E"], "transient": ["X","Y","Z","W","V"], "boundary": ["⊥","⊤","⊢","⊣","⊩"], "unknown": ["?","¿"]}
        try:
            tensor_characteristics = {}
            for name, data in self.tensor_field.items():
                self.logger.debug(f"Analyzing tensor: {name} with shape {data.shape}"); grad_mag_mean = 0.0
                if data.ndim > 0 and data.size > 1:
                    try:
                        float_data = data.astype(np.float64) if not np.issubdtype(data.dtype, np.floating) else data
                        grads = np.gradient(float_data)
                        if not isinstance(grads, list): grads = [grads]
                        sq_sum_grads = sum(g**2 for g in grads); grad_mag_mean = float(np.mean(np.sqrt(sq_sum_grads)))
                    except Exception as e: self.logger.warning(f"Grad for {name} ({data.shape}): {e}")
                chars = {"entropy": self.entropy_maps.get(name,0.5), "mean":np.mean(data) if data.size>0 else 0, "std":np.std(data) if data.size>0 else 0, "min":np.min(data) if data.size>0 else 0, "max":np.max(data) if data.size>0 else 0, "gradient_magnitude":grad_mag_mean, "periodicity":self._measure_periodicity(data), "sparsity":np.count_nonzero(data)/data.size if data.size>0 else 0}
                tensor_characteristics[name] = {k:float(v) for k,v in chars.items()}
            assignments = {}
            for name,c in tensor_characteristics.items():
                if c["periodicity"]>0.55: assignments[name]="periodic"
                elif c["entropy"]>0.75 and c["gradient_magnitude"]>0.4: assignments[name]="chaotic"
                elif c["std"]<0.15*(abs(c.get("mean",1.0))+1e-9) or c["sparsity"]<0.1: assignments[name]="stable"
                elif c["gradient_magnitude"]>0.4: assignments[name]="transient"
                else: assignments[name]="boundary"
            for name, p_type in assignments.items():
                data = self.tensor_field[name]; chars = tensor_characteristics[name]; vocab = symbol_vocabulary.get(p_type, symbol_vocabulary["unknown"])
                segments = self._segment_tensor(data)
                if not segments: self.symbolic_patterns[name]=[vocab[0]]; continue
                seq = []
                for seg in segments:
                    if seg.size==0: continue
                    intensity = np.mean(seg); min_v,max_v = chars["min"],chars["max"]
                    norm_i = 0.5;
                    if max_v > min_v: norm_i = np.clip((intensity-min_v)/(max_v-min_v),0,1)
                    idx = int(round(norm_i*(len(vocab)-1))); seq.append(vocab[idx])
                self.symbolic_patterns[name] = seq if seq else [vocab[0]]
                level = min(2,int(np.clip(chars["entropy"],0,1)*3)); self.abstraction_hierarchy[f"level_{level}"]["symbols"].append(name)
            for level_name, level_data in self.abstraction_hierarchy.items():
                sym_in_lvl = level_data["symbols"]; rels = level_data["relations"]
                for i,n1 in enumerate(sym_in_lvl):
                    for j in range(i+1,len(sym_in_lvl)):
                        n2=sym_in_lvl[j]
                        if n1 in self.symbolic_patterns and n2 in self.symbolic_patterns:
                            sim = self._calculate_symbol_similarity(self.symbolic_patterns[n1],self.symbolic_patterns[n2])
                            if sim > 0.5: rels[f"{n1}__{n2}"]={"type":"similar","strength":round(sim,3)}
            self.logger.info(f"Symbolic encoding complete. Patterns: {len(self.symbolic_patterns)}.")
            return self.symbolic_patterns
        except Exception as e: self.logger.exception("Error in encode_symbolic"); raise RuntimeError(f"Encode failed: {e}")

    def _measure_periodicity(self, tensor: np.ndarray) -> float:
        flat = tensor.flatten()
        if len(flat) < 4 or np.isclose(np.std(flat), 0.0): return 0.0
        flat_norm = (flat - np.mean(flat)) / (np.std(flat) + 1e-9)
        autocorr = np.correlate(flat_norm, flat_norm, mode='full')
        autocorr_lag0 = autocorr[len(flat_norm)-1]
        autocorr = autocorr[len(flat_norm):]
        if not np.isclose(autocorr_lag0, 0.0): autocorr = autocorr / autocorr_lag0
        else: return 0.0
        if autocorr.size < 3 : return 0.0
        peaks = [(i+1, autocorr[i]) for i in range(1,len(autocorr)-1) if autocorr[i]>autocorr[i-1] and autocorr[i]>autocorr[i+1] and autocorr[i]>0.25]
        if not peaks: return 0.0
        avg_peak_h = np.mean([p[1] for p in peaks])
        if len(peaks)<2: return float(avg_peak_h*0.5)
        peak_dist_reg = 1.0 - min(1.0, np.std(np.diff([p[0] for p in peaks])) / (np.mean(np.diff([p[0] for p in peaks])) + 1e-9))
        return float(np.clip((avg_peak_h*0.6) + (peak_dist_reg*0.4),0,1))

    def _segment_tensor(self, tensor: np.ndarray) -> List[np.ndarray]:
        if tensor.ndim == 0: return [np.array([tensor.item()])]
        if tensor.size == 0: return []
        if tensor.ndim == 1: return self._segment_1d_tensor(tensor)
        elif tensor.ndim == 2: return self._segment_2d_tensor(tensor)
        else:
            self.logger.debug(f"Segmenting high-dim tensor {tensor.shape} by slicing first axis.")
            segments: List[np.ndarray] = []
            num_slices = min(max(1,tensor.shape[0]//10 if tensor.shape[0]>=10 else 1),3)
            indices = np.linspace(0,tensor.shape[0]-1,num_slices,dtype=int)
            for i in indices: segments.extend(self._segment_tensor(tensor[i,...]))
            return segments if segments else ([tensor.flatten()] if tensor.size > 0 else [])

    def _segment_1d_tensor(self, tensor: np.ndarray) -> List[np.ndarray]:
        if tensor.size == 0: return []
        seg_size = max(1,min(len(tensor),max(3,len(tensor)//5 if len(tensor)>=15 else 3)))
        segments: List[np.ndarray] = []
        step = max(1, seg_size//2 if seg_size>1 else 1)
        for i in range(0,len(tensor)-seg_size+1,step): segments.append(tensor[i:i+seg_size])
        if not segments and tensor.size>0: segments.append(tensor)
        return segments

    def _segment_2d_tensor(self, tensor: np.ndarray) -> List[np.ndarray]:
        if tensor.size == 0: return []
        h,w = tensor.shape; nbh = max(1,min(3,h//3 if h>=3 else 1)); nbw = max(1,min(3,w//3 if w>=3 else 1))
        gh = max(1,h//nbh); gw = max(1,w//nbw); segments: List[np.ndarray] = []
        for i in range(nbh):
            rs=i*gh; re=(i+1)*gh if i<nbh-1 else h
            for j in range(nbw):
                cs=j*gw; ce=(j+1)*gw if j<nbw-1 else w
                seg = tensor[rs:re,cs:ce]
                if seg.size>0: segments.append(seg)
        return segments if segments else ([tensor] if tensor.size > 0 else [])

    def _calculate_symbol_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        s1=[str(s) for s in seq1]; s2=[str(s) for s in seq2]; m,n=len(s1),len(s2)
        if m==0 and n==0: return 1.0
        if m==0 or n==0: return 0.0
        dp=np.zeros((m+1,n+1),dtype=int)
        for i in range(m+1): dp[i,0]=i
        for j in range(n+1): dp[0,j]=j
        for r in range(1,m+1):
            for c in range(1,n+1):
                cost = 0 if s1[r-1]==s2[c-1] else 1
                dp[r,c] = min(dp[r-1,c]+1, dp[r,c-1]+1, dp[r-1,c-1]+cost)
        return 1.0-(dp[m,n]/max(m,n))

    def verify_interpretability(self) -> Dict[str, Any]:
        self.logger.warning("--- SUT: Entering verify_interpretability ---")
        self.logger.warning(f"--- SUT: Logger handlers: {self.logger.handlers} ---")
        self.logger.warning(f"--- SUT: Logger effective level: {self.logger.getEffectiveLevel()} (WARNING is {logging.WARNING}) ---")
        self.logger.warning(f"--- SUT: Logger isEnabledFor WARNING: {self.logger.isEnabledFor(logging.WARNING)} ---")
        if not self.symbolic_patterns:
            self.logger.error("No symbolic patterns to verify. Run encode_symbolic first.")
            raise ValueError("No symbolic patterns to verify. Run encode_symbolic first.")
        self.logger.warning(f"--- SUT DEBUG: self.grammar is: '{self.grammar}' (type: {type(self.grammar)}) ---")
        if not self.grammar:
            self.logger.warning("--- SUT DEBUG: Condition 'if not self.grammar' is TRUE. Emitting expected warning. ---")
            self.logger.warning("Grammar not yet extracted. Interpretability assessment will be limited.")
        else:
            self.logger.warning("--- SUT DEBUG: Condition 'if not self.grammar' is FALSE. NOT emitting expected warning. ---")
        metrics: Dict[str,Any]={"symbol_entropy":{},"grammar_complexity":None,"abstraction_coherence":{},"human_readability":{},"overall_score":0.0}
        try:
            entropies=[np.nan_to_num(-sum((c/len(s))*np.log2(c/len(s)) for c in Counter(s).values() if c>0)) / (np.log2(len(set(s))) if len(set(s))>1 else 1.0) for s in self.symbolic_patterns.values() if s]
            metrics["symbol_entropy"] = {k:np.nan_to_num(-sum((c/len(s))*np.log2(c/len(s)) for c in Counter(s).values() if c>0))/(np.log2(len(set(s))) if len(set(s))>1 else 1.0) for k,s in self.symbolic_patterns.items() if s}
            avg_sym_ent=np.mean(entropies) if entropies else 0.0
            gc_score=0.5
            if self.grammar and self.grammar.get("production_rules"):
                nr=sum(len(rl) for rl in self.grammar["production_rules"].values()); nnt=len(self.grammar["non_terminals"]); nt=len(self.grammar["terminals"])
                arl=np.mean([len(r) for rl in self.grammar["production_rules"].values() for rsl in rl for r in rsl if r]) if nr>0 else 0
                cs=(np.log1p(nr)/np.log1p(50)+np.log1p(nnt)/np.log1p(20)+np.log1p(arl)/np.log1p(5))/3; gc_score=1.0-np.clip(cs,0,1)
                metrics["grammar_complexity"]={"score":gc_score,"rules":nr,"non_terminals":nnt,"terminals":nt,"avg_rule_length":arl}
            else: metrics["grammar_complexity"]=None
            coherence_scores=[]
            for lvl,dat in self.abstraction_hierarchy.items():
                syms=dat["symbols"];rels=dat["relations"]
                if not syms or len(syms)<2: continue
                max_rels=(len(syms)*(len(syms)-1))/2.0; dens=len(rels)/max_rels if max_rels>0 else 0
                strens=[r.get("strength",0.0) for r in rels.values() if isinstance(r,dict)]; mstr=np.mean(strens) if strens else 0
                coh=(dens*0.5+mstr*0.5); metrics["abstraction_coherence"][lvl]=coh; coherence_scores.append(coh)
            avg_coh=np.mean(coherence_scores) if coherence_scores else 0.5
            read_scores=[]
            for name,seq in self.symbolic_patterns.items():
                if not seq: continue
                slen=len(seq); uniq_c=len(set(seq)); ur=uniq_c/slen if slen>0 else 0
                l_sc=np.clip(1.0-(max(0,slen-20)/80.0),0,1); u_sc=np.clip(1.0-abs(ur-0.5)/0.5,0,1); rep_sc=self._calculate_repetition_index(seq)
                read=(0.4*l_sc+0.3*u_sc+0.3*rep_sc); metrics["human_readability"][name]=read; read_scores.append(read)
            avg_read=np.mean(read_scores) if read_scores else 0.5
            final_gc_score = gc_score if metrics["grammar_complexity"] is not None else 0.5
            overall=(0.25*(1-avg_sym_ent)+0.25*final_gc_score+0.25*avg_coh+0.25*avg_read)
            metrics["overall_score"]=np.clip(overall,0,1); self.interpretability_scores=metrics
            self.logger.info(f"Interpretability verification complete. Score: {metrics['overall_score']:.3f}")
            return metrics
        except Exception as e: self.logger.exception("Error in verify_interpretability"); raise RuntimeError(f"Verify failed: {e}")

    def _calculate_repetition_index(self, sequence: List[str]) -> float:
        if len(sequence)<4: return 0.0
        ng_counts:Counter[Tuple[str,...]]=Counter(); total_ng_instances=0
        for n_val in [2,3]:
            if len(sequence)<n_val: continue
            num_ng_this_n=len(sequence)-n_val+1; total_ng_instances+=num_ng_this_n
            for i in range(num_ng_this_n): ng_counts[tuple(sequence[i:i+n_val])]+=1
        if total_ng_instances==0: return 0.0
        num_rep_occur=sum(c-1 for c in ng_counts.values() if c>1)
        return np.clip(num_rep_occur/total_ng_instances if total_ng_instances>0 else 0.0,0,1)

def main():
    import argparse; import json
    logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    logger=logging.getLogger("SymbolicMappingMain")
    parser=argparse.ArgumentParser(description="SVELTE Symbolic Mapping CLI")
    parser.add_argument('--tensor_file',type=str,help='Path to .npz for tensor_field')
    parser.add_argument('--entropy_file',type=str,help='Path to .json for entropy_maps')
    args=parser.parse_args()
    if args.tensor_file and args.entropy_file:
        try:
            logger.info(f"Loading tensors: {args.tensor_file}"); tf_data=np.load(args.tensor_file); tf={n:tf_data[n] for n in tf_data.files}; logger.info(f"Loaded {len(tf)} tensors.")
            logger.info(f"Loading entropy: {args.entropy_file}");_f=open(args.entropy_file,'r'); em=json.load(_f);_f.close(); logger.info(f"Loaded {len(em)} entries.")
        except Exception as e: logger.error(f"File load error: {e}. Using dummy."); tf={"dummy_T":np.random.rand(5,5)};em={"dummy_T":0.5}
    else:
        logger.info("Paths not given, using dummy data."); tf={"sin":np.sin(np.linspace(0,8*np.pi,100)),"rand":np.random.rand(50)*10,"stable":np.ones((10,10))*5,"chaotic":np.cumsum(np.random.randn(100))}
        em={"sin":0.1,"rand":0.95,"stable":0.01,"chaotic":0.7}; [em.setdefault(k,0.5) for k in tf]
    try:
        mapper=SymbolicMappingModule(em,tf); print("\n--- Encoding ---")
        pats=mapper.encode_symbolic()
        for n,s in pats.items():tf_s=str(mapper.tensor_field[n].shape) if n in mapper.tensor_field else "N/A";e_s=f"{mapper.entropy_maps.get(n,'N/A'):.2f}" if isinstance(mapper.entropy_maps.get(n),(float,int,np.number)) else "N/A";print(f"T:'{n}' (S:{tf_s},E:{e_s}): {' '.join(s[:10])}{'...'if len(s)>10 else ''}(len:{len(s)})")
        print("\n--- Grammar ---");gram=mapper.extract_grammar()
        if gram and gram.get("terminals"):
            print(f"  Terminals ({len(gram['terminals'])}): {gram.get('terminals', [])[:10]}{'...' if len(gram.get('terminals',[])) > 10 else ''}")
            print(f"  Non-Terminals ({len(gram['non_terminals'])}): {gram.get('non_terminals', [])}")
            rules_preview = list(gram.get('production_rules', {}).items())[:3]
            if rules_preview:
                 print(f"  Production Rules (sample of {sum(len(r_list) for r_list in gram.get('production_rules', {}).values())} total rules):")
                 for nt, rules_for_nt in rules_preview:
                     if rules_for_nt: # Ensure there's at least one rule list for this NT
                        print(f"    {nt} -> {' | '.join([' '.join(r) for r in rules_for_nt[:2]])}{'...' if len(rules_for_nt) > 2 else ''}")
            else:
                print("  No production rules generated.")
        else:
            print("  No grammar extracted.")
        print("\n--- Interpretability ---");interp=mapper.verify_interpretability();print(f" Overall Score: {interp.get('overall_score','N/A'):.3f}")
        if isinstance(interp.get('symbol_entropy',{}),dict)and interp['symbol_entropy']:
            print(f" Symbol Entropy(avg norm):{np.mean(list(interp['symbol_entropy'].values())):.3f}")
        if isinstance(interp.get('human_readability',{}),dict)and interp['human_readability']:print(f" Readability(avg):{np.mean(list(interp['human_readability'].values())):.3f}")
        if isinstance(interp.get('grammar_complexity'),dict):print(f" Grammar Score:{interp['grammar_complexity'].get('score','N/A'):.3f}")
    except Exception as e:logger.error(f"Error: {e}",exc_info=True)

if __name__=="__main__":main()
