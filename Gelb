import os
import math
import json
import time
import warnings
import signal
import tempfile
import subprocess
import sys
import hashlib
import pickle
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import fastapi
from pydantic import BaseModel
import uvicorn
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict


@dataclass
class GelbConfig:
    vocab_size: int = 49152
    hidden_dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 128
    ffn_hidden_dim: int = 5460
    max_seq_len: int = 16384
    rope_theta: float = 1_000_000.0
    dropout: float = 0.1
    norm_eps: float = 1e-5
    tokenizer_name: str = "bigcode/starcoder2-3b"
    use_flash_attn: bool = True
    use_fp8: bool = False
    use_cuda_graphs: bool = False
    use_speculative_decoding: bool = False
    use_retrieval: bool = True
    retrieval_top_k: int = 2
    retrieval_dim: int = 512
    use_process_reward: bool = False

    @property
    def num_params(self) -> int:
        embed = self.vocab_size * self.hidden_dim
        attn_per_layer = (
            self.hidden_dim * self.n_heads * self.head_dim +
            2 * self.hidden_dim * self.n_kv_heads * self.head_dim +
            self.n_heads * self.head_dim * self.hidden_dim
        )
        ffn_per_layer = (
            self.hidden_dim * self.ffn_hidden_dim +
            self.hidden_dim * self.ffn_hidden_dim +
            self.ffn_hidden_dim * self.hidden_dim
        )
        total = embed + self.n_layers * (attn_per_layer + ffn_per_layer)
        return total

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: str = "cpu") -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[:dim // 2].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis = freqs_cis[None, :, None, :]
    x_out_complex = x_complex * freqs_cis
    x_out = torch.view_as_real(x_out_complex)
    return x_out.reshape(*x.shape).type_as(x)


def flash_attn_wrapper(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    else:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if causal:
            L = q.size(-2)
            mask = torch.triu(torch.full((L, L), float('-inf'), device=q.device), diagonal=1)
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Linear4bit(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.uint8, device=device),
            requires_grad=False
        )
        
        self.weight_scale = nn.Parameter(
            torch.empty((out_features, 1), dtype=torch.float16, device=device),
            requires_grad=False
        )
        self.weight_zero_point = nn.Parameter(
            torch.empty((out_features, 1), dtype=torch.float16, device=device),
            requires_grad=False
        )
        
        self.lora_A = nn.Parameter(
            torch.empty((in_features, 8), dtype=torch.float16, device=device),
            requires_grad=True
        )
        self.lora_B = nn.Parameter(
            torch.empty((8, out_features), dtype=torch.float16, device=device),
            requires_grad=True
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.float16, device=device),
                requires_grad=True
            )
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        weight_fp = torch.randn(self.out_features, self.in_features, dtype=torch.float16, device=self.device)
        self.weight_scale.data = weight_fp.abs().max(dim=-1, keepdim=True)[0] / 7.5
        weight_quant = torch.clamp(weight_fp / self.weight_scale, -8, 7).round().to(torch.int8)
        self.weight.data = weight_quant.to(torch.uint8)
        self.weight_zero_point.data = torch.zeros_like(self.weight_scale)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lora_A)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        weight_dequant = self.weight.to(torch.int8).to(torch.float16) * self.weight_scale + self.weight_zero_point
        output = F.linear(x, weight_dequant, None)
        lora_output = F.linear(x, self.lora_A.T, None)
        lora_output = F.linear(lora_output, self.lora_B.T, None)
        output = output + lora_output
        if self.bias is not None:
            output = output + self.bias
        return output


class GelbAttention(nn.Module):
    def __init__(self, config: GelbConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.q_per_kv = self.n_heads // self.n_kv_heads

        self.wq = Linear4bit(self.hidden_dim, self.n_heads * self.head_dim, bias=False)
        self.wk = Linear4bit(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear4bit(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = Linear4bit(self.n_heads * self.head_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        B, L, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(B, L, self.n_heads, self.head_dim)
        xk = xk.view(B, L, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, L, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis[:L])
        xk = apply_rotary_emb(xk, freqs_cis[:L])

        if self.q_per_kv > 1:
            xk = xk.unsqueeze(3).expand(B, L, self.n_kv_heads, self.q_per_kv, self.head_dim).flatten(2, 3)
            xv = xv.unsqueeze(3).expand(B, L, self.n_kv_heads, self.q_per_kv, self.head_dim).flatten(2, 3)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        attn_out = flash_attn_wrapper(xq, xk, xv, causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.dropout(self.wo(attn_out))


class GelbFeedForward(nn.Module):
    def __init__(self, config: GelbConfig):
        super().__init__()
        self.w1 = Linear4bit(config.hidden_dim, config.ffn_hidden_dim, bias=False)
        self.w2 = Linear4bit(config.ffn_hidden_dim, config.hidden_dim, bias=False)
        self.w3 = Linear4bit(config.hidden_dim, config.ffn_hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GelbBlock(nn.Module):
    def __init__(self, config: GelbConfig):
        super().__init__()
        self.attention = GelbAttention(config)
        self.feed_forward = GelbFeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        h = self.attention(self.attention_norm(x), freqs_cis)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class GelbModel(nn.Module):
    def __init__(self, config: GelbConfig):
        super().__init__()
        self.config = config
        self.embed = Linear4bit(config.vocab_size, config.hidden_dim, bias=False)
        self.layers = nn.ModuleList([GelbBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.lm_head = Linear4bit(config.hidden_dim, config.vocab_size, bias=False)
        self._tie_weights()
        self.apply(self._init_weights)

    def _tie_weights(self):
        pass

    def _init_weights(self, module):
        if isinstance(module, Linear4bit):
            if hasattr(module, 'lora_A'):
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.lora_A)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

    def _ensure_freqs_cis(self, L: int, device: torch.device):
        if not hasattr(self, '_freqs_cis') or self._freqs_cis is None or self._freqs_cis.shape[0] < L:
            self._freqs_cis = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_seq_len,
                theta=self.config.rope_theta,
                device=device
            )
        return self._freqs_cis

    def forward(self, tokens: Tensor) -> Dict[str, Any]:
        B, L = tokens.shape
        x = self.embed(tokens) * math.sqrt(self.config.hidden_dim)
        freqs_cis = self._ensure_freqs_cis(L, tokens.device)

        for layer in self.layers:
            x = layer(x, freqs_cis[:L])

        x = self.norm(x)
        logits = self.lm_head(x)
        return {"logits": logits}

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        eos_token_id: int = 0,
    ) -> Tensor:
        if input_ids.size(0) != 1:
            raise ValueError("Batched generation not supported")
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self(generated)
            next_token = _sample_next_token(outputs["logits"][:, -1, :], temperature, top_p)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break
        return generated


def _sample_next_token(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    logits = logits / max(temperature, 1e-8)
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    if probs.sum() <= 0:
        probs = torch.ones_like(probs) / probs.shape[-1]
    else:
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, 1)


class GelbTokenizer:
    def __init__(self, config: GelbConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        special_tokens = [
            "<|repo_start|>", "<|file_path|>", "<|func_name|>", "<|class_name|>",
            "<|lang_js|>", "<|endofmask|>", "<|retrieved_context|>", "<|endofcontext|>"
        ]
        existing = set(self.tokenizer.get_vocab().keys())
        to_add = [t for t in special_tokens if t not in existing]
        if to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        self.vocab_size = len(self.tokenizer)

    def encode(self, text: str, return_tensors: str = "pt") -> Tensor:
        return self.tokenizer(text, return_tensors=return_tensors).input_ids

    def decode(self, tokens: Tensor) -> str:
        if tokens.ndim > 1:
            tokens = tokens[0]
        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=False)


class CodeRetriever:
    def __init__(self, codebase_path: str, top_k: int = 2, cache_dir: str = ".gelb_cache"):
        self.codebase_path = Path(codebase_path)
        self.top_k = top_k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            token_pattern=r'\b\w[\w\.]*\w\b'
        )
        self.chunks = []
        self.embeddings = None
        self._load_or_build_index()

    def _parse_js_file(self, file_path: Path) -> List[Tuple[int, int, str]]:
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            content = "".join(lines)
            # Простое разделение по функциям/классам
            import re
            # Паттерны для поиска функций и классов
            patterns = [
                r'(function\s+\w+\s*\([^)]*\)\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})',
                r'(const\s+\w+\s*=\s*function\s*\([^)]*\s*\)\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})',
                r'(const\s+\w+\s*=\s*\(.*?\)\s*=>\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})',
                r'(class\s+\w+\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()
                    # Найдем строки для этого блока
                    start_line = content[:start_pos].count('\n') + 1
                    end_line = content[:end_pos].count('\n') + 1
                    chunk_text = match.group(0)
                    chunks.append((start_line, end_line, chunk_text.strip()))
        except Exception:
            # Если не удалось распарсить - разбиваем на блоки по 10 строк
            for i in range(0, len(lines), 10):
                chunk = "".join(lines[i:i+10]).strip()
                if chunk:
                    chunks.append((i+1, i+10, chunk))
        return chunks

    def _load_or_build_index(self):
        cache_file = self.cache_dir / "index.pkl"
        meta_file = self.cache_dir / "meta.json"
        code_hash = self._compute_codebase_hash()

        need_rebuild = True
        if meta_file.exists() and cache_file.exists():
            try:
                meta = json.load(open(meta_file))
                if meta.get("hash") == code_hash and meta.get("path") == str(self.codebase_path):
                    data = pickle.load(open(cache_file, 'rb'))
                    self.chunks = data["chunks"]
                    self.embeddings = data["embeddings"]
                    need_rebuild = False
            except:
                pass

        if need_rebuild:
            self.chunks = []
            texts = []
            for ext in ["*.js", "*.ts"]:
                for file_path in self.codebase_path.rglob(ext):
                    chunks = self._parse_js_file(file_path)
                    
                    for start, end, chunk in chunks:
                        rel_path = file_path.relative_to(self.codebase_path)
                        self.chunks.append((str(rel_path), start, end, chunk))
                        texts.append(chunk)
            
            if texts:
                self.embeddings = self.vectorizer.fit_transform(texts)
            
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, open(cache_file, 'wb'))
            json.dump({"hash": code_hash, "path": str(self.codebase_path)}, open(meta_file, 'w'))

    def _compute_codebase_hash(self) -> str:
        hash_obj = hashlib.md5()
        for file_path in sorted(self.codebase_path.rglob("*")):
            if file_path.is_file() and file_path.suffix in {".js", ".ts"}:
                try:
                    with open(file_path, 'rb') as f:
                        hash_obj.update(f.read())
                except:
                    pass
        return hash_obj.hexdigest()

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.chunks:
            return []
        
        query_vec = self.vectorizer.transform([query])
        scores = (self.embeddings * query_vec.T).toarray().flatten()
        top_indices = scores.argsort()[-self.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0.1:
                file, start, end, text = self.chunks[idx]
                results.append({
                    "file": file,
                    "lines": f"{start}-{end}",
                    "code": text,
                    "score": float(scores[idx])
                })
        return results


class CodeTester:
    def __init__(self, timeout_sec: int = 3):
        self.timeout_sec = timeout_sec

    def test_js_code(self, code: str, test_cases: List[Tuple[str, Any]] = None) -> Dict[str, Any]:
        if not test_cases:
            return {"success": True, "passed": 0, "total": 0, "error_type": None, "error_message": ""}

        # Создаем JS скрипт для тестирования
        js_script = f"""
{code}

// Тесты
try {{
    let passed = 0;
    const total = {len(test_cases)};

    {"".join([
        f'''
        try {{
            const result = eval({tc[0]!r});
            const expected = {json.dumps(tc[1])};
            if (JSON.stringify(result) === JSON.stringify(expected)) {{
                passed++;
            }} else {{
                console.error(`Test failed: got ${{JSON.stringify(result)}}, expected ${{JSON.stringify(expected)}}`);
                process.exit(1);
            }}
        }} catch (e) {{
            console.error(`Test error: ${{e.message}}`);
            process.exit(1);
        }}
        ''' for i, tc in enumerate(test_cases)
    ])}

    console.log('ALL_TESTS_PASSED');
}} catch (e) {{
    console.error(e.message);
    process.exit(1);
}}
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(js_script)
            f.flush()
            try:
                res = subprocess.run(
                    ["node", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec
                )
                success = res.returncode == 0 and "ALL_TESTS_PASSED" in res.stdout
                error_msg = ""
                if not success:
                    out_err = (res.stdout + res.stderr).strip()
                    if "Test" in out_err:
                        error_msg = out_err.split("Test", 1)[-1].split("\n")[0]
                    else:
                        error_msg = out_err[:300]
                return {
                    "success": success,
                    "passed": len(test_cases) if success else 0,
                    "total": len(test_cases),
                    "error_type": None if success else "TestFailed",
                    "error_message": error_msg,
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "error_type": "Timeout", "error_message": f"Timeout after {self.timeout_sec}s"}
            except FileNotFoundError:
                return {"success": False, "error_type": "NoRuntime", "error_message": "Node.js not found"}
            except Exception as e:
                return {"success": False, "error_type": "Runtime", "error_message": str(e)}
            finally:
                os.unlink(f.name)


class CodeDataset(Dataset):
    def __init__(self, codebase_path: str, tokenizer: GelbTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for ext in ["*.js", "*.ts"]:
            for file_path in Path(codebase_path).rglob(ext):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 0:
                        tokens = tokenizer.encode(content)
                        for i in range(0, len(tokens), max_length):
                            chunk = tokens[i:i + max_length]
                            if len(chunk) == max_length:
                                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class GelbEngine:
    def __init__(
        self,
        target_model: GelbModel,
        draft_model: Optional[GelbModel] = None,
        tokenizer: Optional[GelbTokenizer] = None,
        device: str = "cuda",
        codebase_path: Optional[str] = None,
    ):
        self.target = target_model.eval().to(device)
        self.draft = draft_model.eval().to(device) if draft_model else None
        self.tokenizer = tokenizer
        self.device = device
        self.k = 4
        self.code_tester = CodeTester(timeout_sec=5)
        self.retriever = None

        if tokenizer and target_model.config.use_retrieval and codebase_path:
            try:
                self.retriever = CodeRetriever(codebase_path, top_k=target_model.config.retrieval_top_k)
            except Exception as e:
                warnings.warn(f"RAG init failed: {e}")

    def _augment_prompt(self, prompt: str) -> str:
        if not self.retriever:
            return prompt
        
        retrieved = self.retriever.retrieve(prompt)
        if not retrieved:
            return prompt

        context = "\n".join([
            f"// {item['file']}:{item['lines']}\n{item['code']}"
            for item in retrieved
        ])
        return f"<|retrieved_context|>\n{context}\n<|endofcontext|>\n\n{prompt}"

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> Tensor:
        if self.draft is not None and self.tokenizer and self.target.config.use_speculative_decoding:
            return self._speculative_generate(input_ids, max_new_tokens, temperature, top_p)
        else:
            return self._generate_default(input_ids, max_new_tokens, temperature, top_p)

    def _generate_default(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> Tensor:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer required for eos_token_id")
        return self.target.generate(
            input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.tokenizer.eos_token_id
        )

    def _speculative_generate(self, *args, **kwargs):
        return self._generate_default(*args, **kwargs)

    def generate_and_test_code(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        test_cases: Optional[List[Tuple[str, Any]]] = None
    ) -> Dict[str, Any]:
        augmented_prompt = self._augment_prompt(prompt)
        input_ids = self.tokenizer.encode(augmented_prompt).to(self.device)
        output_ids = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        full_gen = self.tokenizer.decode(output_ids[0])
        gen_only = full_gen[len(augmented_prompt):]

        # Проверяем, является ли запрос JS кодом
        is_js_code = "<|lang_js|>" in prompt
        
        if is_js_code:
            test_result = self.code_tester.test_js_code(gen_only, test_cases)
        else:
            # Для не-JS кода просто возвращаем успех
            test_result = {"success": True, "passed": 0, "total": 0, "error_type": None, "error_message": ""}

        retrieved_info = []
        if self.retriever:
            retrieved = self.retriever.retrieve(prompt)
            retrieved_info = [
                {"file": r["file"], "lines": r["lines"], "score": round(r["score"], 3)}
                for r in retrieved
            ]

        return {
            "prompt": prompt,
            "generated_code": gen_only,
            "full_code": prompt + gen_only,
            "retrieved_context": retrieved_info,
            "test_result": test_result
        }


def get_log_probs(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != ignore_index)
    gathered = gathered * mask
    return gathered.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float = 0.1
) -> Tensor:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()


def train_gelb(
    model: GelbModel,
    tokenizer: GelbTokenizer,
    codebase_path: str,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    max_length: int = 1024,
    save_path: str = "gelb_finetuned.pt"
):
    device = next(model.parameters()).device
    
    dataset = CodeDataset(codebase_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    
    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * epochs
    )
    
    model.train()
    total_steps = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids)
            logits = outputs["logits"]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            total_steps += 1
            
            if total_steps % 100 == 0:
                print(f"Step {total_steps}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config.to_dict()
    }, save_path)
    print(f"Model saved to {save_path}")


app = fastapi.FastAPI(title="Gelb API", version="1.0")

class TestCaseItem(BaseModel):
    input: str
    expected: Any

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    stream: bool = False
    test_cases: Optional[List[TestCaseItem]] = None


global_engine: Optional[GelbEngine] = None


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if global_engine is None:
        return {"error": "Model not initialized"}

    test_cases = [(tc.input, tc.expected) for tc in req.test_cases] if req.test_cases else None

    try:
        result = global_engine.generate_and_test_code(
            req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            test_cases=test_cases
        )
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "model": "gelb-1b",
            "choices": [{
                "text": result["generated_code"],
                "full_code": result["full_code"],
                "retrieved_context": result["retrieved_context"],
                "test_result": result["test_result"]
            }]
        }
    except Exception as e:
        return {"error": f"Generation failed: {e}"}


@app.post("/v1/chat/completions")
async def chat_completions(req: CompletionRequest):
    return await completions(req)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebase", type=str, default=None, help="Path to JavaScript codebase for RAG")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--train_data", type=str, default="js_code/", help="Path to JavaScript training data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    config = GelbConfig()

    try:
        model = GelbModel(config)
    except Exception as e:
        print(f"Model init failed: {e}")
        exit(1)

    try:
        tokenizer = GelbTokenizer(config)
    except Exception as e:
        print(f"Tokenizer init failed: {e}")
        exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        print("🚀 Starting training...")
        train_gelb(
            model=model,
            tokenizer=tokenizer,
            codebase_path=args.train_data,
            epochs=args.epochs,
            batch_size=1,
            learning_rate=1e-4
        )
        print("✅ Training completed!")
        return

    try:
        import deepspeed
        ds_config = {
            "zero_optimization": {
                "stage": 2,
                "offload_param": {"device": "none"},
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
                "cpu_offload": False
            },
            "fp16": {"enabled": True},
            "gradient_clipping": 1.0,
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1
        }
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        model = model_engine.module
    except ImportError:
        model_engine = None

    try:
        engine = GelbEngine(
            model,
            tokenizer=tokenizer,
            device=device,
            codebase_path=args.codebase
        )
    except Exception as e:
        print(f"Engine init failed: {e}")
        exit(1)

    global global_engine
    global_engine = engine

    if device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            pass

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
