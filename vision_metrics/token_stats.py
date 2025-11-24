"""
Token Statistics Metrics

이 모듈은 Vision Tokenizer의 토큰 통계를 계산합니다.

지원 메트릭:
1. BPP (Bits Per Pixel)
   - 픽셀당 필요한 비트 수
   - 압축 효율성 측정
   - 수식: (num_tokens * log2(codebook_size)) / (H * W)

2. Codebook Entropy
   - 토큰 분포의 엔트로피
   - 높을수록 codebook을 고르게 사용
   - 수식: -Σ p(i) * log2(p(i))

3. Codebook Usage
   - 실제 사용된 codebook entries 비율
   - 높을수록 효율적인 codebook 활용

4. Token Diversity
   - 토큰 시퀀스의 다양성 측정
   - Unique tokens, repetition rate 등

사용법:
    from metrics.token_stats import calculate_bpp, calculate_codebook_usage
    
    bpp = calculate_bpp(tokens, image_size, codebook_size)
    usage = calculate_codebook_usage(tokens, codebook_size)
"""

from typing import List, Dict, Any, Union
import torch
import numpy as np
from collections import Counter
import math


def calculate_bpp(tokens: torch.Tensor,
                  image_size: tuple,
                  codebook_size: int) -> float:
    """
    BPP (Bits Per Pixel) 계산
    
    Args:
        tokens: 토큰 텐서, shape (B, N) 또는 (B, H, W)
        image_size: 원본 이미지 크기 (H, W)
        codebook_size: codebook 크기
    
    Returns:
        float: BPP 값
    
    Example:
        >>> bpp = calculate_bpp(tokens, (256, 256), codebook_size=8192)
        >>> print(f"BPP: {bpp:.4f}")
    """
    # Calculate bits per token
    bits_per_token = math.log2(codebook_size)
    
    # Calculate number of tokens
    if tokens.ndim == 3:  # (B, H, W)
        num_tokens = tokens.shape[1] * tokens.shape[2]
    else:  # (B, N)
        num_tokens = tokens.shape[1]
    
    # Calculate total bits
    total_bits = num_tokens * bits_per_token
    
    # Calculate total pixels
    total_pixels = image_size[0] * image_size[1]
    
    # BPP
    bpp = total_bits / total_pixels
    return bpp


def calculate_codebook_entropy(tokens: torch.Tensor,
                               codebook_size: int) -> float:
    """
    Codebook 사용의 엔트로피 계산
    
    Args:
        tokens: 토큰 텐서, shape (B, N) 또는 (B, H, W)
        codebook_size: codebook 크기
    
    Returns:
        float: 엔트로피 값 (bits)
    
    Example:
        >>> entropy = calculate_codebook_entropy(tokens, codebook_size=8192)
        >>> max_entropy = math.log2(8192)
        >>> print(f"Entropy: {entropy:.2f} / {max_entropy:.2f} (max)")
    """
    # Flatten tokens
    tokens_flat = tokens.flatten()
    
    # Count token frequencies
    token_counts = torch.bincount(tokens_flat, minlength=codebook_size)
    token_probs = token_counts.float() / token_counts.sum()
    
    # Calculate entropy
    # H = -Σ p(i) * log2(p(i))
    entropy = 0.0
    for prob in token_probs:
        if prob > 0:
            entropy -= prob.item() * math.log2(prob.item())
    
    return entropy


def calculate_codebook_usage(tokens: torch.Tensor,
                             codebook_size: int) -> Dict[str, Any]:
    """
    Codebook 사용 통계 계산
    
    Args:
        tokens: 토큰 텐서, shape (B, N) 또는 (B, H, W)
        codebook_size: codebook 크기
    
    Returns:
        Dict with:
            - 'unique_codes': int - 사용된 unique code 개수
            - 'usage_ratio': float - 사용 비율 (0~1)
            - 'entropy': float - 엔트로피
            - 'max_entropy': float - 최대 엔트로피
            - 'perplexity': float - perplexity (2^entropy)
            - 'top_k_usage': Dict - 가장 많이 사용된 k개 코드
    
    Example:
        >>> usage = calculate_codebook_usage(tokens, codebook_size=8192)
        >>> print(f"Using {usage['unique_codes']} / {codebook_size} codes")
        >>> print(f"Usage ratio: {usage['usage_ratio']:.2%}")
    """
    tokens_flat = tokens.flatten().cpu().numpy()
    
    # Count unique codes
    unique_codes = np.unique(tokens_flat)
    num_unique = len(unique_codes)
    usage_ratio = num_unique / codebook_size
    
    # Calculate entropy
    entropy = calculate_codebook_entropy(tokens, codebook_size)
    max_entropy = math.log2(codebook_size)
    perplexity = 2 ** entropy
    
    # Top-k most used codes
    counter = Counter(tokens_flat)
    top_k = dict(counter.most_common(10))
    
    return {
        'unique_codes': int(num_unique),
        'usage_ratio': float(usage_ratio),
        'entropy': float(entropy),
        'max_entropy': float(max_entropy),
        'perplexity': float(perplexity),
        'top_k_usage': top_k
    }


def calculate_compression_ratio(original_size: tuple,
                                token_count: int,
                                codebook_size: int,
                                bits_per_channel: int = 8) -> float:
    """
    압축률 계산
    
    Args:
        original_size: 원본 이미지 크기 (C, H, W)
        token_count: 토큰 개수
        codebook_size: codebook 크기
        bits_per_channel: 채널당 비트 수 (일반적으로 8)
    
    Returns:
        float: 압축률 (원본 대비 몇 배 압축)
    
    Example:
        >>> ratio = calculate_compression_ratio((3, 256, 256), 256, 8192)
        >>> print(f"Compression ratio: {ratio:.2f}x")
    """
    # Original size in bits
    original_bits = original_size[0] * original_size[1] * original_size[2] * bits_per_channel
    
    # Compressed size in bits
    bits_per_token = math.log2(codebook_size)
    compressed_bits = token_count * bits_per_token
    
    # Compression ratio
    ratio = original_bits / compressed_bits
    return ratio


def calculate_token_diversity(tokens: torch.Tensor) -> Dict[str, Any]:
    """
    토큰 시퀀스의 다양성 측정
    
    Args:
        tokens: 토큰 텐서, shape (B, N)
    
    Returns:
        Dict with:
            - 'unique_ratio': float - unique tokens 비율
            - 'repetition_rate': float - 반복 비율
            - 'avg_run_length': float - 평균 연속 반복 길이
    
    Example:
        >>> diversity = calculate_token_diversity(tokens)
        >>> print(f"Unique ratio: {diversity['unique_ratio']:.2%}")
    """
    results = []
    
    for batch_idx in range(tokens.shape[0]):
        seq = tokens[batch_idx].flatten().cpu().numpy()
        
        # Unique ratio
        unique_ratio = len(np.unique(seq)) / len(seq)
        
        # Repetition rate
        repetitions = 0
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                repetitions += 1
        repetition_rate = repetitions / (len(seq) - 1) if len(seq) > 1 else 0
        
        # Average run length
        run_lengths = []
        current_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        avg_run_length = np.mean(run_lengths)
        
        results.append({
            'unique_ratio': unique_ratio,
            'repetition_rate': repetition_rate,
            'avg_run_length': avg_run_length
        })
    
    # Average across batch
    return {
        'unique_ratio': float(np.mean([r['unique_ratio'] for r in results])),
        'repetition_rate': float(np.mean([r['repetition_rate'] for r in results])),
        'avg_run_length': float(np.mean([r['avg_run_length'] for r in results]))
    }


def analyze_spatial_distribution(tokens: torch.Tensor,
                                 latent_shape: tuple) -> Dict[str, Any]:
    """
    토큰의 공간적 분포 분석
    
    Args:
        tokens: 토큰 텐서, shape (B, H, W)
        latent_shape: latent grid 크기 (H, W)
    
    Returns:
        Dict with spatial statistics
    
    Example:
        >>> spatial_stats = analyze_spatial_distribution(tokens, (16, 16))
    """
    # TODO: 구현
    # - 공간적 자기상관 (spatial autocorrelation)
    # - 지역별 엔트로피 (local entropy)
    # - 패턴 반복성 (pattern repetition)
    pass


def calculate_all_token_stats(tokens: torch.Tensor,
                              image_size: tuple,
                              codebook_size: int,
                              latent_shape: tuple = None) -> Dict[str, Any]:
    """
    모든 토큰 통계를 한 번에 계산
    
    Args:
        tokens: 토큰 텐서
        image_size: 원본 이미지 크기 (H, W)
        codebook_size: codebook 크기
        latent_shape: latent grid 크기 (H, W)
    
    Returns:
        Dict: 모든 토큰 통계
    
    Example:
        >>> stats = calculate_all_token_stats(tokens, (256, 256), 8192)
        >>> print(f"BPP: {stats['bpp']:.4f}")
        >>> print(f"Codebook usage: {stats['codebook_usage']['usage_ratio']:.2%}")
    """
    stats = {}
    
    try:
        stats['bpp'] = calculate_bpp(tokens, image_size, codebook_size)
    except Exception as e:
        print(f"Failed to calculate BPP: {e}")
        stats['bpp'] = None
    
    try:
        stats['codebook_usage'] = calculate_codebook_usage(tokens, codebook_size)
    except Exception as e:
        print(f"Failed to calculate codebook usage: {e}")
        stats['codebook_usage'] = None
    
    try:
        stats['token_diversity'] = calculate_token_diversity(tokens)
    except Exception as e:
        print(f"Failed to calculate token diversity: {e}")
        stats['token_diversity'] = None
    
    if latent_shape is not None:
        try:
            stats['spatial_distribution'] = analyze_spatial_distribution(tokens, latent_shape)
        except Exception as e:
            print(f"Failed to analyze spatial distribution: {e}")
            stats['spatial_distribution'] = None
    
    return stats

