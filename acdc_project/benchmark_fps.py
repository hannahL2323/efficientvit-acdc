"""
EfficientViT 分割模型 FPS 测试脚本
用法: python benchmark_fps.py
"""

import torch
import time
import sys
import os
import argparse
from typing import Optional, Tuple

# 添加项目根目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.seg_model_zoo import create_efficientvit_seg_model


class FPSBenchmark:
    """EfficientViT FPS 测试类"""
    
    def __init__(
        self,
        model_name: str = "efficientvit-seg-b1-cityscapes",
        weight_url: Optional[str] = "./checkpoints/efficientvit_seg_b1_cityscapes.pt",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.weight_url = weight_url
        self.device = device
        self.model = None
        
    def load_model(self):
        """加载模型"""
        print(f"Loading model: {self.model_name}")
        if self.weight_url:
            self.model = create_efficientvit_seg_model(self.model_name, weight_url=self.weight_url)
        else:
            self.model = create_efficientvit_seg_model(self.model_name, pretrained=True)
        
        self.model.eval()
        self.model.to(self.device)
        print("Model loaded successfully!")
        return self.model
    
    def benchmark(
        self,
        input_size: Tuple[int, int] = (1024, 2048),
        batch_size: int = 1,
        num_iterations: int = 100,
        warmup_iterations: int = 20,
    ) -> dict:
        """
        测试 FPS
        
        Args:
            input_size: (height, width)
            batch_size: 批次大小
            num_iterations: 测试迭代次数
            warmup_iterations: 预热迭代次数
        
        Returns:
            dict: 包含测试结果
        """
        if self.model is None:
            self.load_model()
        
        print(f"\n{'='*60}")
        print(f"Benchmark Configuration:")
        print(f"  Model: {self.model_name}")
        print(f"  Input size: {input_size[0]}x{input_size[1]}")
        print(f"  Batch size: {batch_size}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Warmup: {warmup_iterations}")
        print(f"{'='*60}")
        
        # 创建随机输入
        dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(self.device)
        
        # 预热
        print("\nWarming up...")
        for _ in range(warmup_iterations):
            with torch.inference_mode():
                _ = self.model(dummy_input)
        
        # 同步 GPU
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # 开始测速
        print("Running benchmark...")
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.inference_mode():
                _ = self.model(dummy_input)
        
        # 同步 GPU
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # 计算结果
        total_time = end_time - start_time
        avg_latency_per_batch = total_time / num_iterations
        avg_latency_per_image = total_time / (num_iterations * batch_size)
        fps = (num_iterations * batch_size) / total_time
        
        # GPU 显存统计
        if self.device == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            current_memory_mb = torch.cuda.memory_allocated() / 1024**2
        else:
            peak_memory_mb = 0
            current_memory_mb = 0
        
        results = {
            "model": self.model_name,
            "input_size": input_size,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "total_time": total_time,
            "avg_latency_per_batch_ms": avg_latency_per_batch * 1000,
            "avg_latency_per_image_ms": avg_latency_per_image * 1000,
            "fps": fps,
            "peak_memory_mb": peak_memory_mb,
            "current_memory_mb": current_memory_mb,
        }
        
        # 打印结果
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: dict):
        """打印测试结果"""
        print(f"\n{'='*60}")
        print("Benchmark Results:")
        print(f"{'='*60}")
        print(f"  Total time: {results['total_time']:.4f} sec")
        print(f"  Avg latency per batch: {results['avg_latency_per_batch_ms']:.2f} ms")
        print(f"  Avg latency per image: {results['avg_latency_per_image_ms']:.2f} ms")
        print(f"  📊 FPS: {results['fps']:.2f} images/sec")
        print(f"  💾 Peak GPU memory: {results['peak_memory_mb']:.2f} MB")
        print(f"  💾 Current GPU memory: {results['current_memory_mb']:.2f} MB")
        print(f"{'='*60}\n")
    
    def run_all_benchmarks(self):
        """运行所有预设的测试配置"""
        print(f"\n{'#'*60}")
        print(f"# Running Full FPS Benchmark for {self.model_name}")
        print(f"{'#'*60}")
        
        # 加载模型
        self.load_model()
        
        # 测试配置列表: (input_size, batch_size, iterations)
        configs = [
            # 低分辨率测试
            ((512, 1024), 1, 100),
            ((768, 1536), 1, 100),
            # 标准分辨率测试 (Cityscapes 常用)
            ((1024, 2048), 1, 100),
            ((1024, 2048), 2, 50),
            ((1024, 2048), 4, 25),
            # 高分辨率测试
            ((1536, 3072), 1, 50),
        ]
        
        all_results = []
        for input_size, batch_size, iterations in configs:
            try:
                result = self.benchmark(
                    input_size=input_size,
                    batch_size=batch_size,
                    num_iterations=iterations,
                )
                all_results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"❌ OOM for size {input_size}, batch {batch_size} - skipping")
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        # 打印汇总
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: list):
        """打印汇总结果"""
        print(f"\n{'#'*60}")
        print(f"# Summary of All Benchmarks")
        print(f"{'#'*60}")
        
        print(f"\n{'Input Size':<20} {'Batch':<8} {'FPS':<12} {'Latency/Image':<18} {'Memory':<12}")
        print("-" * 70)
        
        for r in results:
            size_str = f"{r['input_size'][0]}x{r['input_size'][1]}"
            latency_str = f"{r['avg_latency_per_image_ms']:.2f} ms"
            memory_str = f"{r['peak_memory_mb']:.0f} MB"
            print(f"{size_str:<20} {r['batch_size']:<8} {r['fps']:<12.2f} {latency_str:<18} {memory_str:<12}")
        
        print(f"{'#'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark EfficientViT FPS")
    parser.add_argument(
        "--model", 
        type=str, 
        default="efficientvit-seg-b1-cityscapes",
        choices=[
            "efficientvit-seg-b0-cityscapes",
            "efficientvit-seg-b1-cityscapes",
            "efficientvit-seg-b2-cityscapes",
            "efficientvit-seg-b3-cityscapes",
            "efficientvit-seg-l1-cityscapes",
            "efficientvit-seg-l2-cityscapes",
        ],
        help="Model name"
    )
    parser.add_argument(
        "--weight_url",
        type=str,
        default="./checkpoints/efficientvit_seg_b1_cityscapes.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU to use (e.g., '0' or '1')"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick benchmark (standard resolution only)"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Custom input size (e.g., --input_size 1024 2048)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for custom benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for custom benchmark"
    )
    
    args = parser.parse_args()
    
    # 设置 GPU
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 创建测试器
    benchmark = FPSBenchmark(
        model_name=args.model,
        weight_url=args.weight_url,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # 运行测试
    if args.quick:
        # 快速测试：只跑标准分辨率
        benchmark.load_model()
        benchmark.benchmark(
            input_size=(1024, 2048),
            batch_size=args.batch_size,
            num_iterations=args.iterations,
        )
    elif args.input_size is not None:
        # 自定义输入尺寸
        benchmark.load_model()
        benchmark.benchmark(
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            num_iterations=args.iterations,
        )
    else:
        # 运行所有预设配置
        benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()