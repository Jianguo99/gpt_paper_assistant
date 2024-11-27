import time
import random
from typing import Optional, Dict, Any
from litellm import completion
from datetime import datetime

import os

class ContinuousLLMCaller:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        base_retry_delay: float = 2.0,
        max_retry_delay: float = 60.0,
        jitter: float = 0.1,
        max_attempts: Optional[int] = None
    ):
        """
        初始化持续调用LLM的处理器
        
        Args:
            model: 使用的模型名称
            base_retry_delay: 基础重试延迟时间（秒）
            max_retry_delay: 最大重试延迟时间（秒）
            jitter: 随机抖动范围（避免同时重试）
            max_attempts: 最大尝试次数（None表示无限尝试）
        """
        self.model = model
        self.base_retry_delay = max(1.0, base_retry_delay)
        self.max_retry_delay = max(self.base_retry_delay, max_retry_delay)
        self.jitter = min(1.0, max(0.0, jitter))
        self.max_attempts = max_attempts
        self.reset_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self.attempt_count = 0
        self.start_time = None
        self.total_delay = 0
    
    def _calculate_delay(self) -> float:
        """计算下一次重试的延迟时间"""
        delay = min(
            self.max_retry_delay,
            self.base_retry_delay * (2 ** (self.attempt_count - 1))
        )
        
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(self.base_retry_delay, min(delay, self.max_retry_delay))
    
    def _handle_error(self, error: Exception) -> float:
        """处理错误并返回等待时间"""
        error_type = type(error).__name__
        error_message = str(error)
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # 认证错误直接返回None
        if "AuthenticationError" in error_type:
            print(f"[{current_time}] 认证错误，跳过此次请求")
            return -1  # 特殊标记，表示需要直接返回None
        
        # 处理其他错误
        if "RateLimitError" in error_type:
            wait_time = self._calculate_delay() + 10
            print(f"[{current_time}] 遇到频率限制 ({error_message})")
        elif "TimeoutError" in error_type:
            wait_time = self._calculate_delay()
            print(f"[{current_time}] 请求超时 ({error_message})")
        elif "APIError" in error_type or "APIConnectionError" in error_type:
            wait_time = self._calculate_delay() + 5
            print(f"[{current_time}] API错误 ({error_message})")
        else:
            wait_time = self._calculate_delay()
            print(f"[{current_time}] 未知错误 {error_type} ({error_message})")
        
        # 检查是否达到最大尝试次数
        if self.max_attempts and self.attempt_count >= self.max_attempts:
            print(f"达到最大尝试次数 {self.max_attempts}，跳过此次请求")
            return -1  # 特殊标记，表示需要直接返回None
        
        print(f"第 {self.attempt_count} 次尝试失败，等待 {wait_time:.1f} 秒后重试...")
        return wait_time
    
    def get_completion(self, prompt: str, system_instruction: str, **kwargs) -> Optional[str]:
        """
        持续尝试获取LLM的回复，直到成功或达到最大尝试次数
        
        Args:
            prompt: 输入的提示词
            **kwargs: 传递给completion的其他参数
        
        Returns:
            Optional[str]: 成功时返回模型的文本回复，失败时返回None
        """
        self.reset_stats()
        self.start_time = time.time()
        
        while True:
            self.attempt_count += 1

            try:
                response = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt, "system_instruction": system_instruction}],
                    timeout=30,
                    **kwargs
                )
                
                total_time = time.time() - self.start_time
                if self.attempt_count > 1:
                    print(f"\n成功获取回复! 总耗时: {total_time:.1f}秒，尝试次数: {self.attempt_count}")
                
                return response
                
            except Exception as e:
                wait_time = self._handle_error(e)
                
                # 检查是否需要直接返回None
                if wait_time == -1:
                    return None
                
                self.total_delay += wait_time
                time.sleep(wait_time)
                continue
            
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch         
class LocalLLMCaller:
    def __init__(self, model_name, max_new_tokens=1024, max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def get_completion(self, prompt, temperature=1.0, seed=0):
        # 设置随机种子
        torch.manual_seed(seed)
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 生成输出
        outputs = self.model.generate(
            inputs.input_ids,
            # max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            do_sample=True,
            
        )
        
        # 解码输出
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion

    def call_model(self, full_prompt):
        # 调用 get_completion 方法来获取模型的输出
        return self.get_completion(full_prompt)

def main():
    # 使用示例
    caller = ContinuousLLMCaller(
        model="claude-3-5-sonnet-20241022",
        base_retry_delay=2.0,
        max_retry_delay=60.0,
        jitter=0.1,
        max_attempts=3  # 设置最大尝试次数
    )
    
    # 测试多个问题
    prompts = [
        "你好，请介绍一下自己。",
        "什么是人工智能？",
        "请解释一下量子计算。"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n=== 问题 {i} ===")
        print(f"提示词: {prompt}\n")
        
        response = caller.get_completion(prompt)
        
        if response is None:
            print(f"问题 {i} 跳过，继续下一个问题")
        else:
            print(f"\n回复 {i}:")
            print(response)

if __name__ == "__main__":
    main()