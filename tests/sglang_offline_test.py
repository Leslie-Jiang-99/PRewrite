import sglang as sgl
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge
import asyncio


async def main_async():
    print("正在加载模型...")
    generate_model = sgl.Engine(
        model_path="/root/group-shared/jrc/base_models/Qwen3-32B",
        tp_size=8,
        enable_custom_logit_processor=True,
        enable_p2p_check=True,
        mem_fraction_static=0.3,
        max_prefill_tokens=131072,
        log_level="info",
        trust_remote_code=True,
    )
    print("模型加载完成！\n")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    print("开始生成...")
    # 使用异步生成
    outputs = await generate_model.async_generate(prompts, sampling_params)
    
    for prompt, output in zip(prompts, outputs):
        print("=" * 50)
        print(f"Prompt: {prompt}")
        print(f"Generated: {output['text']}")
    
    print("\n正在关闭模型...")
    generate_model.shutdown()
    print("完成！")


def main():
    # 运行异步主函数
    asyncio.run(main_async())


if __name__ == '__main__':
    main()