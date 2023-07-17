# HF-For-RWKVRaven-Alpaca
将RWKV Raven/Pile/PilePlus系列模型由原生pth转为HF格式(这三种模型的词表及tokenizer一致，推荐V11或V12的Raven版本)，并进行Alpaca全量微调。<br>


环境：WIN10+Torch1.31+Cuda11.6 <br>

代码说明：<br>
configuration_rwkv.py：RWKV模型的配置<br>
convert_rwkv_checkpoint_to_hf.py：RWKV的原生pth格式转为HF格式<br>
generate.py：使用HF的RWKV模型架设服务<br>
hello.py：测试HF的RWKV模型<br>
modeling_rwkv.py：RWKV模型的网络结构<br>
alpacatrain.py：使用test.json的alpaca全量微调模型<br>
alpacatest.py：测试alpaca全量微调模型<br>

HF开源地址：<br>
https://huggingface.co/StarRing2022/RWKV-430M-Pile-Alpaca/<br>
https://huggingface.co/StarRing2022/RWKV-4-Raven-3B-v11-zh/<br>
