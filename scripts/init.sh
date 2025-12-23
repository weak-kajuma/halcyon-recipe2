pip install -e .
pip install liger-kernel transformers -U

pip install pybind11
pip install --no-build-isolation transformer_engine[pytorch]

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

cd ../

pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.15.0
pip install "flash-attn==2.8.3" --no-build-isolation