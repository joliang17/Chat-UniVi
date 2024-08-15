
pip install --upgrade pip
pip install -e .
pip install ninja  # If you only intend to perform inference, there's no need to install ```ninja```.
pip install flash-attn --no-build-isolation  # If you only intend to perform inference, there's no need to install ```flash-attn```.
pip uninstall -y byted-wandb
pip uninstall -y wandb
pip install byted-wandb
pip install httpx==0.23.0
pip install openai