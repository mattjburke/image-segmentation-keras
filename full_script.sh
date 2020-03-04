# module install pip3
# echo "pip installed"
# python3 setup.py install --user
# echo "setup complete"
# echo "cuda version is -------- before beginning setup.py install --user"
# nvcc --version

# python3 setup.py install --user

# echo "cuda version is -------- after setup.py and before tensorflow upgrade"
# nvcc --version

# pip3 install --user --upgrade tensorflow-gpu --upgrade-strategy eager

# echo "tensorflow version is"
# tensorflow --version  # not a crrect call

echo "cuda version is -------- after tensorflow upgrade in previous run"
nvcc --version

python3 train_unet.py
python3 train_pspnet.py
python3 train_segnet.py

echo "trin_base complete"

